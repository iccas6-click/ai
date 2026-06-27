from __future__ import annotations

import io
import json
import re

import numpy as np
from PIL import Image

from ..schemas import VisionObservation, VisionProductCandidate


SINGLE_CROP_PROMPT = """
You inspect one cropped pill image for a Korean pill recognition app.
Return only strict JSON.

Primary goal:
- Identify the most likely medication/product names, active ingredients, and key caution points.
- The app needs product_name, ingredient, and caution_points.
- Put the best guesses first in candidates.

Rules:
- Do not refuse just because this is a medication image.
- Do not give dosage instructions.
- caution_points should be short general safety notes, not personalized medical advice.
- If the pill is unclear, still return your best visual candidates with lower confidence.
- If no pill is visible, return empty candidates and confidence 0.
- ingredient should be the active ingredient name in Korean when you can infer it.
- If you are unsure of ingredient, use null rather than inventing a precise ingredient.
- If product_name is a guess, keep confidence low and mention uncertainty in notes.

Schema:
{
  "candidates": [
    {
      "product_name": string,
      "ingredient": string|null,
      "caution_points": string[],
      "confidence": number
    }
  ],
  "confidence": number,
  "notes": string|null
}
"""


BATCH_PROMPT = """
You inspect multiple cropped pill images for a Korean pill recognition app.
Return only strict JSON.

Each following image is labeled as "pill_index N".
Return one result for every pill_index in the same order.

Primary goal:
- Identify the most likely medication/product names, active ingredients, and key caution points for each crop.
- The app needs product_name, ingredient, and caution_points.
- Put the best guesses first in candidates.

Rules:
- Do not refuse just because these are medication images.
- Do not give dosage instructions.
- caution_points should be short general safety notes, not personalized medical advice.
- If a pill is unclear, still return your best visual candidates with lower confidence.
- If no pill is visible for a pill_index, return empty candidates and confidence 0.
- ingredient should be the active ingredient name in Korean when you can infer it.
- If you are unsure of ingredient, use null rather than inventing a precise ingredient.
- If product_name is a guess, keep confidence low and mention uncertainty in notes.

Schema:
{
  "results": [
    {
      "pill_index": number,
      "candidates": [
        {
          "product_name": string,
          "ingredient": string|null,
          "caution_points": string[],
          "confidence": number
        }
      ],
      "confidence": number,
      "notes": string|null
    }
  ]
}
"""


class GeminiVisionProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as error:
            raise RuntimeError(
                "Gemini provider requires the google-genai package. "
                "Install it or use PILL_VISION_PROVIDER=local."
            ) from error
        self.client = genai.Client(api_key=api_key)
        self.types = types
        self.model = model

    def inspect_crop(self, crop_rgb: np.ndarray) -> VisionObservation:
        return self.inspect_crops([crop_rgb])[0]

    def inspect_crops(self, crops_rgb: list[np.ndarray]) -> list[VisionObservation]:
        if not crops_rgb:
            return []
        if len(crops_rgb) == 1:
            return [self._inspect_single_crop(crops_rgb[0])]

        contents = [BATCH_PROMPT]
        for index, crop_rgb in enumerate(crops_rgb, start=1):
            contents.append(f"pill_index {index}")
            contents.append(
                self.types.Part.from_bytes(
                    data=encode_crop_jpeg(crop_rgb),
                    mime_type="image/jpeg",
                )
            )
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
        )
        payload = parse_json_response(getattr(response, "text", "") or "")
        return parse_batch_observations(payload, len(crops_rgb), self.name, self.model)

    def _inspect_single_crop(self, crop_rgb: np.ndarray) -> VisionObservation:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                SINGLE_CROP_PROMPT,
                self.types.Part.from_bytes(
                    data=encode_crop_jpeg(crop_rgb),
                    mime_type="image/jpeg",
                ),
            ],
        )
        payload = parse_json_response(getattr(response, "text", "") or "")
        return observation_from_payload(payload, self.name, self.model)


def encode_crop_jpeg(crop_rgb: np.ndarray) -> bytes:
    image = Image.fromarray(np.ascontiguousarray(crop_rgb)).convert("RGB")
    image.thumbnail((512, 512))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def observation_from_payload(
    payload: dict,
    provider_name: str,
    model: str,
) -> VisionObservation:
    product_candidates = parse_product_candidates(payload)
    return VisionObservation(
        product_candidates=product_candidates,
        possible_product_names=[candidate.product_name for candidate in product_candidates],
        confidence=to_float(payload.get("confidence")),
        notes=clean(payload.get("notes")),
        raw={"provider": provider_name, "model": model, "response": payload},
    )


def parse_batch_observations(
    payload: dict,
    expected_count: int,
    provider_name: str,
    model: str,
) -> list[VisionObservation]:
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        rows = []
    by_index = {}
    for order, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        pill_index = to_int(row.get("pill_index")) or order
        by_index[pill_index] = row
    observations = []
    for pill_index in range(1, expected_count + 1):
        row = by_index.get(pill_index)
        if row is None:
            observations.append(
                VisionObservation(
                    confidence=0.0,
                    notes="Gemini batch response did not include this pill_index.",
                    raw={
                        "provider": provider_name,
                        "model": model,
                        "response": payload,
                        "missing_pill_index": pill_index,
                    },
                )
            )
            continue
        observations.append(observation_from_payload(row, provider_name, model))
    return observations


def parse_json_response(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return {"notes": text}
    return value if isinstance(value, dict) else {"notes": text}


def clean(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    return text


def parse_product_candidates(payload: dict) -> list[VisionProductCandidate]:
    rows = payload.get("candidates", [])
    if not isinstance(rows, list):
        rows = []
    candidates = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        product_name = clean(row.get("product_name"))
        if not product_name:
            continue
        normalized = "".join(product_name.split()).upper()
        if normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(
            VisionProductCandidate(
                product_name=product_name,
                ingredient=clean(row.get("ingredient")),
                caution_points=clean_list(row.get("caution_points")),
                confidence=to_float(row.get("confidence")),
            )
        )
    if candidates:
        return candidates

    legacy_names = payload.get("possible_product_names", [])
    if not isinstance(legacy_names, list):
        return []
    for name in legacy_names:
        product_name = clean(name)
        if not product_name:
            continue
        normalized = "".join(product_name.split()).upper()
        if normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(VisionProductCandidate(product_name=product_name))
    return candidates


def clean_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := clean(item))]


def to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
