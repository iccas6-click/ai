from __future__ import annotations

import io
import json
import re

import numpy as np
from PIL import Image

from ..schemas import VisionObservation, VisionProductCandidate


PROMPT = """
You inspect one cropped pill image for a Korean pill recognition app.
Return only strict JSON.

Primary goal:
- Identify the most likely medication/product names and active ingredients.
- The app only needs product_name and ingredient.
- Put the best guesses first in candidates.

Rules:
- Do not refuse just because this is a medication image.
- Do not give medical advice, dosage advice, or safety advice.
- If the pill is unclear, still return your best visual candidates with lower confidence.
- If no pill is visible, return empty candidates and confidence 0.
- ingredient should be the active ingredient name in Korean when you can infer it.
- If you are unsure of ingredient, use null rather than inventing a precise ingredient.
- Do not return shape, color, imprint, dosage instructions, warnings, or explanations outside notes.

Schema:
{
  "candidates": [
    {
      "product_name": string,
      "ingredient": string|null,
      "confidence": number
    }
  ],
  "confidence": number,
  "notes": string|null
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
        image = Image.fromarray(np.ascontiguousarray(crop_rgb)).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92)
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                PROMPT,
                self.types.Part.from_bytes(
                    data=buffer.getvalue(),
                    mime_type="image/jpeg",
                ),
            ],
        )
        payload = parse_json_response(getattr(response, "text", "") or "")
        product_candidates = parse_product_candidates(payload)
        return VisionObservation(
            product_candidates=product_candidates,
            possible_product_names=[candidate.product_name for candidate in product_candidates],
            confidence=to_float(payload.get("confidence")),
            notes=clean(payload.get("notes")),
            raw={"provider": self.name, "model": self.model, "response": payload},
        )


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


def to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
