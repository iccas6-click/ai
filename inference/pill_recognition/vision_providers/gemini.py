from __future__ import annotations

import io
import json
import re

import numpy as np
from PIL import Image

from ..schemas import VisionObservation


PROMPT = """
You inspect one cropped pill image for a Korean pill recognition app.
Return only strict JSON.

Primary goal:
- Identify the most likely medication/product names from the image.
- Use imprint, shape, color, visible text, and your visual knowledge.
- Put the best guesses first in possible_product_names.

Rules:
- Do not refuse just because this is a medication image.
- Do not give medical advice, dosage advice, or safety advice.
- If the pill is unclear, still return your best visual candidates with lower confidence.
- If no pill is visible, return empty possible_product_names and confidence 0.
- Extract imprints exactly as seen. If there is a split line, write compact text such as "W2" when visually appropriate.
- Use null only for fields that are not visible.

Schema:
{
  "imprint_front": string|null,
  "imprint_back": string|null,
  "shape": "원형"|"타원형"|"장방형"|"반원형"|"삼각형"|"사각형"|"기타"|null,
  "color": "하양"|"노랑"|"주황"|"분홍"|"빨강"|"갈색"|"초록"|"파랑"|"보라"|"회색"|"검정"|"투명"|null,
  "text": string|null,
  "possible_product_names": string[],
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
        return VisionObservation(
            imprint_front=clean(payload.get("imprint_front")),
            imprint_back=clean(payload.get("imprint_back")),
            shape=clean(payload.get("shape")),
            color=clean(payload.get("color")),
            text=clean(payload.get("text")),
            possible_product_names=[
                str(value).strip()
                for value in payload.get("possible_product_names", [])
                if str(value).strip()
            ],
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


def to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
