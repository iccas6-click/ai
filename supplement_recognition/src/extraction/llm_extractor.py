from __future__ import annotations

import json
import os

import anthropic

from src.ocr.reader import OcrResult
from src.schema.result import FunctionalIngredient, SupplementProduct

_SYSTEM_PROMPT = """\
You are a structured data extractor for Korean health supplement labels.
Given OCR text from a label image, extract:
- product_name: exact product name as written
- functional_ingredients: list of functional ingredients with name, amount, unit, and the exact evidence_text from the OCR that supports each field

Rules:
- Only extract information explicitly present in the OCR text. Do not infer or guess.
- If amount or unit is not found, set them to null.
- evidence_text must be a verbatim substring from the OCR text.
- Respond with JSON only, no commentary.

Output schema:
{
  "product_name": "string",
  "functional_ingredients": [
    {
      "name": "string",
      "amount": number | null,
      "unit": "string | null",
      "evidence_text": "string"
    }
  ]
}
"""


def extract_from_ocr(ocr_result: OcrResult) -> SupplementProduct:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": ocr_result.full_text}],
    )

    raw = message.content[0].text.strip()
    data = json.loads(raw)

    ingredients = [
        FunctionalIngredient(
            name=item["name"],
            amount=item.get("amount"),
            unit=item.get("unit"),
            evidence_text=item["evidence_text"],
        )
        for item in data.get("functional_ingredients", [])
    ]

    return SupplementProduct(
        product_name=data["product_name"],
        functional_ingredients=ingredients,
        confidence=0.0,  # matching 단계에서 갱신
    )
