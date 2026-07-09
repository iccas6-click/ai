"""
DB 미등재 건강기능식품의 성분을 Gemini에게 식약처 기준으로 추출.

CBNUAI Gateway → GEMINI_API_KEY 순으로 fallback.
성분명은 한국어(식약처 공전 기준)로 반환 — 번역은 백엔드에서 처리.
"""
from __future__ import annotations

import os
import re

from openai import OpenAI

_GATEWAY_BASE_URL = "https://factchat-cloud.mindlogic.ai/v1/gateway"
_MODEL_PRIMARY = "gemini-3.5-flash"
_GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_MODEL_FALLBACK = "gemini-2.0-flash"

_PROMPT_TEMPLATE = """\
'{product_name}' 건강기능식품의 식약처(MFDS) 등록 기능성 원료 성분명을 알려줘.

규칙:
- 식약처 건강기능식품 공전에 등재된 기능성 원료명 기준으로 반환.
- 성분명만 콤마(,)로 구분해서 한 줄로 반환. 함량·단위·설명 없이.
- 모르거나 확인 불가하면 빈 문자열만 반환.

예시 출력:
비타민D, 칼슘, 마그네슘
"""


def _parse(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    ingredients = []
    for part in re.split(r"[,，\n]+", raw):
        name = re.sub(r"\(.*?\)", "", part).strip()
        name = re.sub(r"\s+", " ", name)
        if name and len(name) <= 60:
            ingredients.append(name)
    return ingredients


def lookup_ingredients_from_gemini(product_name: str) -> list[str]:
    """DB 미등재 제품명으로 Gemini에게 식약처 기준 성분 추출 요청.

    성분명은 한국어로 반환. 번역은 백엔드에서 처리.
    """
    prompt = _PROMPT_TEMPLATE.format(product_name=product_name)
    messages = [{"role": "user", "content": prompt}]

    primary_key = os.environ.get("CBNUAI_API_KEY", "").strip()
    if primary_key:
        try:
            client = OpenAI(api_key=primary_key, base_url=_GATEWAY_BASE_URL)
            raw = client.chat.completions.create(
                model=_MODEL_PRIMARY, messages=messages
            ).choices[0].message.content.strip()
            result = _parse(raw)
            if result:
                return result
        except Exception:
            pass

    fallback_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if fallback_key:
        try:
            client = OpenAI(api_key=fallback_key, base_url=_GOOGLE_BASE_URL)
            raw = client.chat.completions.create(
                model=_MODEL_FALLBACK, messages=messages
            ).choices[0].message.content.strip()
            result = _parse(raw)
            if result:
                return result
        except Exception:
            pass

    return []
