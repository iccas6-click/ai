from __future__ import annotations

import base64
import os
import re
import time
from pathlib import Path

from openai import OpenAI

# 1순위: 충북대 AI Gateway
_GATEWAY_BASE_URL = "https://factchat-cloud.mindlogic.ai/v1/gateway"
_MODEL_PRIMARY = "gemini-3.5-flash"

# 2순위 fallback: Google 공식 API (GEMINI_API_KEY 환경변수 설정 시 활성화)
_GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_MODEL_FALLBACK = "gemini-2.0-flash"

_MAX_RETRIES = 2
_RETRY_DELAY = 1.0  # 초, 재시도마다 2배씩 증가

_PROMPT = """\
이 건강기능식품 라벨 이미지에서 제품명 후보를 추출해줘.

규칙:
- 제품명을 가장 가능성이 높은 순서대로 3개 이하로 반환.
- 브랜드명(회사명)이 있으면 반드시 포함해서 반환.
  예) "고려은단 비타민C 1000", "락토핏 골드", "BB LAB 멀티비타민 슈퍼샷"
- 후보 1: 라벨에서 가장 크게 표시된 제품명 전체 (브랜드명 포함)
- 후보 2: 브랜드명 제외 제품 고유명만
- 후보 3: 다른 표기 방식이 있으면 추가 (없으면 생략)
- 용량, 정수, 광고 문구, 설명 문장은 제외.
- 한글로 반환할 것.
- 아래 형식으로만 반환, 다른 설명 없이:
1. 제품명 후보1
2. 제품명 후보2
3. 제품명 후보3
"""


def _build_messages(image_data: str, mime_type: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                },
            ],
        }
    ]


def _call_with_retry(client: OpenAI, model: str, messages: list[dict]) -> str:
    """지정 클라이언트로 최대 _MAX_RETRIES회 재시도."""
    last_exc: Exception = RuntimeError("재시도 횟수 초과")
    for attempt in range(max(_MAX_RETRIES + 1, 1)):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_exc = e
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY * (2 ** attempt))
    raise last_exc


def _parse_candidates(raw: str) -> list[str]:
    """'1. 제품명\n2. 제품명\n3. 제품명' 형식을 리스트로 파싱."""
    candidates = []
    for line in raw.strip().splitlines():
        line = line.strip()
        # "1. ", "2. " 등 번호 제거
        cleaned = re.sub(r"^\d+\.\s*", "", line).strip()
        if cleaned:
            candidates.append(cleaned)
    # 파싱 실패 시 원문 전체를 후보 1개로 반환
    return candidates if candidates else [raw.strip()]


def extract_product_name(image_path: Path | str) -> str:
    """하위 호환용 — 첫 번째 후보만 반환."""
    return extract_product_candidates(image_path)[0]


def extract_product_candidates(image_path: Path | str) -> list[str]:
    """Gemini로 제품명 후보 최대 3개를 반환."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    ext = Path(image_path).suffix.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    mime_type = f"image/{ext}"
    messages = _build_messages(image_data, mime_type)

    # 1순위: 충북대 AI Gateway
    primary_key = os.environ.get("CBNUAI_API_KEY", "")
    if primary_key:
        try:
            client = OpenAI(api_key=primary_key, base_url=_GATEWAY_BASE_URL)
            raw = _call_with_retry(client, _MODEL_PRIMARY, messages)
            return _parse_candidates(raw)
        except Exception:
            pass  # fallback으로 넘어감

    # 2순위 fallback: Google 공식 Gemini API
    fallback_key = os.environ.get("GEMINI_API_KEY", "")
    if fallback_key:
        client = OpenAI(api_key=fallback_key, base_url=_GOOGLE_BASE_URL)
        raw = _call_with_retry(client, _MODEL_FALLBACK, messages)
        return _parse_candidates(raw)

    raise RuntimeError(
        "Gemini API 키가 없습니다. CBNUAI_API_KEY 또는 GEMINI_API_KEY를 설정하세요."
    )
