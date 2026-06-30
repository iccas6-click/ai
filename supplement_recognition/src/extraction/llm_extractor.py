from __future__ import annotations

import base64
import os
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
이 건강기능식품 라벨 이미지에서 제품 고유명을 추출해줘.

규칙:
- 제품 고유명만 반환. 브랜드명(회사명) 제외.
- 예) "고려은단 메가도스D 비타민D3" → "메가도스D 비타민D3"
- 예) "종근당 칼슘앤마그네슘 비타민D 아연" → "칼슘앤마그네슘 비타민D 아연"
- 예) "센트룸 멀티 구미" → "센트룸 멀티 구미" (센트룸은 브랜드가 아닌 제품명)
- 용량, 정수, 광고 문구 등은 제외.
- 한글로 반환할 것.
- 설명 없이 제품명만 반환.
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
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
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
    raise last_exc  # type: ignore[misc]


def extract_product_name(image_path: Path | str) -> str:
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
            return _call_with_retry(client, _MODEL_PRIMARY, messages)
        except Exception:
            pass  # fallback으로 넘어감

    # 2순위 fallback: Google 공식 Gemini API
    fallback_key = os.environ.get("GEMINI_API_KEY", "")
    if fallback_key:
        client = OpenAI(api_key=fallback_key, base_url=_GOOGLE_BASE_URL)
        return _call_with_retry(client, _MODEL_FALLBACK, messages)

    raise RuntimeError(
        "Gemini API 키가 없습니다. CBNUAI_API_KEY 또는 GEMINI_API_KEY를 설정하세요."
    )
