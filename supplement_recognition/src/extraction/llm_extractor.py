from __future__ import annotations

import os

from openai import OpenAI

_GATEWAY_BASE_URL = "https://factchat-cloud.mindlogic.ai/v1/gateway"
_MODEL = "gemini-3.5-flash"

_PROMPT = """\
아래는 건강기능식품 라벨에서 OCR로 추출한 텍스트입니다.
이 텍스트에서 제품명만 추출해줘.

규칙:
- 제품명만 반환할 것. 설명 없이.
- 브랜드명, 용량, 정수 등은 제외하고 핵심 제품명만.
- OCR 오류로 깨진 텍스트는 문맥상 가장 가까운 한글로 보정해서 반환.

OCR 텍스트:
{ocr_text}
"""


def extract_product_name(ocr_text: str) -> str:
    client = OpenAI(
        api_key=os.environ["CBNUAI_API_KEY"],
        base_url=_GATEWAY_BASE_URL,
    )
    response = client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": _PROMPT.format(ocr_text=ocr_text)}],
    )
    return response.choices[0].message.content.strip()
