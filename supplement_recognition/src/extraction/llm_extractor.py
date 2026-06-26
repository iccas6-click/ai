from __future__ import annotations

import base64
import os
from pathlib import Path

from openai import OpenAI

_GATEWAY_BASE_URL = "https://factchat-cloud.mindlogic.ai/v1/gateway"
_MODEL = "gemini-3.5-flash"

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


def extract_product_name(image_path: Path | str) -> str:
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    ext = Path(image_path).suffix.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    mime_type = f"image/{ext}"

    client = OpenAI(
        api_key=os.environ["CBNUAI_API_KEY"],
        base_url=_GATEWAY_BASE_URL,
    )
    response = client.chat.completions.create(
        model=_MODEL,
        messages=[
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
        ],
    )
    return response.choices[0].message.content.strip()
