from __future__ import annotations

import os
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

_PROMPT = """\
이 건강기능식품 이미지에서 제품명만 추출해줘.

규칙:
- 이미지에 명시된 제품명만 반환할 것.
- 브랜드명, 부제목, 용량 등은 제외하고 핵심 제품명만.
- 제품명 텍스트만 응답할 것. 설명 없이.

예시:
- "종근당 칼슘앤마그네슘 비타민D 아연 1000mg 180정" → 칼슘앤마그네슘 비타민D 아연
- "CJ 아이시안 루테인지아잔틴 플러스 케어+" → 아이시안 루테인지아잔틴
"""


def extract_product_name(image_path: Path | str) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    image = Image.open(image_path)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[image, _PROMPT],
    )
    return response.text.strip()
