from __future__ import annotations

from pathlib import Path

import easyocr
import numpy as np
from PIL import Image

_reader: easyocr.Reader | None = None


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["ko", "en"], gpu=False)
    return _reader


def extract_text(image_path: Path | str) -> str:
    """이미지에서 텍스트 추출 후 하나의 문자열로 반환."""
    reader = _get_reader()
    img = np.array(Image.open(image_path).convert("RGB"))
    results = reader.readtext(img, detail=0)
    return " ".join(results)
