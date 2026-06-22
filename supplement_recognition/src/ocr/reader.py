from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import easyocr
import numpy as np
from PIL import Image


@dataclass
class OcrBlock:
    text: str
    bbox: list[list[int]]  # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    confidence: float


@dataclass
class OcrResult:
    blocks: list[OcrBlock] = field(default_factory=list)
    full_text: str = ""

    @classmethod
    def from_easyocr(cls, raw: list) -> "OcrResult":
        blocks = [
            OcrBlock(text=text, bbox=bbox, confidence=conf)
            for bbox, text, conf in raw
        ]
        full_text = "\n".join(b.text for b in blocks)
        return cls(blocks=blocks, full_text=full_text)


_reader: easyocr.Reader | None = None


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["ko", "en"], gpu=False)
    return _reader


def run_ocr(image: Path | str | np.ndarray) -> OcrResult:
    reader = _get_reader()
    if isinstance(image, (Path, str)):
        img = np.array(Image.open(image).convert("RGB"))
    else:
        img = image
    raw = reader.readtext(img, detail=1)
    return OcrResult.from_easyocr(raw)
