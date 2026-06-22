import numpy as np
import pytest

from src.ocr.reader import OcrResult, run_ocr


def test_ocr_result_from_easyocr_empty():
    result = OcrResult.from_easyocr([])
    assert result.full_text == ""
    assert result.blocks == []


def test_ocr_result_from_easyocr_single_block():
    raw = [([[0, 0], [10, 0], [10, 10], [0, 10]], "오메가3", 0.95)]
    result = OcrResult.from_easyocr(raw)
    assert result.full_text == "오메가3"
    assert len(result.blocks) == 1
    assert result.blocks[0].confidence == 0.95


def test_run_ocr_blank_image():
    blank = np.ones((100, 100, 3), dtype=np.uint8) * 255
    result = run_ocr(blank)
    assert isinstance(result, OcrResult)
