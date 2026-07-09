from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from supplement_recognition.src.pipeline import recognize

router = APIRouter()

_UPLOAD_DIR = Path("/tmp/supplement_uploads")
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/supplement/recognize")
async def recognize_supplement(image: UploadFile = File(...)):
    """
    건강기능식품 라벨 이미지를 업로드하면 제품명, 성분 정보를 반환합니다.
    성분명은 한국어(식약처 기준)로 반환 — 번역은 백엔드에서 처리합니다.
    """
    ext = Path(image.filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=400, detail="JPG, PNG 파일만 지원합니다.")

    tmp_path = _UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    try:
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(image.file, f)
        result = recognize(tmp_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return result
