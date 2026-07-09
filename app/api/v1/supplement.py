from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from supplement_recognition.src.pipeline import recognize

router = APIRouter()

_UPLOAD_DIR = Path("/tmp/supplement_uploads")
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/supplement/recognize")
async def recognize_supplement(
    image: UploadFile = File(...),
    lang: str = Query(default="ko", description="성분명 번역 대상 언어 (ko/en/fr)"),
):
    """
    건강기능식품 라벨 이미지를 업로드하면 제품명, 성분 정보를 반환합니다.
    DB 미등재 제품의 경우 Gemini로 성분을 추출하며, lang 파라미터로 번역 언어를 지정할 수 있습니다.
    """
    ext = Path(image.filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=400, detail="JPG, PNG 파일만 지원합니다.")

    tmp_path = _UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    try:
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(image.file, f)
        result = recognize(tmp_path, lang=lang)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return result
