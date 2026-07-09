from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.prescription_recognition import recognize_prescription_document

router = APIRouter()

_UPLOAD_DIR = Path("/tmp/pill_document_uploads")
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/pill/recognize")
async def recognize_pill(
    file: UploadFile = File(...),
    recognizer: str | None = Form(None),
    allowed_pill_ids: list[str] | None = Form(None),
    allowed_item_seqs: list[str] | None = Form(None),
    allowed_product_names: list[str] | None = Form(None),
):
    """처방전/약봉투 문서 이미지에서 복용 의약품과 성분 후보를 추출한다.

    기존 프론트가 보내던 form field는 호환성 때문에 유지하지만, 개별 알약
    detector/레거시 코드 선택값으로는 사용하지 않는다.
    """
    _ = (recognizer, allowed_pill_ids, allowed_item_seqs, allowed_product_names)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=400, detail="JPG, PNG, WEBP 파일만 지원합니다.")

    tmp_path = _UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    try:
        with tmp_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
        return recognize_prescription_document(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
