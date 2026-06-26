from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.post("/pill/recognize")
async def recognize_pill():
    """알약 인식 엔드포인트 (미구현)"""
    return {"status": "not_implemented"}
