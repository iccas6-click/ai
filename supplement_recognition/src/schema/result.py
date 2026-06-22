from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RecognitionStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    NEEDS_CONFIRMATION = "needs_confirmation"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class ErrorCode(str, Enum):
    INVALID_FILE = "INVALID_FILE"
    LOW_IMAGE_QUALITY = "LOW_IMAGE_QUALITY"
    OCR_TEXT_NOT_FOUND = "OCR_TEXT_NOT_FOUND"
    PRODUCT_NOT_MATCHED = "PRODUCT_NOT_MATCHED"
    MODEL_INFERENCE_FAILED = "MODEL_INFERENCE_FAILED"


class FunctionalIngredient(BaseModel):
    name: str
    amount: Optional[float] = None
    unit: Optional[str] = None
    evidence_text: str = Field(description="OCR 원문 중 이 성분 정보의 근거가 된 텍스트")


class SupplementProduct(BaseModel):
    product_code: Optional[str] = None
    product_name: str
    functional_ingredients: list[FunctionalIngredient] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class SupplementRecognitionResult(BaseModel):
    request_id: str
    status: RecognitionStatus
    product: Optional[SupplementProduct] = None
    needs_confirmation: bool = False
    warnings: list[str] = Field(default_factory=list)
    error_code: Optional[ErrorCode] = None
    error_detail: Optional[str] = None
