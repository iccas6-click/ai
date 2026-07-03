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


class SupplementProduct(BaseModel):
    product_code: Optional[str] = None
    product_name: str
    manufacturer: Optional[str] = None
    product_image_url: Optional[str] = None
    main_function: Optional[str] = None
    base_standard: Optional[str] = None
    ingredients: list[str] = Field(default_factory=list)  # 파싱된 성분명 리스트
    confidence: float = Field(ge=0.0, le=1.0)


class SupplementRecognitionResult(BaseModel):
    request_id: str
    status: RecognitionStatus
    product: Optional[SupplementProduct] = None
    needs_confirmation: bool = False
    warnings: list[str] = Field(default_factory=list)
    error_code: Optional[ErrorCode] = None
    error_detail: Optional[str] = None
