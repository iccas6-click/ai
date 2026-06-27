from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class VisionProductCandidate:
    product_name: str
    ingredient: str | None = None
    caution_points: list[str] = field(default_factory=list)
    confidence: float | None = None


@dataclass(frozen=True)
class VisionObservation:
    imprint_front: str | None = None
    imprint_back: str | None = None
    shape: str | None = None
    color: str | None = None
    text: str | None = None
    product_candidates: list[VisionProductCandidate] = field(default_factory=list)
    possible_product_names: list[str] = field(default_factory=list)
    confidence: float | None = None
    notes: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProductCandidate:
    rank: int
    pill_id: str
    score: float
    source: str = "aihub_db"
    product_name: str | None = None
    ingredient: str | None = None
    caution_points: list[str] = field(default_factory=list)
    company: str | None = None
    item_seq: str | None = None
    etc_otc_code: str | None = None
    print_front: str | None = None
    print_back: str | None = None
    drug_shape: str | None = None
    color_class1: str | None = None
    color_class2: str | None = None
    matched: str | None = None
    reference_image_url: str | None = None


@dataclass
class PillDetection:
    pill_id: int
    bbox: tuple[int, int, int, int]
    crop_bbox: tuple[int, int, int, int]
    detector_confidence: float
    vision: VisionObservation
    candidates: list[ProductCandidate]
    status: str
    status_reason: str | None = None


@dataclass
class RecognitionResult:
    image_width: int
    image_height: int
    pill_count: int
    model_version: str
    detections: list[PillDetection]
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
