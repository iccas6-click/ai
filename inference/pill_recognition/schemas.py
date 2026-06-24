from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Candidate:
    rank: int
    class_id: int
    class_name: str
    confidence: float
    source: str = ""


@dataclass
class PillDetection:
    pill_id: int
    bbox: tuple[int, int, int, int]
    status: str
    rtmdet_candidates: list[Candidate] = field(default_factory=list)
    aihub_candidates: list[Candidate] = field(default_factory=list)
    cnn_candidates: list[Candidate] = field(default_factory=list)


@dataclass
class RecognitionResult:
    image_width: int
    image_height: int
    pill_count: int
    model_version: str
    detections: list[PillDetection]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
