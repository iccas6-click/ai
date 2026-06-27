from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from .schemas import Candidate


@dataclass
class PredictionGroup:
    representative_bbox: np.ndarray
    representative_score: float
    class_scores: dict[int, float] = field(default_factory=dict)


def bbox_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection = intersection_width * intersection_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def group_predictions(
    bboxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    confidence_threshold: float,
    iou_threshold: float,
    top_k: int,
) -> list[tuple[tuple[int, int, int, int], list[Candidate]]]:
    keep = scores >= confidence_threshold
    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    if not len(bboxes):
        return []

    order = np.argsort(scores)[::-1]
    groups: list[PredictionGroup] = []

    for index in order:
        bbox = bboxes[index]
        score = float(scores[index])
        class_id = int(labels[index])

        target = next(
            (
                group
                for group in groups
                if bbox_iou(bbox, group.representative_bbox) > iou_threshold
            ),
            None,
        )
        if target is None:
            target = PredictionGroup(bbox.copy(), score)
            groups.append(target)

        target.class_scores[class_id] = max(
            score,
            target.class_scores.get(class_id, 0.0),
        )

    results = []
    for group in groups:
        ranked = sorted(
            group.class_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        candidates = [
            Candidate(
                rank=rank,
                class_id=class_id,
                class_name=class_names[class_id],
                confidence=round(confidence, 4),
            )
            for rank, (class_id, confidence) in enumerate(ranked, start=1)
        ]
        bbox = tuple(int(round(value)) for value in group.representative_bbox)
        results.append((bbox, candidates))

    return results
