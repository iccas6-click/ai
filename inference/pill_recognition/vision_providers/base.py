from __future__ import annotations

from typing import Protocol

import numpy as np

from ..schemas import VisionObservation


class VisionProvider(Protocol):
    name: str

    def inspect_crop(self, crop_rgb: np.ndarray) -> VisionObservation:
        """Extract product and ingredient candidates from a single pill crop."""

    def inspect_crops(self, crops_rgb: list[np.ndarray]) -> list[VisionObservation]:
        """Extract product and ingredient candidates from multiple pill crops."""
