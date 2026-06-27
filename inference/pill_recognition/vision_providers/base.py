from __future__ import annotations

from typing import Protocol

import numpy as np

from ..schemas import VisionObservation


class VisionProvider(Protocol):
    name: str

    def inspect_crop(self, crop_rgb: np.ndarray) -> VisionObservation:
        """Extract visible pill clues such as imprint, color, shape, and text."""
