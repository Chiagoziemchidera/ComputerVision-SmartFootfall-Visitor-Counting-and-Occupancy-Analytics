from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PersonTrack:
    track_id: int
    box: np.ndarray
    confidence: float

    @property
    def foot_point(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.box
        return float((x1 + x2) / 2), float(y2)


@dataclass
class FaceObservation:
    face_data: np.ndarray
    embedding: np.ndarray
    quality: float
    centre_x: float
    centre_y: float
