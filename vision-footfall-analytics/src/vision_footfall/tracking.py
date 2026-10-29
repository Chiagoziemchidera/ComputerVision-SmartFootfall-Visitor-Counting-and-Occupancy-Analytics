from __future__ import annotations

import numpy as np
import supervision as sv

from .entities import PersonTrack


class PersonTracker:

    def __init__(self, frame_rate: int = 25):
        self.tracker = sv.ByteTrack(frame_rate=max(1, int(frame_rate)))

    def update(self, boxes: np.ndarray, confidences: np.ndarray) -> list[PersonTrack]:
        if len(boxes) == 0:
            detections = sv.Detections.empty()
        else:
            detections = sv.Detections(
                xyxy=boxes,
                confidence=confidences,
                class_id=np.zeros(len(boxes), dtype=int),
            )
        tracked = self.tracker.update_with_detections(detections)
        if tracked.tracker_id is None or len(tracked) == 0:
            return []
        results = []
        for index in range(len(tracked)):
            results.append(
                PersonTrack(
                    track_id=int(tracked.tracker_id[index]),
                    box=np.asarray(tracked.xyxy[index], dtype=np.float32),
                    confidence=float(tracked.confidence[index]),
                )
            )
        return results
