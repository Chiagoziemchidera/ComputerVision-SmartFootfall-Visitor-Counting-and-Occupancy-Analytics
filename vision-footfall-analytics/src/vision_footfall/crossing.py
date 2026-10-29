from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .entities import FaceObservation, PersonTrack


def line_side(
    point: tuple[float, float], line: tuple[float, float, float, float]
) -> float:
    x, y = point
    x1, y1, x2, y2 = line
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)


def crossing_is_allowed(
    previous_side: float, current_side: float, allowed_crossing: str
) -> bool:
    if previous_side == 0 or current_side == 0:
        return False
    if previous_side * current_side >= 0:
        return False
    direction = allowed_crossing.lower()
    if direction == "any":
        return True
    if direction == "negative_to_positive":
        return previous_side < 0 and current_side > 0
    if direction == "positive_to_negative":
        return previous_side > 0 and current_side < 0
    raise ValueError(f"Unknown crossing direction: {allowed_crossing}")


@dataclass
class TrackMemory:
    previous_side: float | None = None
    best_embedding: np.ndarray | None = None
    best_face_quality: float = 0.0
    last_seen_frame: int = 0
    crossing_recorded: bool = False


class CrossingMonitor:

    def __init__(self, normalized_line: list[float], allowed_crossing: str):
        self.normalized_line = normalized_line
        self.allowed_crossing = allowed_crossing
        self.track_memory: dict[int, TrackMemory] = {}

    def actual_line(self, frame: np.ndarray) -> tuple[float, float, float, float]:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self.normalized_line
        return (x1 * width, y1 * height, x2 * width, y2 * height)

    def update(
        self,
        frame: np.ndarray,
        frame_number: int,
        tracks: list[PersonTrack],
        face_assignments: dict[int, FaceObservation],
    ) -> list[tuple[PersonTrack, np.ndarray | None]]:
        line = self.actual_line(frame)
        crossings = []
        active_ids = set()
        for track in tracks:
            active_ids.add(track.track_id)
            memory = self.track_memory.setdefault(track.track_id, TrackMemory())
            memory.last_seen_frame = frame_number
            face = face_assignments.get(track.track_id)
            if face is not None and face.quality > memory.best_face_quality:
                memory.best_embedding = face.embedding.copy()
                memory.best_face_quality = face.quality
            current_side = line_side(track.foot_point, line)
            if (
                not memory.crossing_recorded
                and memory.previous_side is not None
                and crossing_is_allowed(
                    memory.previous_side, current_side, self.allowed_crossing
                )
            ):
                memory.crossing_recorded = True
                crossings.append((track, memory.best_embedding))
            memory.previous_side = current_side
        expired_ids = [
            track_id
            for track_id, memory in self.track_memory.items()
            if frame_number - memory.last_seen_frame > 300
        ]
        for track_id in expired_ids:
            del self.track_memory[track_id]
        return crossings
