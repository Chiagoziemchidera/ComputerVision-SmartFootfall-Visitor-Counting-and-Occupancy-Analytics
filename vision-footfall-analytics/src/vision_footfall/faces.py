from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

from .embeddings import normalize_embedding
from .entities import FaceObservation, PersonTrack


class FaceEngine:

    def __init__(
        self,
        detector_path: str,
        recognizer_path: str,
        detection_threshold: float,
        minimum_face_width: int,
    ):
        if not Path(detector_path).exists():
            raise FileNotFoundError(f"YuNet model was not found: {detector_path}")
        if not Path(recognizer_path).exists():
            raise FileNotFoundError(f"SFace model was not found: {recognizer_path}")
        self.minimum_face_width = int(minimum_face_width)
        self.lock = threading.RLock()
        self.detector = cv2.FaceDetectorYN.create(
            detector_path, "", (320, 320), float(detection_threshold), 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(recognizer_path, "")

    def detect_and_embed(self, frame: np.ndarray) -> list[FaceObservation]:
        with self.lock:
            return self._detect_and_embed(frame)

    def _detect_and_embed(self, frame: np.ndarray) -> list[FaceObservation]:
        frame_height, frame_width = frame.shape[:2]
        self.detector.setInputSize((frame_width, frame_height))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return []
        observations = []
        for face_data in faces:
            face_width = float(face_data[2])
            face_height = float(face_data[3])
            detection_score = float(face_data[-1])
            if face_width < self.minimum_face_width:
                continue
            try:
                aligned_face = self.recognizer.alignCrop(frame, face_data)
                embedding = self.recognizer.feature(aligned_face)
            except cv2.error:
                continue
            embedding = normalize_embedding(embedding)
            centre_x = float(face_data[0]) + face_width / 2
            centre_y = float(face_data[1]) + face_height / 2
            quality = face_width * face_height * detection_score
            observations.append(
                FaceObservation(
                    face_data=face_data,
                    embedding=embedding,
                    quality=quality,
                    centre_x=centre_x,
                    centre_y=centre_y,
                )
            )
        return observations


def point_inside_box(x: float, y: float, box: np.ndarray) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def assign_faces_to_tracks(
    tracks: list[PersonTrack], faces: list[FaceObservation]
) -> dict[int, FaceObservation]:
    assignments = {}
    for track in tracks:
        possible_faces = [
            face
            for face in faces
            if point_inside_box(face.centre_x, face.centre_y, track.box)
        ]
        if not possible_faces:
            continue
        best_face = max(possible_faces, key=lambda item: item.quality)
        assignments[track.track_id] = best_face
    return assignments
