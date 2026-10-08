from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sqlite3
import threading
import time
import uuid

from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv
import yaml


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
)

logger = logging.getLogger("footfall-pilot")


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return embedding

    return embedding / norm


def cosine_similarity(
    first_embedding: np.ndarray,
    second_embedding: np.ndarray,
) -> float:
    first = normalize_embedding(first_embedding)
    second = normalize_embedding(second_embedding)

    return float(np.dot(first, second))


def expand_environment_variables(value):
    if isinstance(value, dict):
        return {
            key: expand_environment_variables(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            expand_environment_variables(item)
            for item in value
        ]

    if isinstance(value, str):
        return os.path.expandvars(value)

    return value


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    return expand_environment_variables(config)


def parse_clock(value: str) -> dt_time:
    return datetime.strptime(value, "%H:%M").time()


def open_video_source(source):
    if isinstance(source, int):
        return source

    source = str(source)

    if source.isdigit():
        return int(source)

    return source


def source_is_live(source) -> bool:
    if isinstance(source, int):
        return True

    source = str(source).lower()

    return source.startswith(
        (
            "rtsp://",
            "rtsps://",
            "http://",
            "https://",
        )
    )


# ============================================================
# YOLOX PERSON DETECTOR
# ============================================================

def yolox_postprocess(
    outputs: np.ndarray,
    input_size: tuple[int, int],
    p6: bool = False,
) -> np.ndarray:
    grids = []
    expanded_strides = []

    strides = [8, 16, 32]

    if p6:
        strides.append(64)

    input_height, input_width = input_size

    for stride in strides:
        height_size = input_height // stride
        width_size = input_width // stride

        grid_y, grid_x = np.meshgrid(
            np.arange(height_size),
            np.arange(width_size),
            indexing="ij",
        )

        grid = np.stack(
            (grid_x, grid_y),
            axis=2,
        ).reshape(1, -1, 2)

        grids.append(grid)

        shape = grid.shape[:2]

        expanded_strides.append(
            np.full(
                (*shape, 1),
                stride,
                dtype=np.float32,
            )
        )

    grids = np.concatenate(grids, axis=1).astype(np.float32)

    expanded_strides = np.concatenate(
        expanded_strides,
        axis=1,
    ).astype(np.float32)

    outputs[..., :2] = (
        outputs[..., :2] + grids
    ) * expanded_strides

    outputs[..., 2:4] = (
        np.exp(outputs[..., 2:4])
        * expanded_strides
    )

    return outputs


def preprocess_yolox(
    frame: np.ndarray,
    input_size: tuple[int, int],
) -> tuple[np.ndarray, float]:
    input_height, input_width = input_size

    frame_height, frame_width = frame.shape[:2]

    scale = min(
        input_height / frame_height,
        input_width / frame_width,
    )

    resized_width = int(frame_width * scale)
    resized_height = int(frame_height * scale)

    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    padded = np.full(
        (input_height, input_width, 3),
        114,
        dtype=np.uint8,
    )

    padded[:resized_height, :resized_width] = resized

    tensor = padded.transpose(2, 0, 1)
    tensor = np.ascontiguousarray(tensor, dtype=np.float32)
    tensor = np.expand_dims(tensor, axis=0)

    return tensor, scale


class YOLOXPersonDetector:
    def __init__(
        self,
        model_path: str,
        input_width: int,
        input_height: int,
        confidence_threshold: float,
        nms_threshold: float,
    ):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"YOLOX model was not found: {model_path}"
            )

        self.input_size = (
            int(input_height),
            int(input_width),
        )

        self.confidence_threshold = float(
            confidence_threshold
        )

        self.nms_threshold = float(nms_threshold)

        available_providers = ort.get_available_providers()

        preferred_providers = [
            provider
            for provider in [
                "CUDAExecutionProvider",
                "OpenVINOExecutionProvider",
                "CPUExecutionProvider",
            ]
            if provider in available_providers
        ]

        self.session = ort.InferenceSession(
            model_path,
            providers=preferred_providers,
        )

        self.input_name = self.session.get_inputs()[0].name

        logger.info(
            "YOLOX loaded with providers: %s",
            self.session.get_providers(),
        )

    def detect(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        tensor, scale = preprocess_yolox(
            frame,
            self.input_size,
        )

        raw_output = self.session.run(
            None,
            {self.input_name: tensor},
        )[0]

        predictions = yolox_postprocess(
            raw_output,
            self.input_size,
        )[0]

        if predictions.size == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        centre_x = predictions[:, 0]
        centre_y = predictions[:, 1]
        width = predictions[:, 2]
        height = predictions[:, 3]

        boxes = np.column_stack(
            (
                centre_x - width / 2,
                centre_y - height / 2,
                centre_x + width / 2,
                centre_y + height / 2,
            )
        )

        object_confidence = predictions[:, 4]

        class_scores = predictions[:, 5:]

        if class_scores.shape[1] == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        # COCO class 0 represents a person.
        person_confidence = (
            object_confidence * class_scores[:, 0]
        )

        keep = (
            person_confidence
            >= self.confidence_threshold
        )

        boxes = boxes[keep]
        person_confidence = person_confidence[keep]

        if len(boxes) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        boxes = boxes / scale

        frame_height, frame_width = frame.shape[:2]

        boxes[:, 0] = np.clip(
            boxes[:, 0],
            0,
            frame_width - 1,
        )

        boxes[:, 1] = np.clip(
            boxes[:, 1],
            0,
            frame_height - 1,
        )

        boxes[:, 2] = np.clip(
            boxes[:, 2],
            0,
            frame_width - 1,
        )

        boxes[:, 3] = np.clip(
            boxes[:, 3],
            0,
            frame_height - 1,
        )

        nms_boxes = []

        for x1, y1, x2, y2 in boxes:
            nms_boxes.append(
                [
                    int(x1),
                    int(y1),
                    int(x2 - x1),
                    int(y2 - y1),
                ]
            )

        indexes = cv2.dnn.NMSBoxes(
            nms_boxes,
            person_confidence.tolist(),
            self.confidence_threshold,
            self.nms_threshold,
        )

        if len(indexes) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        indexes = np.asarray(indexes).reshape(-1)

        return (
            boxes[indexes].astype(np.float32),
            person_confidence[indexes].astype(np.float32),
        )


# ============================================================
# BYTETRACK
# ============================================================

@dataclass
class PersonTrack:
    track_id: int
    box: np.ndarray
    confidence: float

    @property
    def foot_point(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.box

        return (
            float((x1 + x2) / 2),
            float(y2),
        )


class PersonTracker:
    def __init__(self, frame_rate: int = 25):
        self.tracker = sv.ByteTrack(
            frame_rate=max(1, int(frame_rate))
        )

    def update(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
    ) -> list[PersonTrack]:
        if len(boxes) == 0:
            detections = sv.Detections.empty()
        else:
            detections = sv.Detections(
                xyxy=boxes,
                confidence=confidences,
                class_id=np.zeros(
                    len(boxes),
                    dtype=int,
                ),
            )

        tracked = self.tracker.update_with_detections(
            detections
        )

        if (
            tracked.tracker_id is None
            or len(tracked) == 0
        ):
            return []

        results = []

        for index in range(len(tracked)):
            results.append(
                PersonTrack(
                    track_id=int(
                        tracked.tracker_id[index]
                    ),
                    box=np.asarray(
                        tracked.xyxy[index],
                        dtype=np.float32,
                    ),
                    confidence=float(
                        tracked.confidence[index]
                    ),
                )
            )

        return results


# ============================================================
# FACE DETECTION AND EMBEDDINGS
# ============================================================

@dataclass
class FaceObservation:
    face_data: np.ndarray
    embedding: np.ndarray
    quality: float
    centre_x: float
    centre_y: float


class FaceEngine:
    def __init__(
        self,
        detector_path: str,
        recognizer_path: str,
        detection_threshold: float,
        minimum_face_width: int,
    ):
        if not Path(detector_path).exists():
            raise FileNotFoundError(
                f"YuNet model was not found: {detector_path}"
            )

        if not Path(recognizer_path).exists():
            raise FileNotFoundError(
                f"SFace model was not found: {recognizer_path}"
            )

        self.minimum_face_width = int(
            minimum_face_width
        )

        self.detector = cv2.FaceDetectorYN.create(
            detector_path,
            "",
            (320, 320),
            float(detection_threshold),
            0.3,
            5000,
        )

        self.recognizer = (
            cv2.FaceRecognizerSF.create(
                recognizer_path,
                "",
            )
        )

    def detect_and_embed(
        self,
        frame: np.ndarray,
    ) -> list[FaceObservation]:
        frame_height, frame_width = frame.shape[:2]

        self.detector.setInputSize(
            (frame_width, frame_height)
        )

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
                aligned_face = (
                    self.recognizer.alignCrop(
                        frame,
                        face_data,
                    )
                )

                embedding = self.recognizer.feature(
                    aligned_face
                )

            except cv2.error:
                continue

            embedding = normalize_embedding(embedding)

            centre_x = (
                float(face_data[0])
                + face_width / 2
            )

            centre_y = (
                float(face_data[1])
                + face_height / 2
            )

            quality = (
                face_width
                * face_height
                * detection_score
            )

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


def point_inside_box(
    x: float,
    y: float,
    box: np.ndarray,
) -> bool:
    x1, y1, x2, y2 = box

    return (
        x1 <= x <= x2
        and y1 <= y <= y2
    )


def assign_faces_to_tracks(
    tracks: list[PersonTrack],
    faces: list[FaceObservation],
) -> dict[int, FaceObservation]:
    assignments = {}

    for track in tracks:
        possible_faces = [
            face
            for face in faces
            if point_inside_box(
                face.centre_x,
                face.centre_y,
                track.box,
            )
        ]

        if not possible_faces:
            continue

        best_face = max(
            possible_faces,
            key=lambda item: item.quality,
        )

        assignments[track.track_id] = best_face

    return assignments


# ============================================================
# VIRTUAL LINE CROSSING
# ============================================================

def line_side(
    point: tuple[float, float],
    line: tuple[float, float, float, float],
) -> float:
    x, y = point
    x1, y1, x2, y2 = line

    return (
        (x - x1) * (y2 - y1)
        - (y - y1) * (x2 - x1)
    )


def crossing_is_allowed(
    previous_side: float,
    current_side: float,
    allowed_crossing: str,
) -> bool:
    if previous_side == 0 or current_side == 0:
        return False

    if previous_side * current_side >= 0:
        return False

    direction = allowed_crossing.lower()

    if direction == "any":
        return True

    if direction == "negative_to_positive":
        return (
            previous_side < 0
            and current_side > 0
        )

    if direction == "positive_to_negative":
        return (
            previous_side > 0
            and current_side < 0
        )

    raise ValueError(
        f"Unknown crossing direction: "
        f"{allowed_crossing}"
    )


@dataclass
class TrackMemory:
    previous_side: Optional[float] = None
    best_embedding: Optional[np.ndarray] = None
    best_face_quality: float = 0.0
    last_seen_frame: int = 0
    crossing_recorded: bool = False


class CrossingMonitor:
    def __init__(
        self,
        normalized_line: list[float],
        allowed_crossing: str,
    ):
        self.normalized_line = normalized_line
        self.allowed_crossing = allowed_crossing
        self.track_memory: dict[int, TrackMemory] = {}

    def actual_line(
        self,
        frame: np.ndarray,
    ) -> tuple[float, float, float, float]:
        height, width = frame.shape[:2]

        x1, y1, x2, y2 = self.normalized_line

        return (
            x1 * width,
            y1 * height,
            x2 * width,
            y2 * height,
        )

    def update(
        self,
        frame: np.ndarray,
        frame_number: int,
        tracks: list[PersonTrack],
        face_assignments: dict[int, FaceObservation],
    ) -> list[tuple[PersonTrack, Optional[np.ndarray]]]:
        line = self.actual_line(frame)
        crossings = []

        active_ids = set()

        for track in tracks:
            active_ids.add(track.track_id)

            memory = self.track_memory.setdefault(
                track.track_id,
                TrackMemory(),
            )

            memory.last_seen_frame = frame_number

            face = face_assignments.get(
                track.track_id
            )

            if (
                face is not None
                and face.quality
                > memory.best_face_quality
            ):
                memory.best_embedding = (
                    face.embedding.copy()
                )

                memory.best_face_quality = (
                    face.quality
                )

            current_side = line_side(
                track.foot_point,
                line,
            )

            if (
                not memory.crossing_recorded
                and memory.previous_side is not None
                and crossing_is_allowed(
                    memory.previous_side,
                    current_side,
                    self.allowed_crossing,
                )
            ):
                memory.crossing_recorded = True

                crossings.append(
                    (
                        track,
                        memory.best_embedding,
                    )
                )

            memory.previous_side = current_side

        expired_ids = [
            track_id
            for track_id, memory
            in self.track_memory.items()
            if frame_number
            - memory.last_seen_frame
            > 300
        ]

        for track_id in expired_ids:
            del self.track_memory[track_id]

        return crossings


# ============================================================
# DATABASE AND COUNTING RULES
# ============================================================

class FootfallDatabase:
    def __init__(
        self,
        events_path: str,
        identities_path: str,
        timezone_name: str,
        match_threshold: float,
        repeat_visit_hours: float,
        location_id: str,
    ):
        Path(events_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        Path(identities_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.timezone = ZoneInfo(timezone_name)
        self.match_threshold = float(match_threshold)

        self.repeat_interval = timedelta(
            hours=float(repeat_visit_hours)
        )

        self.location_id = location_id
        self.lock = threading.RLock()

        self.events = sqlite3.connect(
            events_path,
            check_same_thread=False,
        )

        self.identities = sqlite3.connect(
            identities_path,
            check_same_thread=False,
        )

        self.events.row_factory = sqlite3.Row
        self.identities.row_factory = sqlite3.Row

        self.events.execute(
            "PRAGMA journal_mode=WAL"
        )

        self.identities.execute(
            "PRAGMA journal_mode=WAL"
        )

        self.identities.execute(
            "PRAGMA secure_delete=ON"
        )

        self.create_tables()
        self.purge_old_identities()

    def create_tables(self):
        with self.events:
            self.events.executescript(
                """
                CREATE TABLE IF NOT EXISTS counters (
                    day TEXT PRIMARY KEY,
                    location_id TEXT NOT NULL,
                    physical_entries INTEGER NOT NULL DEFAULT 0,
                    physical_exits INTEGER NOT NULL DEFAULT 0,
                    verified_daily_unique INTEGER NOT NULL DEFAULT 0,
                    unresolved_identity_entries INTEGER NOT NULL DEFAULT 0,
                    repeat_entries INTEGER NOT NULL DEFAULT 0,
                    qualified_repeat_visits INTEGER NOT NULL DEFAULT 0,
                    occupancy INTEGER NOT NULL DEFAULT 0,
                    peak_occupancy INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS crossing_events (
                    event_id TEXT PRIMARY KEY,
                    location_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    event_time TEXT NOT NULL,
                    day TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    identity_status TEXT NOT NULL,
                    match_score REAL,
                    unique_increment INTEGER NOT NULL DEFAULT 0,
                    repeat_increment INTEGER NOT NULL DEFAULT 0,
                    qualified_repeat_increment INTEGER NOT NULL DEFAULT 0,
                    occupancy_after INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS occupancy_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    location_id TEXT NOT NULL,
                    event_time TEXT NOT NULL,
                    occupancy INTEGER NOT NULL
                );
                """
            )

        with self.identities:
            self.identities.executescript(
                """
                CREATE TABLE IF NOT EXISTS daily_identities (
                    day TEXT NOT NULL,
                    visitor_id TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    sample_count INTEGER NOT NULL DEFAULT 1,
                    has_entered INTEGER NOT NULL DEFAULT 0,
                    first_seen TEXT NOT NULL,
                    last_entry TEXT,
                    last_exit TEXT,
                    is_inside INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (day, visitor_id)
                );
                """
            )

    def ensure_counter(
        self,
        day: str,
        timestamp: datetime,
    ):
        self.events.execute(
            """
            INSERT OR IGNORE INTO counters (
                day,
                location_id,
                updated_at
            )
            VALUES (?, ?, ?)
            """,
            (
                day,
                self.location_id,
                timestamp.isoformat(),
            ),
        )

    def find_identity(
        self,
        day: str,
        embedding: np.ndarray,
    ) -> tuple[Optional[sqlite3.Row], Optional[float]]:
        rows = self.identities.execute(
            """
            SELECT *
            FROM daily_identities
            WHERE day = ?
            """,
            (day,),
        ).fetchall()

        if not rows:
            return None, None

        best_row = None
        best_score = -1.0

        for row in rows:
            saved_embedding = np.frombuffer(
                row["embedding"],
                dtype=np.float32,
            )

            score = cosine_similarity(
                embedding,
                saved_embedding,
            )

            if score > best_score:
                best_score = score
                best_row = row

        if best_score >= self.match_threshold:
            return best_row, best_score

        return None, best_score

    def create_identity(
        self,
        day: str,
        timestamp: datetime,
        embedding: np.ndarray,
        has_entered: bool,
    ) -> str:
        visitor_id = (
            f"{day}-"
            f"{uuid.uuid4().hex[:12].upper()}"
        )

        embedding = normalize_embedding(embedding)

        self.identities.execute(
            """
            INSERT INTO daily_identities (
                day,
                visitor_id,
                embedding,
                sample_count,
                has_entered,
                first_seen,
                last_entry,
                last_exit,
                is_inside
            )
            VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)
            """,
            (
                day,
                visitor_id,
                embedding.astype(
                    np.float32
                ).tobytes(),
                1 if has_entered else 0,
                timestamp.isoformat(),
                timestamp.isoformat()
                if has_entered
                else None,
                None
                if has_entered
                else timestamp.isoformat(),
                1 if has_entered else 0,
            ),
        )

        return visitor_id

    def update_identity_embedding(
        self,
        row: sqlite3.Row,
        new_embedding: np.ndarray,
    ):
        old_embedding = np.frombuffer(
            row["embedding"],
            dtype=np.float32,
        )

        sample_count = int(row["sample_count"])

        updated_embedding = normalize_embedding(
            (
                old_embedding * sample_count
                + normalize_embedding(new_embedding)
            )
            / (sample_count + 1)
        )

        self.identities.execute(
            """
            UPDATE daily_identities
            SET
                embedding = ?,
                sample_count = sample_count + 1
            WHERE
                day = ?
                AND visitor_id = ?
            """,
            (
                updated_embedding.astype(
                    np.float32
                ).tobytes(),
                row["day"],
                row["visitor_id"],
            ),
        )

    def process_crossing(
        self,
        camera_id: str,
        direction: str,
        timestamp: datetime,
        embedding: Optional[np.ndarray],
    ) -> dict:
        with self.lock:
            day = timestamp.astimezone(
                self.timezone
            ).date().isoformat()

            self.ensure_counter(day, timestamp)

            counter = self.events.execute(
                """
                SELECT *
                FROM counters
                WHERE day = ?
                """,
                (day,),
            ).fetchone()

            occupancy = int(counter["occupancy"])
            peak_occupancy = int(
                counter["peak_occupancy"]
            )

            identity_status = "unresolved"
            match_score = None

            unique_increment = 0
            repeat_increment = 0
            qualified_repeat_increment = 0

            identity_row = None

            if embedding is not None:
                embedding = normalize_embedding(
                    embedding
                )

                identity_row, match_score = (
                    self.find_identity(
                        day,
                        embedding,
                    )
                )

            if direction == "entry":
                occupancy += 1
                peak_occupancy = max(
                    peak_occupancy,
                    occupancy,
                )

                if embedding is None:
                    identity_status = "unresolved"

                elif identity_row is None:
                    self.create_identity(
                        day=day,
                        timestamp=timestamp,
                        embedding=embedding,
                        has_entered=True,
                    )

                    identity_status = "new"
                    unique_increment = 1

                else:
                    self.update_identity_embedding(
                        identity_row,
                        embedding,
                    )

                    has_entered = bool(
                        identity_row["has_entered"]
                    )

                    if not has_entered:
                        identity_status = (
                            "first_entry_after_exit"
                        )

                        unique_increment = 1

                    else:
                        identity_status = "repeat"
                        repeat_increment = 1

                        if identity_row["last_exit"]:
                            last_exit = (
                                datetime.fromisoformat(
                                    identity_row[
                                        "last_exit"
                                    ]
                                )
                            )

                            time_away = (
                                timestamp - last_exit
                            )

                            if (
                                time_away
                                >= self.repeat_interval
                            ):
                                qualified_repeat_increment = 1

                    self.identities.execute(
                        """
                        UPDATE daily_identities
                        SET
                            has_entered = 1,
                            last_entry = ?,
                            is_inside = 1
                        WHERE
                            day = ?
                            AND visitor_id = ?
                        """,
                        (
                            timestamp.isoformat(),
                            day,
                            identity_row[
                                "visitor_id"
                            ],
                        ),
                    )

                self.events.execute(
                    """
                    UPDATE counters
                    SET
                        physical_entries =
                            physical_entries + 1,
                        verified_daily_unique =
                            verified_daily_unique + ?,
                        unresolved_identity_entries =
                            unresolved_identity_entries + ?,
                        repeat_entries =
                            repeat_entries + ?,
                        qualified_repeat_visits =
                            qualified_repeat_visits + ?,
                        occupancy = ?,
                        peak_occupancy = ?,
                        updated_at = ?
                    WHERE day = ?
                    """,
                    (
                        unique_increment,
                        1
                        if identity_status
                        == "unresolved"
                        else 0,
                        repeat_increment,
                        qualified_repeat_increment,
                        occupancy,
                        peak_occupancy,
                        timestamp.isoformat(),
                        day,
                    ),
                )

            elif direction == "exit":
                occupancy = max(0, occupancy - 1)

                if embedding is None:
                    identity_status = "unresolved"

                elif identity_row is None:
                    self.create_identity(
                        day=day,
                        timestamp=timestamp,
                        embedding=embedding,
                        has_entered=False,
                    )

                    identity_status = (
                        "exit_identity_created"
                    )

                else:
                    identity_status = "matched_exit"

                    self.update_identity_embedding(
                        identity_row,
                        embedding,
                    )

                    self.identities.execute(
                        """
                        UPDATE daily_identities
                        SET
                            last_exit = ?,
                            is_inside = 0
                        WHERE
                            day = ?
                            AND visitor_id = ?
                        """,
                        (
                            timestamp.isoformat(),
                            day,
                            identity_row[
                                "visitor_id"
                            ],
                        ),
                    )

                self.events.execute(
                    """
                    UPDATE counters
                    SET
                        physical_exits =
                            physical_exits + 1,
                        occupancy = ?,
                        updated_at = ?
                    WHERE day = ?
                    """,
                    (
                        occupancy,
                        timestamp.isoformat(),
                        day,
                    ),
                )

            else:
                raise ValueError(
                    f"Invalid direction: {direction}"
                )

            event_id = str(uuid.uuid4())

            self.events.execute(
                """
                INSERT INTO crossing_events (
                    event_id,
                    location_id,
                    camera_id,
                    event_time,
                    day,
                    direction,
                    identity_status,
                    match_score,
                    unique_increment,
                    repeat_increment,
                    qualified_repeat_increment,
                    occupancy_after,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    self.location_id,
                    camera_id,
                    timestamp.isoformat(),
                    day,
                    direction,
                    identity_status,
                    match_score,
                    unique_increment,
                    repeat_increment,
                    qualified_repeat_increment,
                    occupancy,
                    datetime.now(
                        self.timezone
                    ).isoformat(),
                ),
            )

            self.events.execute(
                """
                INSERT INTO occupancy_snapshots (
                    snapshot_id,
                    location_id,
                    event_time,
                    occupancy
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    self.location_id,
                    timestamp.isoformat(),
                    occupancy,
                ),
            )

            self.events.commit()
            self.identities.commit()

            result = {
                "event_id": event_id,
                "camera_id": camera_id,
                "event_time": timestamp.isoformat(),
                "direction": direction,
                "identity_status": identity_status,
                "match_score": match_score,
                "unique_increment": unique_increment,
                "repeat_increment": repeat_increment,
                "qualified_repeat_increment":
                    qualified_repeat_increment,
                "occupancy_after": occupancy,
            }

            logger.info(
                "Crossing event: %s",
                json.dumps(result),
            )

            return result

    def get_counters(
        self,
        day: Optional[str] = None,
    ) -> dict:
        with self.lock:
            if day is None:
                day = datetime.now(
                    self.timezone
                ).date().isoformat()

            row = self.events.execute(
                """
                SELECT *
                FROM counters
                WHERE day = ?
                """,
                (day,),
            ).fetchone()

            return dict(row) if row else {}

    def purge_old_identities(self):
        with self.lock:
            current_day = datetime.now(
                self.timezone
            ).date().isoformat()

            deleted = self.identities.execute(
                """
                DELETE FROM daily_identities
                WHERE day < ?
                """,
                (current_day,),
            ).rowcount

            self.identities.commit()

            if deleted:
                self.identities.execute("VACUUM")
                self.identities.commit()

                logger.info(
                    "Deleted %s previous-day "
                    "biometric identities.",
                    deleted,
                )

    def close(self):
        with self.lock:
            self.events.close()
            self.identities.close()


# ============================================================
# CAMERA WORKER
# ============================================================

class CameraWorker(threading.Thread):
    def __init__(
        self,
        camera_config: dict,
        general_config: dict,
        detector: YOLOXPersonDetector,
        face_engine: FaceEngine,
        database: FootfallDatabase,
        stop_event: threading.Event,
        timezone_name: str,
    ):
        super().__init__(
            name=camera_config["id"],
            daemon=True,
        )

        self.camera_config = camera_config
        self.general_config = general_config
        self.detector = detector
        self.face_engine = face_engine
        self.database = database
        self.stop_event = stop_event

        self.timezone = ZoneInfo(timezone_name)

        self.camera_id = camera_config["id"]
        self.role = camera_config["role"].lower()
        self.source = open_video_source(
            camera_config["source"]
        )

        self.is_live = source_is_live(self.source)

        self.test_mode = bool(
            general_config.get("test_mode", False)
        )

        self.business_start = parse_clock(
            general_config["business_start"]
        )

        self.business_end = parse_clock(
            general_config["business_end"]
        )

        self.process_every_nth_frame = max(
            1,
            int(
                general_config.get(
                    "process_every_nth_frame",
                    1,
                )
            ),
        )

        self.face_every_nth_frame = max(
            1,
            int(
                general_config.get(
                    "face_every_nth_frame",
                    3,
                )
            ),
        )

        self.show_preview = bool(
            general_config.get(
                "show_preview",
                False,
            )
        )

        self.crossing_monitor = CrossingMonitor(
            normalized_line=camera_config[
                "counting_line"
            ],
            allowed_crossing=camera_config.get(
                "allowed_crossing",
                "any",
            ),
        )

    def within_business_hours(self) -> bool:
        if self.test_mode:
            return True

        current_time = datetime.now(
            self.timezone
        ).time()

        return (
            self.business_start
            <= current_time
            < self.business_end
        )

    def connect(self):
        logger.info(
            "Connecting to camera source: %s",
            self.camera_id,
        )

        capture = cv2.VideoCapture(self.source)

        if self.is_live:
            capture.set(
                cv2.CAP_PROP_BUFFERSIZE,
                2,
            )

        return capture

    def run(self):
        capture = None
        frame_number = 0
        tracker = None

        while not self.stop_event.is_set():
            if not self.within_business_hours():
                time.sleep(10)
                continue

            if (
                capture is None
                or not capture.isOpened()
            ):
                capture = self.connect()

                if not capture.isOpened():
                    logger.error(
                        "Could not open camera: %s",
                        self.camera_id,
                    )

                    capture.release()
                    capture = None
                    time.sleep(5)
                    continue

                fps = capture.get(cv2.CAP_PROP_FPS)

                if not fps or fps <= 0:
                    fps = 25

                tracker = PersonTracker(
                    frame_rate=int(fps)
                )

            success, frame = capture.read()

            if not success:
                if self.is_live:
                    logger.warning(
                        "Camera disconnected: %s",
                        self.camera_id,
                    )

                    capture.release()
                    capture = None
                    time.sleep(2)
                    continue

                logger.info(
                    "Video completed: %s",
                    self.camera_id,
                )
                break

            frame_number += 1

            if (
                frame_number
                % self.process_every_nth_frame
                != 0
            ):
                continue

            boxes, confidences = (
                self.detector.detect(frame)
            )

            tracks = tracker.update(
                boxes,
                confidences,
            )

            faces = []

            if (
                frame_number
                % self.face_every_nth_frame
                == 0
            ):
                faces = (
                    self.face_engine.detect_and_embed(
                        frame
                    )
                )

            face_assignments = (
                assign_faces_to_tracks(
                    tracks,
                    faces,
                )
            )

            crossings = (
                self.crossing_monitor.update(
                    frame=frame,
                    frame_number=frame_number,
                    tracks=tracks,
                    face_assignments=face_assignments,
                )
            )

            timestamp = datetime.now(
                self.timezone
            )

            for track, embedding in crossings:
                self.database.process_crossing(
                    camera_id=self.camera_id,
                    direction=self.role,
                    timestamp=timestamp,
                    embedding=embedding,
                )

            if self.show_preview:
                self.draw_preview(
                    frame,
                    tracks,
                )

                cv2.imshow(
                    self.camera_id,
                    frame,
                )

                if cv2.waitKey(1) == 27:
                    self.stop_event.set()
                    break

        if capture is not None:
            capture.release()

        if self.show_preview:
            cv2.destroyWindow(self.camera_id)

    def draw_preview(
        self,
        frame: np.ndarray,
        tracks: list[PersonTrack],
    ):
        line = self.crossing_monitor.actual_line(
            frame
        )

        cv2.line(
            frame,
            (int(line[0]), int(line[1])),
            (int(line[2]), int(line[3])),
            (0, 255, 255),
            3,
        )

        for track in tracks:
            x1, y1, x2, y2 = (
                track.box.astype(int)
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Track {track.track_id}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )


# ============================================================
# MIDNIGHT BIOMETRIC DELETION
# ============================================================

class RetentionWorker(threading.Thread):
    def __init__(
        self,
        database: FootfallDatabase,
        stop_event: threading.Event,
        timezone_name: str,
    ):
        super().__init__(
            name="retention-worker",
            daemon=True,
        )

        self.database = database
        self.stop_event = stop_event
        self.timezone = ZoneInfo(timezone_name)

    def run(self):
        last_seen_day = datetime.now(
            self.timezone
        ).date()

        while not self.stop_event.wait(30):
            current_day = datetime.now(
                self.timezone
            ).date()

            if current_day != last_seen_day:
                self.database.purge_old_identities()
                last_seen_day = current_day


# ============================================================
# APPLICATION
# ============================================================

def validate_config(config: dict):
    cameras = config.get("cameras", [])

    if len(cameras) != 2:
        raise ValueError(
            "The pilot configuration must contain "
            "exactly two cameras."
        )

    roles = {
        camera["role"].lower()
        for camera in cameras
    }

    if roles != {"entry", "exit"}:
        raise ValueError(
            "One camera must have role 'entry' "
            "and one must have role 'exit'."
        )


def main(config_path: str):
    config = load_config(config_path)
    validate_config(config)

    location_config = config["location"]
    operation_config = config["operation"]
    model_config = config["models"]
    database_config = config["database"]

    timezone_name = location_config.get(
        "timezone",
        "Africa/Lagos",
    )

    stop_event = threading.Event()

    database = FootfallDatabase(
        events_path=database_config["events"],
        identities_path=database_config[
            "identities"
        ],
        timezone_name=timezone_name,
        match_threshold=operation_config[
            "face_match_threshold"
        ],
        repeat_visit_hours=operation_config[
            "repeat_visit_hours"
        ],
        location_id=location_config["id"],
    )

    person_detector = YOLOXPersonDetector(
        model_path=model_config[
            "person_detector"
        ],
        input_width=model_config.get(
            "person_input_width",
            640,
        ),
        input_height=model_config.get(
            "person_input_height",
            640,
        ),
        confidence_threshold=model_config.get(
            "person_confidence",
            0.35,
        ),
        nms_threshold=model_config.get(
            "person_nms_threshold",
            0.45,
        ),
    )

    face_engine = FaceEngine(
        detector_path=model_config[
            "face_detector"
        ],
        recognizer_path=model_config[
            "face_recognizer"
        ],
        detection_threshold=operation_config.get(
            "face_detection_threshold",
            0.85,
        ),
        minimum_face_width=operation_config.get(
            "minimum_face_width",
            60,
        ),
    )

    workers = []

    for camera_config in config["cameras"]:
        worker = CameraWorker(
            camera_config=camera_config,
            general_config=operation_config,
            detector=person_detector,
            face_engine=face_engine,
            database=database,
            stop_event=stop_event,
            timezone_name=timezone_name,
        )

        workers.append(worker)

    retention_worker = RetentionWorker(
        database=database,
        stop_event=stop_event,
        timezone_name=timezone_name,
    )

    def stop_application(*_):
        logger.info("Stopping application...")
        stop_event.set()

    signal.signal(
        signal.SIGINT,
        stop_application,
    )

    signal.signal(
        signal.SIGTERM,
        stop_application,
    )

    retention_worker.start()

    for worker in workers:
        worker.start()

    try:
        while (
            not stop_event.is_set()
            and any(
                worker.is_alive()
                for worker in workers
            )
        ):
            counters = database.get_counters()

            if counters:
                logger.info(
                    "Current counters: %s",
                    json.dumps(counters),
                )

            time.sleep(10)

    finally:
        stop_event.set()

        for worker in workers:
            worker.join(timeout=10)

        retention_worker.join(timeout=5)

        final_counters = (
            database.get_counters()
        )

        logger.info(
            "Final counters: %s",
            json.dumps(final_counters),
        )

        database.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Two-camera automated footfall pilot"
        )
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )

    arguments = parser.parse_args()

    main(arguments.config)