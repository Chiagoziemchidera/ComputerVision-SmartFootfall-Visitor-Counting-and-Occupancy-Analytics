from __future__ import annotations

import logging
import math
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from .config import open_video_source, parse_clock, source_is_live, time_in_window
from .crossing import CrossingMonitor
from .detection import YOLOXPersonDetector
from .faces import FaceEngine, assign_faces_to_tracks
from .repository import FootfallRepository
from .tracking import PersonTrack, PersonTracker

logger = logging.getLogger(__name__)


class CameraWorker(threading.Thread):

    def __init__(
        self,
        camera_config: dict,
        general_config: dict,
        detector: YOLOXPersonDetector,
        face_engine: FaceEngine,
        database: FootfallRepository,
        stop_event: threading.Event,
        timezone_name: str,
    ):
        super().__init__(name=camera_config["id"], daemon=True)
        self.camera_config = camera_config
        self.general_config = general_config
        self.detector = detector
        self.face_engine = face_engine
        self.database = database
        self.stop_event = stop_event
        self.timezone = ZoneInfo(timezone_name)
        self.camera_id = camera_config["id"]
        self.role = camera_config["role"].lower()
        self.source = open_video_source(camera_config["source"])
        self.is_live = source_is_live(self.source)
        self.test_mode = bool(general_config.get("test_mode", False))
        self.processing_start = parse_clock(
            general_config.get(
                "processing_start",
                general_config["business_start"],
            )
        )
        self.processing_end = parse_clock(
            general_config.get(
                "processing_end",
                general_config["business_end"],
            )
        )
        self.target_fps = max(1, int(general_config.get("target_fps", 15)))
        self.process_every_nth_frame = max(
            1, int(general_config.get("process_every_nth_frame", 1))
        )
        self.face_every_nth_frame = max(
            1, int(general_config.get("face_every_nth_frame", 3))
        )
        self.show_preview = bool(general_config.get("show_preview", False))
        self.crossing_monitor = CrossingMonitor(
            normalized_line=camera_config["counting_line"],
            allowed_crossing=camera_config.get("allowed_crossing", "any"),
        )

    def within_processing_hours(self) -> bool:
        if self.test_mode:
            return True
        current_time = datetime.now(self.timezone).time()
        return time_in_window(
            current_time,
            self.processing_start,
            self.processing_end,
        )

    def connect(self):
        logger.info("Connecting to camera source: %s", self.camera_id)
        capture = cv2.VideoCapture(self.source)
        if self.is_live:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return capture

    def run(self):
        capture = None
        frame_number = 0
        frame_stride = self.process_every_nth_frame
        tracker = None
        while not self.stop_event.is_set():
            if not self.within_processing_hours():
                if capture is not None:
                    capture.release()
                    capture = None
                time.sleep(10)
                continue
            if capture is None or not capture.isOpened():
                capture = self.connect()
                if not capture.isOpened():
                    logger.error("Could not open camera: %s", self.camera_id)
                    capture.release()
                    capture = None
                    time.sleep(5)
                    continue
                fps = capture.get(cv2.CAP_PROP_FPS)
                if not fps or fps <= 0:
                    fps = self.target_fps
                target_stride = max(1, math.ceil(fps / self.target_fps))
                frame_stride = max(self.process_every_nth_frame, target_stride)
                tracker = PersonTracker(frame_rate=max(1, round(fps / frame_stride)))
            success, frame = capture.read()
            if not success:
                if self.is_live:
                    logger.warning("Camera disconnected: %s", self.camera_id)
                    capture.release()
                    capture = None
                    time.sleep(2)
                    continue
                logger.info("Video completed: %s", self.camera_id)
                break
            frame_number += 1
            if frame_number % frame_stride != 0:
                continue
            boxes, confidences = self.detector.detect(frame)
            tracks = tracker.update(boxes, confidences)
            faces = []
            if frame_number % self.face_every_nth_frame == 0:
                faces = self.face_engine.detect_and_embed(frame)
            face_assignments = assign_faces_to_tracks(tracks, faces)
            crossings = self.crossing_monitor.update(
                frame=frame,
                frame_number=frame_number,
                tracks=tracks,
                face_assignments=face_assignments,
            )
            timestamp = datetime.now(self.timezone)
            for track, embedding in crossings:
                self.database.process_crossing(
                    camera_id=self.camera_id,
                    direction=self.role,
                    timestamp=timestamp,
                    embedding=embedding,
                )
            if self.show_preview:
                self.draw_preview(frame, tracks)
                cv2.imshow(self.camera_id, frame)
                if cv2.waitKey(1) == 27:
                    self.stop_event.set()
                    break
        if capture is not None:
            capture.release()
        if self.show_preview:
            cv2.destroyWindow(self.camera_id)

    def draw_preview(self, frame: np.ndarray, tracks: list[PersonTrack]):
        line = self.crossing_monitor.actual_line(frame)
        cv2.line(
            frame,
            (int(line[0]), int(line[1])),
            (int(line[2]), int(line[3])),
            (0, 255, 255),
            3,
        )
        for track in tracks:
            x1, y1, x2, y2 = track.box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Track {track.track_id}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
