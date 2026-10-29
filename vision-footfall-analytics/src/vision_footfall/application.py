from __future__ import annotations

import json
import logging
import signal
import threading
import time

import cv2

from .camera import CameraWorker
from .config import load_config
from .detection import YOLOXPersonDetector
from .faces import FaceEngine
from .postgres_repository import PostgresFootfallRepository
from .retention import RetentionWorker
from .sqlite_repository import FootfallDatabase

logger = logging.getLogger(__name__)


def build_repository(config: dict):
    location = config["location"]
    operation = config["operation"]
    database = config["database"]
    driver = str(database.get("driver", "sqlite")).lower()

    common = {
        "timezone_name": location.get("timezone", "Africa/Lagos"),
        "match_threshold": operation["face_match_threshold"],
        "repeat_visit_hours": operation["repeat_visit_hours"],
        "location_id": location["id"],
    }

    if driver == "sqlite":
        return FootfallDatabase(
            events_path=database["events"],
            identities_path=database["identities"],
            **common,
        )

    if driver in {"postgres", "postgresql"}:
        dsn = database.get("dsn")
        if not dsn:
            raise ValueError("database.dsn is required for PostgreSQL.")
        return PostgresFootfallRepository(dsn=dsn, **common)

    raise ValueError(f"Unsupported database driver: {driver}")


def build_models(config: dict):
    models = config["models"]
    operation = config["operation"]

    detector = YOLOXPersonDetector(
        model_path=models["person_detector"],
        input_width=models.get("person_input_width", 416),
        input_height=models.get("person_input_height", 416),
        confidence_threshold=models.get("person_confidence", 0.35),
        nms_threshold=models.get("person_nms_threshold", 0.45),
    )
    face_engine = FaceEngine(
        detector_path=models["face_detector"],
        recognizer_path=models["face_recognizer"],
        detection_threshold=operation.get("face_detection_threshold", 0.85),
        minimum_face_width=operation.get("minimum_face_width", 60),
    )
    return detector, face_engine


def run(config_path: str) -> None:
    config = load_config(config_path)
    timezone_name = config["location"].get("timezone", "Africa/Lagos")
    stop_event = threading.Event()
    repository = build_repository(config)
    detector, face_engine = build_models(config)

    workers = [
        CameraWorker(
            camera_config=camera,
            general_config=config["operation"],
            detector=detector,
            face_engine=face_engine,
            database=repository,
            stop_event=stop_event,
            timezone_name=timezone_name,
        )
        for camera in config["cameras"]
    ]
    retention_worker = RetentionWorker(
        database=repository,
        stop_event=stop_event,
        timezone_name=timezone_name,
    )

    def stop_application(*_):
        logger.info("Stopping VisionFootfall.")
        stop_event.set()

    signal.signal(signal.SIGINT, stop_application)
    signal.signal(signal.SIGTERM, stop_application)

    retention_worker.start()
    for worker in workers:
        worker.start()

    try:
        while not stop_event.is_set() and any(worker.is_alive() for worker in workers):
            counters = repository.get_counters()
            if counters:
                logger.info("Current counters: %s", json.dumps(counters, default=str))
            time.sleep(10)
    finally:
        stop_event.set()
        for worker in workers:
            worker.join(timeout=10)
        retention_worker.join(timeout=5)

        logger.info(
            "Final counters: %s",
            json.dumps(repository.get_counters(), default=str),
        )
        repository.close()
        cv2.destroyAllWindows()


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
    )
