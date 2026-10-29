from __future__ import annotations

import os
from datetime import datetime
from datetime import time as dt_time

import yaml


def expand_environment_variables(value):
    if isinstance(value, dict):
        return {key: expand_environment_variables(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_environment_variables(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise TypeError("The configuration file must contain a YAML mapping.")
    config = expand_environment_variables(config)
    validate_config(config)
    return config


def parse_clock(value: str) -> dt_time:
    return datetime.strptime(value, "%H:%M").time()


def time_in_window(current: dt_time, start: dt_time, end: dt_time) -> bool:
    if start == end:
        return True
    if start < end:
        return start <= current < end
    return current >= start or current < end


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
    return source.startswith(("rtsp://", "rtsps://", "http://", "https://"))


def validate_config(config: dict) -> None:
    for section in ("location", "operation", "models", "database", "cameras"):
        if section not in config:
            raise ValueError(f"Missing configuration section: {section}")

    cameras = config["cameras"]
    if not isinstance(cameras, list) or len(cameras) < 2:
        raise ValueError("At least two cameras are required.")

    camera_ids = [camera.get("id") for camera in cameras]
    if any(not camera_id for camera_id in camera_ids):
        raise ValueError("Every camera must have an id.")
    if len(camera_ids) != len(set(camera_ids)):
        raise ValueError("Camera ids must be unique.")

    roles = [str(camera.get("role", "")).lower() for camera in cameras]
    invalid_roles = sorted(set(roles) - {"entry", "exit"})
    if invalid_roles:
        raise ValueError(f"Invalid camera roles: {', '.join(invalid_roles)}")
    if "entry" not in roles or "exit" not in roles:
        raise ValueError(
            "The configuration needs at least one entry and one exit camera."
        )

    for camera in cameras:
        if not camera.get("source"):
            raise ValueError(f"Camera {camera['id']} has no video source.")
        line = camera.get("counting_line")
        if not isinstance(line, list) or len(line) != 4:
            raise ValueError(
                f"Camera {camera['id']} needs four counting-line coordinates."
            )
        if not all(0 <= float(value) <= 1 for value in line):
            raise ValueError(
                f"Camera {camera['id']} counting-line coordinates must be 0 to 1."
            )
