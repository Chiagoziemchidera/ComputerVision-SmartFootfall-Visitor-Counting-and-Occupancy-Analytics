from datetime import time

import pytest

from vision_footfall.config import time_in_window, validate_config


def base_config():
    return {
        "location": {"id": "TEST-01"},
        "operation": {},
        "models": {},
        "database": {},
        "cameras": [
            {
                "id": "ENTRY-01",
                "role": "entry",
                "source": "entry.mp4",
                "counting_line": [0.1, 0.6, 0.9, 0.6],
            },
            {
                "id": "ENTRY-02",
                "role": "entry",
                "source": "entry-2.mp4",
                "counting_line": [0.1, 0.6, 0.9, 0.6],
            },
            {
                "id": "EXIT-01",
                "role": "exit",
                "source": "exit.mp4",
                "counting_line": [0.1, 0.6, 0.9, 0.6],
            },
        ],
    }


def test_multiple_camera_layout_is_valid():
    validate_config(base_config())


def test_entry_and_exit_are_both_required():
    config = base_config()
    for camera in config["cameras"]:
        camera["role"] = "entry"

    with pytest.raises(ValueError, match="at least one entry and one exit"):
        validate_config(config)


def test_processing_window_can_end_at_midnight():
    start = time(6, 0)
    end = time(0, 0)

    assert time_in_window(time(6, 0), start, end)
    assert time_in_window(time(23, 59), start, end)
    assert not time_in_window(time(0, 0), start, end)
    assert not time_in_window(time(5, 59), start, end)
