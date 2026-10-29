import numpy as np

from vision_footfall.crossing import CrossingMonitor, crossing_is_allowed
from vision_footfall.entities import PersonTrack


def test_crossing_direction_rules():
    assert crossing_is_allowed(-1, 1, "negative_to_positive")
    assert crossing_is_allowed(1, -1, "positive_to_negative")
    assert not crossing_is_allowed(1, -1, "negative_to_positive")
    assert not crossing_is_allowed(1, 1, "any")


def test_monitor_records_one_crossing_per_track():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    monitor = CrossingMonitor([0.0, 0.5, 1.0, 0.5], "any")

    above = PersonTrack(7, np.array([30, 10, 50, 40]), 0.9)
    below = PersonTrack(7, np.array([30, 30, 50, 70]), 0.9)

    assert monitor.update(frame, 1, [above], {}) == []
    first_crossing = monitor.update(frame, 2, [below], {})
    second_crossing = monitor.update(frame, 3, [above], {})

    assert len(first_crossing) == 1
    assert second_crossing == []
