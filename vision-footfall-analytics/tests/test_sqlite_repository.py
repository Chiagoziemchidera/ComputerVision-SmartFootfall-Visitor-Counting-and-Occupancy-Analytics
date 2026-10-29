from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from vision_footfall.sqlite_repository import FootfallDatabase


def make_repository(tmp_path):
    return FootfallDatabase(
        events_path=str(tmp_path / "events.sqlite"),
        identities_path=str(tmp_path / "identities.sqlite"),
        timezone_name="Africa/Lagos",
        match_threshold=0.8,
        repeat_visit_hours=2,
        location_id="TEST-LOCATION",
    )


def test_unique_occupancy_and_repeat_rules(tmp_path):
    repository = make_repository(tmp_path)
    timezone = ZoneInfo("Africa/Lagos")
    visitor = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    start = datetime(2026, 10, 20, 9, 0, tzinfo=timezone)

    repository.process_crossing("ENTRY-01", "entry", start, visitor)
    repository.process_crossing("EXIT-01", "exit", start + timedelta(hours=1), visitor)
    quick_return = repository.process_crossing(
        "ENTRY-01", "entry", start + timedelta(hours=1, minutes=30), visitor
    )
    repository.process_crossing("EXIT-01", "exit", start + timedelta(hours=2), visitor)
    qualified_return = repository.process_crossing(
        "ENTRY-01", "entry", start + timedelta(hours=5), visitor
    )

    counters = repository.get_counters("2026-10-20")
    repository.close()

    assert quick_return["repeat_increment"] == 1
    assert quick_return["qualified_repeat_increment"] == 0
    assert qualified_return["repeat_increment"] == 1
    assert qualified_return["qualified_repeat_increment"] == 1
    assert counters["verified_daily_unique"] == 1
    assert counters["physical_entries"] == 3
    assert counters["physical_exits"] == 2
    assert counters["repeat_entries"] == 2
    assert counters["qualified_repeat_visits"] == 1
    assert counters["occupancy"] == 1
    assert counters["peak_occupancy"] == 1


def test_exit_seen_before_first_entry_becomes_unique_on_entry(tmp_path):
    repository = make_repository(tmp_path)
    timezone = ZoneInfo("Africa/Lagos")
    visitor = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    timestamp = datetime(2026, 10, 20, 8, 0, tzinfo=timezone)

    repository.process_crossing("EXIT-01", "exit", timestamp, visitor)
    result = repository.process_crossing(
        "ENTRY-01", "entry", timestamp + timedelta(minutes=30), visitor
    )
    counters = repository.get_counters("2026-10-20")
    repository.close()

    assert result["identity_status"] == "first_entry_after_exit"
    assert result["unique_increment"] == 1
    assert counters["verified_daily_unique"] == 1


def test_previous_day_identities_are_deleted(tmp_path):
    repository = make_repository(tmp_path)
    timezone = ZoneInfo("Africa/Lagos")
    timestamp = datetime.now(timezone) - timedelta(days=2)
    visitor = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    repository.create_identity(
        day=timestamp.date().isoformat(),
        timestamp=timestamp,
        embedding=visitor,
        has_entered=True,
    )
    repository.identities.commit()

    deleted = repository.purge_old_identities()
    remaining = repository.identities.execute(
        "SELECT COUNT(*) FROM daily_identities"
    ).fetchone()[0]
    repository.close()

    assert deleted == 1
    assert remaining == 0
