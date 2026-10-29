from __future__ import annotations

import argparse
import os

import psycopg

CREATE_REPORTING_TABLE = """
CREATE TABLE IF NOT EXISTS footfall_hourly (
    location_id TEXT NOT NULL,
    hour_start TIMESTAMPTZ NOT NULL,
    physical_entries INTEGER NOT NULL,
    physical_exits INTEGER NOT NULL,
    daily_unique_increments INTEGER NOT NULL,
    repeat_entries INTEGER NOT NULL,
    qualified_repeat_visits INTEGER NOT NULL,
    unresolved_entries INTEGER NOT NULL,
    peak_occupancy INTEGER NOT NULL,
    refreshed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (location_id, hour_start)
)
"""


UPSERT_HOURLY_DATA = """
INSERT INTO footfall_hourly (
    location_id,
    hour_start,
    physical_entries,
    physical_exits,
    daily_unique_increments,
    repeat_entries,
    qualified_repeat_visits,
    unresolved_entries,
    peak_occupancy,
    refreshed_at
)
SELECT
    location_id,
    DATE_TRUNC('hour', event_time) AS hour_start,
    COUNT(*) FILTER (WHERE direction = 'entry') AS physical_entries,
    COUNT(*) FILTER (WHERE direction = 'exit') AS physical_exits,
    SUM(unique_increment) AS daily_unique_increments,
    SUM(repeat_increment) AS repeat_entries,
    SUM(qualified_repeat_increment) AS qualified_repeat_visits,
    COUNT(*) FILTER (
        WHERE direction = 'entry' AND identity_status = 'unresolved'
    ) AS unresolved_entries,
    MAX(occupancy_after) AS peak_occupancy,
    NOW()
FROM crossing_events
WHERE event_time >= NOW() - (%s * INTERVAL '1 hour')
GROUP BY location_id, DATE_TRUNC('hour', event_time)
ON CONFLICT (location_id, hour_start)
DO UPDATE SET
    physical_entries = EXCLUDED.physical_entries,
    physical_exits = EXCLUDED.physical_exits,
    daily_unique_increments = EXCLUDED.daily_unique_increments,
    repeat_entries = EXCLUDED.repeat_entries,
    qualified_repeat_visits = EXCLUDED.qualified_repeat_visits,
    unresolved_entries = EXCLUDED.unresolved_entries,
    peak_occupancy = EXCLUDED.peak_occupancy,
    refreshed_at = NOW()
"""


def run_etl(dsn: str, lookback_hours: int = 48) -> None:
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute(CREATE_REPORTING_TABLE)
        cursor.execute(UPSERT_HOURLY_DATA, (lookback_hours,))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hourly footfall reporting data")
    parser.add_argument("--lookback-hours", type=int, default=48)
    arguments = parser.parse_args()

    dsn = os.getenv("FOOTFALL_POSTGRES_DSN")
    if not dsn:
        raise RuntimeError("FOOTFALL_POSTGRES_DSN is not set.")

    run_etl(dsn, arguments.lookback_hours)


if __name__ == "__main__":
    main()
