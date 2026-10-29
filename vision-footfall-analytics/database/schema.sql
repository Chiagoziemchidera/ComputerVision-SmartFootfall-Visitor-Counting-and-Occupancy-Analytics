CREATE TABLE IF NOT EXISTS counters (
    location_id TEXT NOT NULL,
    day DATE NOT NULL,
    physical_entries INTEGER NOT NULL DEFAULT 0,
    physical_exits INTEGER NOT NULL DEFAULT 0,
    verified_daily_unique INTEGER NOT NULL DEFAULT 0,
    unresolved_identity_entries INTEGER NOT NULL DEFAULT 0,
    repeat_entries INTEGER NOT NULL DEFAULT 0,
    qualified_repeat_visits INTEGER NOT NULL DEFAULT 0,
    occupancy INTEGER NOT NULL DEFAULT 0,
    peak_occupancy INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (location_id, day)
);

CREATE TABLE IF NOT EXISTS crossing_events (
    event_id TEXT PRIMARY KEY,
    location_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    day DATE NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('entry', 'exit')),
    identity_status TEXT NOT NULL,
    match_score DOUBLE PRECISION,
    unique_increment INTEGER NOT NULL DEFAULT 0,
    repeat_increment INTEGER NOT NULL DEFAULT 0,
    qualified_repeat_increment INTEGER NOT NULL DEFAULT 0,
    occupancy_after INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS occupancy_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    location_id TEXT NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    occupancy INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_identities (
    location_id TEXT NOT NULL,
    day DATE NOT NULL,
    visitor_id TEXT NOT NULL,
    embedding BYTEA NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 1,
    has_entered BOOLEAN NOT NULL DEFAULT FALSE,
    first_seen TIMESTAMPTZ NOT NULL,
    last_entry TIMESTAMPTZ,
    last_exit TIMESTAMPTZ,
    is_inside BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (location_id, day, visitor_id)
);

CREATE INDEX IF NOT EXISTS idx_crossing_events_location_time
ON crossing_events (location_id, event_time);

CREATE INDEX IF NOT EXISTS idx_daily_identities_location_day
ON daily_identities (location_id, day);
