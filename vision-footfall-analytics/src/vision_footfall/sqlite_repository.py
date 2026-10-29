from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .embeddings import cosine_similarity, normalize_embedding

logger = logging.getLogger(__name__)


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
        Path(events_path).parent.mkdir(parents=True, exist_ok=True)
        Path(identities_path).parent.mkdir(parents=True, exist_ok=True)
        self.timezone = ZoneInfo(timezone_name)
        self.match_threshold = float(match_threshold)
        self.repeat_interval = timedelta(hours=float(repeat_visit_hours))
        self.location_id = location_id
        self.lock = threading.RLock()
        self.events = sqlite3.connect(events_path, check_same_thread=False)
        self.identities = sqlite3.connect(identities_path, check_same_thread=False)
        self.events.row_factory = sqlite3.Row
        self.identities.row_factory = sqlite3.Row
        self.events.execute("PRAGMA journal_mode=WAL")
        self.identities.execute("PRAGMA journal_mode=WAL")
        self.identities.execute("PRAGMA secure_delete=ON")
        self.create_tables()
        self.purge_old_identities()

    def create_tables(self):
        with self.events:
            self.events.executescript(
                "\n                CREATE TABLE IF NOT EXISTS counters (\n                    day TEXT PRIMARY KEY,\n                    location_id TEXT NOT NULL,\n                    physical_entries INTEGER NOT NULL DEFAULT 0,\n                    physical_exits INTEGER NOT NULL DEFAULT 0,\n                    verified_daily_unique INTEGER NOT NULL DEFAULT 0,\n                    unresolved_identity_entries INTEGER NOT NULL DEFAULT 0,\n                    repeat_entries INTEGER NOT NULL DEFAULT 0,\n                    qualified_repeat_visits INTEGER NOT NULL DEFAULT 0,\n                    occupancy INTEGER NOT NULL DEFAULT 0,\n                    peak_occupancy INTEGER NOT NULL DEFAULT 0,\n                    updated_at TEXT NOT NULL\n                );\n\n                CREATE TABLE IF NOT EXISTS crossing_events (\n                    event_id TEXT PRIMARY KEY,\n                    location_id TEXT NOT NULL,\n                    camera_id TEXT NOT NULL,\n                    event_time TEXT NOT NULL,\n                    day TEXT NOT NULL,\n                    direction TEXT NOT NULL,\n                    identity_status TEXT NOT NULL,\n                    match_score REAL,\n                    unique_increment INTEGER NOT NULL DEFAULT 0,\n                    repeat_increment INTEGER NOT NULL DEFAULT 0,\n                    qualified_repeat_increment INTEGER NOT NULL DEFAULT 0,\n                    occupancy_after INTEGER NOT NULL,\n                    created_at TEXT NOT NULL\n                );\n\n                CREATE TABLE IF NOT EXISTS occupancy_snapshots (\n                    snapshot_id TEXT PRIMARY KEY,\n                    location_id TEXT NOT NULL,\n                    event_time TEXT NOT NULL,\n                    occupancy INTEGER NOT NULL\n                );\n                "
            )
        with self.identities:
            self.identities.executescript(
                "\n                CREATE TABLE IF NOT EXISTS daily_identities (\n                    day TEXT NOT NULL,\n                    visitor_id TEXT NOT NULL,\n                    embedding BLOB NOT NULL,\n                    sample_count INTEGER NOT NULL DEFAULT 1,\n                    has_entered INTEGER NOT NULL DEFAULT 0,\n                    first_seen TEXT NOT NULL,\n                    last_entry TEXT,\n                    last_exit TEXT,\n                    is_inside INTEGER NOT NULL DEFAULT 0,\n                    PRIMARY KEY (day, visitor_id)\n                );\n                "
            )

    def ensure_counter(self, day: str, timestamp: datetime):
        self.events.execute(
            "\n            INSERT OR IGNORE INTO counters (\n                day,\n                location_id,\n                updated_at\n            )\n            VALUES (?, ?, ?)\n            ",
            (day, self.location_id, timestamp.isoformat()),
        )

    def find_identity(
        self, day: str, embedding: np.ndarray
    ) -> tuple[sqlite3.Row | None, float | None]:
        rows = self.identities.execute(
            "\n            SELECT *\n            FROM daily_identities\n            WHERE day = ?\n            ",
            (day,),
        ).fetchall()
        if not rows:
            return (None, None)
        best_row = None
        best_score = -1.0
        for row in rows:
            saved_embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            score = cosine_similarity(embedding, saved_embedding)
            if score > best_score:
                best_score = score
                best_row = row
        if best_score >= self.match_threshold:
            return (best_row, best_score)
        return (None, best_score)

    def create_identity(
        self, day: str, timestamp: datetime, embedding: np.ndarray, has_entered: bool
    ) -> str:
        visitor_id = f"{day}-{uuid.uuid4().hex[:12].upper()}"
        embedding = normalize_embedding(embedding)
        self.identities.execute(
            "\n            INSERT INTO daily_identities (\n                day,\n                visitor_id,\n                embedding,\n                sample_count,\n                has_entered,\n                first_seen,\n                last_entry,\n                last_exit,\n                is_inside\n            )\n            VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)\n            ",
            (
                day,
                visitor_id,
                embedding.astype(np.float32).tobytes(),
                1 if has_entered else 0,
                timestamp.isoformat(),
                timestamp.isoformat() if has_entered else None,
                None if has_entered else timestamp.isoformat(),
                1 if has_entered else 0,
            ),
        )
        return visitor_id

    def update_identity_embedding(self, row: sqlite3.Row, new_embedding: np.ndarray):
        old_embedding = np.frombuffer(row["embedding"], dtype=np.float32)
        sample_count = int(row["sample_count"])
        updated_embedding = normalize_embedding(
            (old_embedding * sample_count + normalize_embedding(new_embedding))
            / (sample_count + 1)
        )
        self.identities.execute(
            "\n            UPDATE daily_identities\n            SET\n                embedding = ?,\n                sample_count = sample_count + 1\n            WHERE\n                day = ?\n                AND visitor_id = ?\n            ",
            (
                updated_embedding.astype(np.float32).tobytes(),
                row["day"],
                row["visitor_id"],
            ),
        )

    def process_crossing(
        self,
        camera_id: str,
        direction: str,
        timestamp: datetime,
        embedding: np.ndarray | None,
    ) -> dict:
        with self.lock:
            day = timestamp.astimezone(self.timezone).date().isoformat()
            self.ensure_counter(day, timestamp)
            counter = self.events.execute(
                "\n                SELECT *\n                FROM counters\n                WHERE day = ?\n                ",
                (day,),
            ).fetchone()
            occupancy = int(counter["occupancy"])
            peak_occupancy = int(counter["peak_occupancy"])
            identity_status = "unresolved"
            match_score = None
            unique_increment = 0
            repeat_increment = 0
            qualified_repeat_increment = 0
            identity_row = None
            if embedding is not None:
                embedding = normalize_embedding(embedding)
                identity_row, match_score = self.find_identity(day, embedding)
            if direction == "entry":
                occupancy += 1
                peak_occupancy = max(peak_occupancy, occupancy)
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
                    self.update_identity_embedding(identity_row, embedding)
                    has_entered = bool(identity_row["has_entered"])
                    if not has_entered:
                        identity_status = "first_entry_after_exit"
                        unique_increment = 1
                    else:
                        identity_status = "repeat"
                        repeat_increment = 1
                        if identity_row["last_exit"]:
                            last_exit = datetime.fromisoformat(
                                identity_row["last_exit"]
                            )
                            time_away = timestamp - last_exit
                            if time_away >= self.repeat_interval:
                                qualified_repeat_increment = 1
                    self.identities.execute(
                        "\n                        UPDATE daily_identities\n                        SET\n                            has_entered = 1,\n                            last_entry = ?,\n                            is_inside = 1\n                        WHERE\n                            day = ?\n                            AND visitor_id = ?\n                        ",
                        (timestamp.isoformat(), day, identity_row["visitor_id"]),
                    )
                self.events.execute(
                    "\n                    UPDATE counters\n                    SET\n                        physical_entries =\n                            physical_entries + 1,\n                        verified_daily_unique =\n                            verified_daily_unique + ?,\n                        unresolved_identity_entries =\n                            unresolved_identity_entries + ?,\n                        repeat_entries =\n                            repeat_entries + ?,\n                        qualified_repeat_visits =\n                            qualified_repeat_visits + ?,\n                        occupancy = ?,\n                        peak_occupancy = ?,\n                        updated_at = ?\n                    WHERE day = ?\n                    ",
                    (
                        unique_increment,
                        1 if identity_status == "unresolved" else 0,
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
                    identity_status = "exit_identity_created"
                else:
                    identity_status = "matched_exit"
                    self.update_identity_embedding(identity_row, embedding)
                    self.identities.execute(
                        "\n                        UPDATE daily_identities\n                        SET\n                            last_exit = ?,\n                            is_inside = 0\n                        WHERE\n                            day = ?\n                            AND visitor_id = ?\n                        ",
                        (timestamp.isoformat(), day, identity_row["visitor_id"]),
                    )
                self.events.execute(
                    "\n                    UPDATE counters\n                    SET\n                        physical_exits =\n                            physical_exits + 1,\n                        occupancy = ?,\n                        updated_at = ?\n                    WHERE day = ?\n                    ",
                    (occupancy, timestamp.isoformat(), day),
                )
            else:
                raise ValueError(f"Invalid direction: {direction}")
            event_id = str(uuid.uuid4())
            self.events.execute(
                "\n                INSERT INTO crossing_events (\n                    event_id,\n                    location_id,\n                    camera_id,\n                    event_time,\n                    day,\n                    direction,\n                    identity_status,\n                    match_score,\n                    unique_increment,\n                    repeat_increment,\n                    qualified_repeat_increment,\n                    occupancy_after,\n                    created_at\n                )\n                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n                ",
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
                    datetime.now(self.timezone).isoformat(),
                ),
            )
            self.events.execute(
                "\n                INSERT INTO occupancy_snapshots (\n                    snapshot_id,\n                    location_id,\n                    event_time,\n                    occupancy\n                )\n                VALUES (?, ?, ?, ?)\n                ",
                (str(uuid.uuid4()), self.location_id, timestamp.isoformat(), occupancy),
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
                "qualified_repeat_increment": qualified_repeat_increment,
                "occupancy_after": occupancy,
            }
            logger.info("Crossing event: %s", json.dumps(result))
            return result

    def get_counters(self, day: str | None = None) -> dict:
        with self.lock:
            if day is None:
                day = datetime.now(self.timezone).date().isoformat()
            row = self.events.execute(
                "\n                SELECT *\n                FROM counters\n                WHERE day = ?\n                ",
                (day,),
            ).fetchone()
            return dict(row) if row else {}

    def purge_old_identities(self) -> int:
        with self.lock:
            current_day = datetime.now(self.timezone).date().isoformat()
            deleted = self.identities.execute(
                "\n                DELETE FROM daily_identities\n                WHERE day < ?\n                ",
                (current_day,),
            ).rowcount
            self.identities.commit()
            if deleted:
                self.identities.execute("VACUUM")
                self.identities.commit()
                logger.info("Deleted %s previous-day biometric identities.", deleted)
            return deleted

    def close(self):
        with self.lock:
            self.events.close()
            self.identities.close()
