from __future__ import annotations

import logging
import threading
import uuid
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from .embeddings import cosine_similarity, normalize_embedding

logger = logging.getLogger(__name__)


class PostgresFootfallRepository:
    def __init__(
        self,
        dsn: str,
        timezone_name: str,
        match_threshold: float,
        repeat_visit_hours: float,
        location_id: str,
    ):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError("PostgreSQL support requires psycopg[binary].") from exc

        self.timezone = ZoneInfo(timezone_name)
        self.match_threshold = float(match_threshold)
        self.repeat_interval = timedelta(hours=float(repeat_visit_hours))
        self.location_id = location_id
        self.lock = threading.RLock()
        self.connection = psycopg.connect(dsn, row_factory=dict_row)

        self.create_tables()
        self.purge_old_identities()

    def create_tables(self) -> None:
        statements = [
            """
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
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS crossing_events (
                event_id TEXT PRIMARY KEY,
                location_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                event_time TIMESTAMPTZ NOT NULL,
                day DATE NOT NULL,
                direction TEXT NOT NULL,
                identity_status TEXT NOT NULL,
                match_score DOUBLE PRECISION,
                unique_increment INTEGER NOT NULL DEFAULT 0,
                repeat_increment INTEGER NOT NULL DEFAULT 0,
                qualified_repeat_increment INTEGER NOT NULL DEFAULT 0,
                occupancy_after INTEGER NOT NULL,
                created_at TIMESTAMPTZ NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS occupancy_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                location_id TEXT NOT NULL,
                event_time TIMESTAMPTZ NOT NULL,
                occupancy INTEGER NOT NULL
            )
            """,
            """
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
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_crossing_events_location_time
            ON crossing_events (location_id, event_time)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_daily_identities_location_day
            ON daily_identities (location_id, day)
            """,
        ]

        with self.connection.transaction(), self.connection.cursor() as cursor:
            for statement in statements:
                cursor.execute(statement)

    def ensure_counter(self, day: date, timestamp: datetime) -> None:
        self.connection.execute(
            """
            INSERT INTO counters (location_id, day, updated_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (location_id, day) DO NOTHING
            """,
            (self.location_id, day, timestamp),
        )

    def find_identity(
        self,
        day: date,
        embedding: np.ndarray,
    ) -> tuple[dict | None, float | None]:
        rows = self.connection.execute(
            """
            SELECT *
            FROM daily_identities
            WHERE location_id = %s AND day = %s
            """,
            (self.location_id, day),
        ).fetchall()

        best_row = None
        best_score = -1.0

        for row in rows:
            saved = np.frombuffer(bytes(row["embedding"]), dtype=np.float32)
            score = cosine_similarity(embedding, saved)
            if score > best_score:
                best_row = row
                best_score = score

        if best_row is not None and best_score >= self.match_threshold:
            return best_row, best_score

        return None, None if not rows else best_score

    def create_identity(
        self,
        day: date,
        timestamp: datetime,
        embedding: np.ndarray,
        has_entered: bool,
    ) -> str:
        visitor_id = f"{day.isoformat()}-{uuid.uuid4().hex[:12].upper()}"
        embedding_bytes = normalize_embedding(embedding).astype(np.float32).tobytes()

        self.connection.execute(
            """
            INSERT INTO daily_identities (
                location_id, day, visitor_id, embedding, sample_count,
                has_entered, first_seen, last_entry, last_exit, is_inside
            )
            VALUES (%s, %s, %s, %s, 1, %s, %s, %s, %s, %s)
            """,
            (
                self.location_id,
                day,
                visitor_id,
                embedding_bytes,
                has_entered,
                timestamp,
                timestamp if has_entered else None,
                None if has_entered else timestamp,
                has_entered,
            ),
        )
        return visitor_id

    def update_identity_embedding(
        self,
        row: dict,
        new_embedding: np.ndarray,
    ) -> None:
        old_embedding = np.frombuffer(bytes(row["embedding"]), dtype=np.float32)
        sample_count = int(row["sample_count"])
        combined = (
            old_embedding * sample_count + normalize_embedding(new_embedding)
        ) / (sample_count + 1)
        embedding_bytes = normalize_embedding(combined).astype(np.float32).tobytes()

        self.connection.execute(
            """
            UPDATE daily_identities
            SET embedding = %s, sample_count = sample_count + 1
            WHERE location_id = %s AND day = %s AND visitor_id = %s
            """,
            (
                embedding_bytes,
                self.location_id,
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
        direction = direction.lower()
        if direction not in {"entry", "exit"}:
            raise ValueError(f"Invalid direction: {direction}")

        with self.lock, self.connection.transaction():
            local_timestamp = timestamp.astimezone(self.timezone)
            day = local_timestamp.date()
            self.ensure_counter(day, local_timestamp)

            counter = self.connection.execute(
                """
                SELECT *
                FROM counters
                WHERE location_id = %s AND day = %s
                FOR UPDATE
                """,
                (self.location_id, day),
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
                    self.create_identity(day, local_timestamp, embedding, True)
                    identity_status = "new"
                    unique_increment = 1
                else:
                    self.update_identity_embedding(identity_row, embedding)
                    if not identity_row["has_entered"]:
                        identity_status = "first_entry_after_exit"
                        unique_increment = 1
                    else:
                        identity_status = "repeat"
                        repeat_increment = 1
                        last_exit = identity_row["last_exit"]
                        if (
                            last_exit
                            and local_timestamp - last_exit >= self.repeat_interval
                        ):
                            qualified_repeat_increment = 1

                    self.connection.execute(
                        """
                        UPDATE daily_identities
                        SET has_entered = TRUE, last_entry = %s, is_inside = TRUE
                        WHERE location_id = %s AND day = %s AND visitor_id = %s
                        """,
                        (
                            local_timestamp,
                            self.location_id,
                            day,
                            identity_row["visitor_id"],
                        ),
                    )

                self.connection.execute(
                    """
                    UPDATE counters
                    SET physical_entries = physical_entries + 1,
                        verified_daily_unique = verified_daily_unique + %s,
                        unresolved_identity_entries =
                            unresolved_identity_entries + %s,
                        repeat_entries = repeat_entries + %s,
                        qualified_repeat_visits =
                            qualified_repeat_visits + %s,
                        occupancy = %s,
                        peak_occupancy = %s,
                        updated_at = %s
                    WHERE location_id = %s AND day = %s
                    """,
                    (
                        unique_increment,
                        int(identity_status == "unresolved"),
                        repeat_increment,
                        qualified_repeat_increment,
                        occupancy,
                        peak_occupancy,
                        local_timestamp,
                        self.location_id,
                        day,
                    ),
                )
            else:
                occupancy = max(0, occupancy - 1)

                if embedding is None:
                    identity_status = "unresolved"
                elif identity_row is None:
                    self.create_identity(day, local_timestamp, embedding, False)
                    identity_status = "exit_identity_created"
                else:
                    identity_status = "matched_exit"
                    self.update_identity_embedding(identity_row, embedding)
                    self.connection.execute(
                        """
                        UPDATE daily_identities
                        SET last_exit = %s, is_inside = FALSE
                        WHERE location_id = %s AND day = %s AND visitor_id = %s
                        """,
                        (
                            local_timestamp,
                            self.location_id,
                            day,
                            identity_row["visitor_id"],
                        ),
                    )

                self.connection.execute(
                    """
                    UPDATE counters
                    SET physical_exits = physical_exits + 1,
                        occupancy = %s,
                        updated_at = %s
                    WHERE location_id = %s AND day = %s
                    """,
                    (occupancy, local_timestamp, self.location_id, day),
                )

            event_id = str(uuid.uuid4())
            self.connection.execute(
                """
                INSERT INTO crossing_events (
                    event_id, location_id, camera_id, event_time, day,
                    direction, identity_status, match_score, unique_increment,
                    repeat_increment, qualified_repeat_increment,
                    occupancy_after, created_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    event_id,
                    self.location_id,
                    camera_id,
                    local_timestamp,
                    day,
                    direction,
                    identity_status,
                    match_score,
                    unique_increment,
                    repeat_increment,
                    qualified_repeat_increment,
                    occupancy,
                    datetime.now(self.timezone),
                ),
            )
            self.connection.execute(
                """
                INSERT INTO occupancy_snapshots (
                    snapshot_id, location_id, event_time, occupancy
                )
                VALUES (%s, %s, %s, %s)
                """,
                (
                    str(uuid.uuid4()),
                    self.location_id,
                    local_timestamp,
                    occupancy,
                ),
            )

        result = {
            "event_id": event_id,
            "camera_id": camera_id,
            "event_time": local_timestamp.isoformat(),
            "direction": direction,
            "identity_status": identity_status,
            "match_score": match_score,
            "unique_increment": unique_increment,
            "repeat_increment": repeat_increment,
            "qualified_repeat_increment": qualified_repeat_increment,
            "occupancy_after": occupancy,
        }
        logger.info("Crossing event: %s", result)
        return result

    def get_counters(self, day: str | None = None) -> dict:
        day = day or datetime.now(self.timezone).date().isoformat()
        with self.lock:
            row = self.connection.execute(
                """
                SELECT *
                FROM counters
                WHERE location_id = %s AND day = %s
                """,
                (self.location_id, day),
            ).fetchone()
        return dict(row) if row else {}

    def purge_old_identities(self) -> int:
        current_day = datetime.now(self.timezone).date()
        with self.lock, self.connection.transaction():
            cursor = self.connection.execute(
                """
                DELETE FROM daily_identities
                WHERE location_id = %s AND day < %s
                """,
                (self.location_id, current_day),
            )
            deleted = cursor.rowcount

        if deleted:
            logger.info("Deleted %s previous-day biometric identities.", deleted)
        return deleted

    def close(self) -> None:
        with self.lock:
            self.connection.close()
