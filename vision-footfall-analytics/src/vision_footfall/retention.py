from __future__ import annotations

import threading
from datetime import datetime
from zoneinfo import ZoneInfo

from .repository import FootfallRepository


class RetentionWorker(threading.Thread):

    def __init__(
        self,
        database: FootfallRepository,
        stop_event: threading.Event,
        timezone_name: str,
    ):
        super().__init__(name="retention-worker", daemon=True)
        self.database = database
        self.stop_event = stop_event
        self.timezone = ZoneInfo(timezone_name)

    def run(self):
        last_seen_day = datetime.now(self.timezone).date()
        while not self.stop_event.wait(30):
            current_day = datetime.now(self.timezone).date()
            if current_day != last_seen_day:
                self.database.purge_old_identities()
                last_seen_day = current_day
