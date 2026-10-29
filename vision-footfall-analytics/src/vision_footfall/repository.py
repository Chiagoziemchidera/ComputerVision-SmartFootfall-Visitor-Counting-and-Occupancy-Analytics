from __future__ import annotations

from datetime import datetime
from typing import Protocol

import numpy as np


class FootfallRepository(Protocol):
    def process_crossing(
        self,
        camera_id: str,
        direction: str,
        timestamp: datetime,
        embedding: np.ndarray | None,
    ) -> dict: ...

    def get_counters(self, day: str | None = None) -> dict: ...

    def purge_old_identities(self) -> int: ...

    def close(self) -> None: ...
