"""
CooledAI API Rate Limiter — sliding-window counter per API key.

No external dependencies (no Redis).  Designed for single-process deployment
behind Railway / Uvicorn.
"""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock
from typing import Tuple


class SlidingWindowRateLimiter:
    """In-process sliding-window rate limiter.

    Parameters
    ----------
    max_requests : int
        Maximum allowed requests within *window_seconds*.
    window_seconds : int
        Length of the sliding window.
    """

    def __init__(
        self, max_requests: int = 60, window_seconds: int = 60
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def check(self, key: str) -> Tuple[bool, int]:
        """Check whether *key* is within its rate limit.

        Returns
        -------
        (allowed, retry_after)
            *allowed* is ``True`` when the request may proceed.
            *retry_after* is the number of seconds the caller should wait
            before retrying (0 when allowed).
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            timestamps = self._buckets[key]
            # Purge expired entries
            timestamps[:] = [ts for ts in timestamps if ts > cutoff]

            if len(timestamps) >= self.max_requests:
                # Earliest entry that will expire
                retry_after = int(timestamps[0] - cutoff) + 1
                return False, max(retry_after, 1)

            timestamps.append(now)
            return True, 0

    def remaining(self, key: str) -> int:
        """Return remaining requests in current window for *key*."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            timestamps = self._buckets.get(key, [])
            active = sum(1 for ts in timestamps if ts > cutoff)
            return max(0, self.max_requests - active)
