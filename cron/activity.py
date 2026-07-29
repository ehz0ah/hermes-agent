"""Process-local interactive activity signal used by background cron jobs."""

from __future__ import annotations

import threading
import time
from typing import Optional

_lock = threading.Lock()
_last_interactive_monotonic: Optional[float] = None


def mark_interactive_activity() -> None:
    """Record that an interactive gateway turn has started."""
    global _last_interactive_monotonic
    with _lock:
        _last_interactive_monotonic = time.monotonic()


def seconds_since_interactive_activity() -> Optional[float]:
    """Return seconds since the latest interactive turn, or None if unknown."""
    with _lock:
        last = _last_interactive_monotonic
    if last is None:
        return None
    return max(0.0, time.monotonic() - last)


def _reset_interactive_activity_for_tests() -> None:
    global _last_interactive_monotonic
    with _lock:
        _last_interactive_monotonic = None
