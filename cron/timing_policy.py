"""Scheduling policy helpers for exact and low-contention background jobs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as clock_time, timedelta
import hashlib
from typing import Any, Mapping, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from cron.activity import seconds_since_interactive_activity

EXACT = "exact"
BACKGROUND = "background"
VALID_TIMING_POLICIES = frozenset({EXACT, BACKGROUND})


def normalize_timing_policy(value: Any) -> str:
    policy = str(value or EXACT).strip().lower()
    if policy not in VALID_TIMING_POLICIES:
        allowed = ", ".join(sorted(VALID_TIMING_POLICIES))
        raise ValueError(f"timing_policy must be one of: {allowed}")
    return policy


@dataclass(frozen=True)
class BackgroundWindow:
    timezone: ZoneInfo
    start: clock_time
    end: clock_time
    active_grace_seconds: int
    max_deferral_seconds: int
    jitter_seconds: int


def _parse_clock(value: Any, default: str) -> clock_time:
    text = str(value or default).strip()
    try:
        return datetime.strptime(text, "%H:%M").time()
    except ValueError as exc:
        raise ValueError(f"background window time must use HH:MM (got {text!r})") from exc


def resolve_background_window(config: Optional[Mapping[str, Any]] = None) -> BackgroundWindow:
    cfg = config if isinstance(config, Mapping) else {}
    cron_cfg = cfg.get("cron") if isinstance(cfg.get("cron"), Mapping) else {}
    raw = (
        cron_cfg.get("background_window")
        if isinstance(cron_cfg.get("background_window"), Mapping)
        else {}
    )
    timezone_name = str(raw.get("timezone") or "Asia/Singapore").strip()
    try:
        timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"unknown cron background timezone: {timezone_name}") from exc
    return BackgroundWindow(
        timezone=timezone,
        start=_parse_clock(raw.get("start"), "20:00"),
        end=_parse_clock(raw.get("end"), "08:00"),
        active_grace_seconds=max(0, int(raw.get("active_grace_seconds", 300))),
        max_deferral_seconds=max(0, int(raw.get("max_deferral_seconds", 43200))),
        jitter_seconds=max(0, int(raw.get("jitter_seconds", 300))),
    )


def _inside_window(local_now: datetime, window: BackgroundWindow) -> bool:
    current = local_now.time().replace(tzinfo=None)
    if window.start == window.end:
        return True
    if window.start < window.end:
        return window.start <= current < window.end
    return current >= window.start or current < window.end


def _next_window_start(local_now: datetime, window: BackgroundWindow) -> datetime:
    candidate = datetime.combine(local_now.date(), window.start, window.timezone)
    if candidate <= local_now:
        candidate += timedelta(days=1)
    return candidate


def _stable_jitter(job_id: str, limit: int) -> int:
    if limit <= 0:
        return 0
    digest = hashlib.sha256(str(job_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (limit + 1)


def background_defer_until(
    job: Mapping[str, Any],
    now: datetime,
    *,
    config: Optional[Mapping[str, Any]] = None,
    activity_age_seconds: Optional[float] = None,
) -> Optional[datetime]:
    """Return a later dispatch instant, or None when the job may run now.

    The first deferral establishes a durable deadline. Once that deadline is
    reached, the job runs even if the preferred window or activity grace would
    otherwise defer it again.
    """
    if normalize_timing_policy(job.get("timing_policy")) != BACKGROUND:
        return None

    window = resolve_background_window(config)
    local_now = now.astimezone(window.timezone)
    deferred_since_raw = job.get("background_deferred_since")
    try:
        deferred_since = (
            datetime.fromisoformat(str(deferred_since_raw)).astimezone(window.timezone)
            if deferred_since_raw
            else local_now
        )
    except (TypeError, ValueError):
        deferred_since = local_now

    if window.max_deferral_seconds <= 0:
        return None
    deadline = deferred_since + timedelta(seconds=window.max_deferral_seconds)
    if local_now >= deadline:
        return None

    if not _inside_window(local_now, window):
        candidate = _next_window_start(local_now, window)
        candidate += timedelta(
            seconds=_stable_jitter(str(job.get("id") or ""), window.jitter_seconds)
        )
        return min(candidate, deadline).astimezone(now.tzinfo)

    age = (
        seconds_since_interactive_activity()
        if activity_age_seconds is None
        else activity_age_seconds
    )
    if age is not None and age < window.active_grace_seconds:
        wait_seconds = window.active_grace_seconds - max(0.0, age)
        candidate = local_now + timedelta(seconds=wait_seconds)
        return min(candidate, deadline).astimezone(now.tzinfo)
    return None
