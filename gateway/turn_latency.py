"""Low-overhead, content-free gateway turn latency instrumentation."""

from __future__ import annotations

import json
import logging
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Mapping

logger = logging.getLogger(__name__)


@dataclass
class TurnLatencyTracker:
    """Accumulates timing data for one handled gateway turn."""

    started_at: float
    platform: str
    chat_type: str
    participation_debounce_seconds: float = 0.0
    participation_classifier_seconds: float = 0.0
    context_assembly_seconds: float = 0.0
    openviking_prefetch_seconds: float = 0.0
    model_started_at: float | None = None
    model_first_token_at: float | None = None
    model_total_seconds: float = 0.0
    tool_count: int = 0
    tool_total_seconds: float = 0.0
    delivery_started_at: float | None = None
    delivery_seconds: float = 0.0

    def to_record(self, *, outcome: str, finished_at: float) -> dict[str, Any]:
        first_token_seconds = 0.0
        if (
            self.model_started_at is not None
            and self.model_first_token_at is not None
        ):
            first_token_seconds = max(
                0.0,
                self.model_first_token_at - self.model_started_at,
            )
        return {
            "event": "gateway_turn_latency",
            "platform": self.platform,
            "chat_type": self.chat_type,
            "outcome": outcome,
            "participation_debounce_seconds": self.participation_debounce_seconds,
            "participation_classifier_seconds": self.participation_classifier_seconds,
            "context_assembly_seconds": self.context_assembly_seconds,
            "openviking_prefetch_seconds": self.openviking_prefetch_seconds,
            "model_time_to_first_token_seconds": first_token_seconds,
            "model_total_seconds": self.model_total_seconds,
            "tool_count": self.tool_count,
            "tool_total_seconds": self.tool_total_seconds,
            "delivery_seconds": self.delivery_seconds,
            "total_turn_seconds": max(0.0, finished_at - self.started_at),
        }


_CURRENT_TRACKER: ContextVar[TurnLatencyTracker | None] = ContextVar(
    "gateway_turn_latency_tracker",
    default=None,
)


def _nonnegative_float(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def start_turn(
    *,
    enabled: bool,
    platform: str,
    chat_type: str,
    participation: Mapping[str, Any] | None = None,
) -> Token[TurnLatencyTracker | None] | None:
    """Install a tracker for the current async turn when enabled."""
    if not enabled:
        return None
    participation = participation or {}
    tracker = TurnLatencyTracker(
        started_at=time.monotonic(),
        platform=str(platform or "unknown"),
        chat_type=str(chat_type or "unknown"),
        participation_debounce_seconds=_nonnegative_float(
            participation.get("debounce_seconds")
        ),
        participation_classifier_seconds=_nonnegative_float(
            participation.get("classifier_seconds")
        ),
    )
    return _CURRENT_TRACKER.set(tracker)


def current_tracker() -> TurnLatencyTracker | None:
    return _CURRENT_TRACKER.get()


def record_context_assembly(seconds: float) -> None:
    tracker = current_tracker()
    if tracker is not None:
        tracker.context_assembly_seconds += _nonnegative_float(seconds)


def record_openviking_prefetch(seconds: float) -> None:
    tracker = current_tracker()
    if tracker is not None:
        tracker.openviking_prefetch_seconds += _nonnegative_float(seconds)


def mark_model_start() -> float | None:
    tracker = current_tracker()
    if tracker is None:
        return None
    started_at = time.monotonic()
    if tracker.model_started_at is None:
        tracker.model_started_at = started_at
    return started_at


def mark_model_first_token() -> None:
    tracker = current_tracker()
    if tracker is not None and tracker.model_first_token_at is None:
        tracker.model_first_token_at = time.monotonic()


def mark_model_end(started_at: float | None) -> None:
    tracker = current_tracker()
    if tracker is None or started_at is None:
        return
    finished_at = time.monotonic()
    tracker.model_total_seconds += max(0.0, finished_at - started_at)
    if tracker.model_first_token_at is None:
        tracker.model_first_token_at = finished_at


def record_tool(seconds: float) -> None:
    tracker = current_tracker()
    if tracker is not None:
        tracker.tool_count += 1
        tracker.tool_total_seconds += _nonnegative_float(seconds)


def record_delivery(seconds: float) -> None:
    """Add one completed platform I/O operation to the current turn."""
    tracker = current_tracker()
    if tracker is not None:
        tracker.delivery_seconds += _nonnegative_float(seconds)


def mark_delivery_start() -> None:
    tracker = current_tracker()
    if tracker is not None and tracker.delivery_started_at is None:
        tracker.delivery_started_at = time.monotonic()


def mark_delivery_end() -> None:
    tracker = current_tracker()
    if tracker is None or tracker.delivery_started_at is None:
        return
    tracker.delivery_seconds += max(
        0.0,
        time.monotonic() - tracker.delivery_started_at,
    )
    tracker.delivery_started_at = None


def finish_turn(
    token: Token[TurnLatencyTracker | None] | None,
    *,
    outcome: str,
) -> None:
    """Emit exactly one structured latency record and clear the tracker."""
    if token is None:
        return
    tracker = current_tracker()
    try:
        if tracker is not None:
            mark_delivery_end()
            record = tracker.to_record(
                outcome=str(outcome or "unknown"),
                finished_at=time.monotonic(),
            )
            logger.info(
                "turn_latency %s",
                json.dumps(record, sort_keys=True, separators=(",", ":")),
            )
    except Exception:
        logger.debug("turn latency emission failed", exc_info=True)
    finally:
        _CURRENT_TRACKER.reset(token)


def emit_participation_only(
    *,
    enabled: bool,
    platform: str,
    chat_type: str,
    outcome: str,
    started_at: float,
    debounce_seconds: float,
    classifier_seconds: float,
) -> None:
    """Emit one record when adaptive participation does not dispatch a turn."""
    if not enabled:
        return
    try:
        finished_at = time.monotonic()
        tracker = TurnLatencyTracker(
            started_at=min(_nonnegative_float(started_at), finished_at),
            platform=str(platform or "unknown"),
            chat_type=str(chat_type or "unknown"),
            participation_debounce_seconds=_nonnegative_float(
                debounce_seconds
            ),
            participation_classifier_seconds=_nonnegative_float(
                classifier_seconds
            ),
        )
        logger.info(
            "turn_latency %s",
            json.dumps(
                tracker.to_record(
                    outcome=str(outcome or "unknown"),
                    finished_at=finished_at,
                ),
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    except Exception:
        logger.debug(
            "participation latency emission failed",
            exc_info=True,
        )
