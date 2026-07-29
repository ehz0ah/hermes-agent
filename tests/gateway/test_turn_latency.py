"""Tests for content-free gateway turn latency instrumentation."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from types import SimpleNamespace

import pytest

from gateway import turn_latency


def _latency_records(caplog):
    records = []
    for record in caplog.records:
        if record.name != "gateway.turn_latency":
            continue
        prefix, payload = record.getMessage().split(" ", 1)
        assert prefix == "turn_latency"
        records.append(json.loads(payload))
    return records


def test_disabled_tracker_emits_nothing(caplog):
    caplog.set_level(logging.INFO, logger="gateway.turn_latency")

    token = turn_latency.start_turn(
        enabled=False,
        platform="feishu",
        chat_type="group",
    )
    turn_latency.record_context_assembly(0.5)
    turn_latency.finish_turn(token, outcome="success")

    assert turn_latency.current_tracker() is None
    assert _latency_records(caplog) == []


def test_enabled_tracker_emits_one_content_free_record(caplog):
    caplog.set_level(logging.INFO, logger="gateway.turn_latency")
    secret_text = "private message text"

    token = turn_latency.start_turn(
        enabled=True,
        platform="feishu",
        chat_type="group",
        participation={
            "debounce_seconds": 0.2,
            "classifier_seconds": 0.3,
            "message": secret_text,
        },
    )
    turn_latency.record_context_assembly(0.01)
    turn_latency.record_openviking_prefetch(0.02)
    model_started_at = turn_latency.mark_model_start()
    turn_latency.mark_model_first_token()
    turn_latency.mark_model_end(model_started_at)
    turn_latency.record_tool(0.04)
    turn_latency.mark_delivery_start()
    turn_latency.mark_delivery_end()
    turn_latency.finish_turn(token, outcome="success")

    records = _latency_records(caplog)
    assert len(records) == 1
    record = records[0]
    assert record["event"] == "gateway_turn_latency"
    assert record["platform"] == "feishu"
    assert record["chat_type"] == "group"
    assert record["outcome"] == "success"
    assert record["participation_debounce_seconds"] == 0.2
    assert record["participation_classifier_seconds"] == 0.3
    assert record["context_assembly_seconds"] == 0.01
    assert record["openviking_prefetch_seconds"] == 0.02
    assert record["tool_count"] == 1
    assert record["tool_total_seconds"] == 0.04
    assert record["total_turn_seconds"] >= 0
    assert secret_text not in caplog.text
    assert turn_latency.current_tracker() is None


def test_nested_trackers_restore_outer_context(caplog):
    caplog.set_level(logging.INFO, logger="gateway.turn_latency")
    outer_token = turn_latency.start_turn(
        enabled=True,
        platform="feishu",
        chat_type="group",
    )
    outer = turn_latency.current_tracker()
    inner_token = turn_latency.start_turn(
        enabled=True,
        platform="feishu",
        chat_type="dm",
    )

    turn_latency.finish_turn(inner_token, outcome="success")
    assert turn_latency.current_tracker() is outer
    turn_latency.finish_turn(outer_token, outcome="success")
    assert turn_latency.current_tracker() is None
    assert len(_latency_records(caplog)) == 2


@pytest.mark.asyncio
async def test_stream_consumer_records_actual_platform_delivery():
    from gateway.stream_consumer import GatewayStreamConsumer

    class _Adapter:
        async def send(self, **_kwargs):
            await asyncio.sleep(0.01)
            return SimpleNamespace(success=True, message_id="message-1")

    token = turn_latency.start_turn(
        enabled=True,
        platform="feishu",
        chat_type="group",
    )
    consumer = GatewayStreamConsumer(_Adapter(), "chat-1")

    result = await consumer._send_message(
        chat_id="chat-1",
        content="final",
    )

    assert result.success is True
    tracker = turn_latency.current_tracker()
    assert tracker is not None
    assert tracker.delivery_seconds >= 0.009
    turn_latency.finish_turn(token, outcome="success")


def test_disabled_hook_overhead_is_below_two_milliseconds_p95():
    samples = []
    for _ in range(2_000):
        started_at = time.perf_counter()
        token = turn_latency.start_turn(
            enabled=False,
            platform="feishu",
            chat_type="group",
        )
        turn_latency.record_context_assembly(0.001)
        turn_latency.record_openviking_prefetch(0.001)
        turn_latency.record_tool(0.001)
        turn_latency.finish_turn(token, outcome="success")
        samples.append(time.perf_counter() - started_at)

    p95 = sorted(samples)[int(len(samples) * 0.95)]
    assert p95 < 0.002


def test_enabled_hook_overhead_is_below_two_milliseconds_p95(monkeypatch):
    monkeypatch.setattr(turn_latency.logger, "info", lambda *args, **kwargs: None)
    samples = []
    for _ in range(2_000):
        started_at = time.perf_counter()
        token = turn_latency.start_turn(
            enabled=True,
            platform="feishu",
            chat_type="group",
            participation={
                "debounce_seconds": 0.001,
                "classifier_seconds": 0.001,
            },
        )
        turn_latency.record_context_assembly(0.001)
        turn_latency.record_openviking_prefetch(0.001)
        model_started_at = turn_latency.mark_model_start()
        turn_latency.mark_model_first_token()
        turn_latency.mark_model_end(model_started_at)
        turn_latency.record_tool(0.001)
        turn_latency.mark_delivery_start()
        turn_latency.mark_delivery_end()
        turn_latency.finish_turn(token, outcome="success")
        samples.append(time.perf_counter() - started_at)

    p95 = sorted(samples)[int(len(samples) * 0.95)]
    assert p95 < 0.002


def test_participation_only_emits_content_free_record(caplog):
    caplog.set_level(logging.INFO, logger="gateway.turn_latency")

    turn_latency.emit_participation_only(
        enabled=True,
        platform="feishu",
        chat_type="group",
        outcome="participation_silent",
        started_at=time.monotonic() - 0.5,
        debounce_seconds=0.25,
        classifier_seconds=0.2,
    )

    records = _latency_records(caplog)
    assert len(records) == 1
    assert records[0]["outcome"] == "participation_silent"
    assert records[0]["participation_debounce_seconds"] == 0.25
    assert records[0]["participation_classifier_seconds"] == 0.2
    assert records[0]["model_total_seconds"] == 0.0
    assert records[0]["delivery_seconds"] == 0.0
