"""Desktop/TUI turn-dispatch observability (#86647).

During the #79278/#86647 persistent-mute investigation the decisive evidence
was an *absence*: a Desktop request left no INFO record in ``agent.log`` or
``gateway.log`` at all (``0 platform=desktop`` across the whole file), so a
muted window was structurally indistinguishable from a request that never
arrived. This suite pins the two-record contract that fixes that:

* ``_run_prompt_submit`` logs one ``tui prompt accepted`` INFO record before
  the turn thread starts, carrying the UI session id, the gateway
  ``session_key``, and the agent's live ``session_id`` (rotated independently
  by compression — the triple is what a rotation-mute trace needs).
* The turn's ``finally`` logs exactly one ``tui turn finished`` bookend on
  every path (success, returned error, exception), re-reading
  ``agent.session_id`` so a mid-turn compression rotation shows up as an
  accepted/finished pair with different agent ids.
* No prompt content is ever logged.
"""

from __future__ import annotations

import logging
import threading
import types

import pytest

from tui_gateway import server

_REAL_THREAD = threading.Thread
_REAL_EMIT = server._emit


class _InlineThread:
    """Run the turn synchronously so tests observe its final state."""

    def __init__(self, target=None, daemon=None, args=(), kwargs=None, name=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None

def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "gw-session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "inflight_turn": None,
        **extra,
    }

@pytest.fixture()
def turn_env(monkeypatch, tmp_path):
    """Neutralize the turn pipeline's environment-heavy side paths."""
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})

def _records(caplog, needle):
    return [r for r in caplog.records if needle in r.getMessage()]

SECRETISH_PROMPT = "please rotate QDRANT_API_KEY=hunter2-super-secret now"

def test_accepted_and_finished_records_on_success(turn_env, caplog):
    agent = types.SimpleNamespace(
        session_id="agent-sid-1",
        run_conversation=lambda *a, **k: {"final_response": "done"},
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)

    with caplog.at_level(logging.INFO, logger="tui_gateway.server"):
        server._run_prompt_submit("rid", "ui-sid", session, SECRETISH_PROMPT)

    accepted = _records(caplog, "tui prompt accepted")
    finished = _records(caplog, "tui turn finished")
    assert len(accepted) == 1
    assert len(finished) == 1

    msg = accepted[0].getMessage()
    # Prompt content is never logged — only its length.
    assert "hunter2" not in msg
    assert "QDRANT_API_KEY" not in msg

    fin = finished[0].getMessage()
    assert "hunter2" not in fin


def test_blocked_memory_trim_does_not_hold_turn_completion(turn_env, monkeypatch, caplog):
    """Best-effort allocator cleanup must not keep a completed turn busy (#131740)."""
    from hermes_cli import mem_trim

    trim_started = threading.Event()
    release_trim = threading.Event()
    retired = []

    def blocking_trim(**_kwargs):
        trim_started.set()
        assert release_trim.wait(timeout=5)

    monkeypatch.setattr(server.threading, "Thread", _REAL_THREAD)
    monkeypatch.setattr(server, "_sessions_quiescent", lambda exclude=None: True)
    monkeypatch.setattr(mem_trim, "trim_memory", blocking_trim)
    monkeypatch.setattr(server, "_record_turn_marker", lambda *a, **k: "turn-marker")
    monkeypatch.setattr(server, "_retire_turn_marker", lambda _session, key: retired.append(key))
    monkeypatch.setattr(server, "_emit_settled_session_info", lambda *a, **k: None)

    agent = types.SimpleNamespace(
        session_id="agent-sid-1",
        run_conversation=lambda *a, **k: {"final_response": "done"},
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True)

    try:
        with caplog.at_level(logging.INFO, logger="tui_gateway.server"):
            assert server._run_prompt_submit("rid", "ui-sid", session, "finish this")
            assert trim_started.wait(timeout=2), "turn never reached post-turn memory trim"

            assert session["running"] is False
            assert "turn-marker" in retired
            assert len(_records(caplog, "tui turn finished")) == 1
    finally:
        release_trim.set()
        if thread := session.get("_run_thread"):
            thread.join(timeout=5)


def test_blocked_usage_emit_does_not_hold_turn_completion(turn_env, monkeypatch, caplog):
    """A stalled live-usage update must not keep a completed turn busy (#131740)."""
    usage_emit_started = threading.Event()
    release_usage_emit = threading.Event()
    conversation_returned = threading.Event()
    session_settled = threading.Event()
    arrivals = []
    retired = []
    usage_samples = iter(({"total": 0}, {"total": 1}))

    def moving_usage(_agent):
        return next(usage_samples, {"total": 1})

    class BlockingTransport:
        def write(self, frame):
            event = (frame.get("params") or {}).get("type")
            if event == "session.usage":
                usage_emit_started.set()
                assert release_usage_emit.wait(timeout=5)
            arrivals.append((event, (frame.get("params") or {}).get("seq")))
            return True

        def close(self):
            return None

    def run_conversation(*_args, **_kwargs):
        assert usage_emit_started.wait(timeout=2), "usage ticker never entered its emit"
        conversation_returned.set()
        return {"final_response": "done"}

    real_ticker = server._start_usage_ticker
    monkeypatch.setattr(server.threading, "Thread", _REAL_THREAD)
    monkeypatch.setattr(server, "_emit", _REAL_EMIT)
    monkeypatch.setattr(server, "_get_usage", moving_usage)
    monkeypatch.setattr(server, "_USAGE_TICKER_JOIN_TIMEOUT_S", 0.05)
    monkeypatch.setattr(
        server,
        "_start_usage_ticker",
        lambda sid, agent: real_ticker(sid, agent, interval=0.01),
    )
    monkeypatch.setattr(server, "_record_turn_marker", lambda *a, **k: "turn-marker")
    monkeypatch.setattr(server, "_retire_turn_marker", lambda _session, key: retired.append(key))
    monkeypatch.setattr(server, "_emit_settled_session_info", lambda *a, **k: session_settled.set())

    agent = types.SimpleNamespace(
        session_id="agent-sid-1",
        run_conversation=run_conversation,
        clear_interrupt=lambda: None,
    )
    session = _session(agent=agent, running=True, transport=BlockingTransport())
    server._sessions["ui-sid"] = session

    try:
        with caplog.at_level(logging.INFO, logger="tui_gateway.server"):
            assert server._run_prompt_submit("rid", "ui-sid", session, "finish this")
            assert conversation_returned.wait(timeout=2), "agent did not finish"
            assert session_settled.wait(timeout=2), "turn did not pass the bounded ticker join"

            assert session["running"] is False
            assert "turn-marker" in retired
            assert len(_records(caplog, "tui turn finished")) == 1
            complete = next(item for item in arrivals if item[0] == "message.complete")
    finally:
        release_usage_emit.set()
        if thread := session.get("_run_thread"):
            thread.join(timeout=5)
        server._sessions.pop("ui-sid", None)

    usage = next(item for item in arrivals if item[0] == "session.usage")
    assert arrivals.index(complete) < arrivals.index(usage)
    assert usage[1] < complete[1]
