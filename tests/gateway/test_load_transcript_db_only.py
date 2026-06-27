"""Verify load_transcript returns SQLite messages without any JSONL file."""


from gateway.session import SessionStore
from gateway.config import GatewayConfig


def test_load_transcript_returns_db_messages_when_no_jsonl(tmp_path, monkeypatch):
    """Reading a transcript must work from SQLite alone — no JSONL fallback needed.

    Pin DEFAULT_DB_PATH to tmp_path so this test cannot write to the real
    ~/.hermes/state.db. (DEFAULT_DB_PATH is a module-level constant computed
    at hermes_state import time, before pytest's HERMES_HOME monkeypatch
    fires — the autouse fixture's HERMES_HOME override doesn't help here.)
    """
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

    config = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path, config=config)

    sid = "test-session-db-only"
    store._db.create_session(session_id=sid, source="test")
    store.append_to_transcript(sid, {"role": "user", "content": "hello", "timestamp": 1.0})
    store.append_to_transcript(sid, {"role": "assistant", "content": "world", "timestamp": 2.0})

    history = store.load_transcript(sid)
    assert len(history) == 2
    assert history[0]["content"] == "hello"
    assert history[1]["content"] == "world"


def test_load_transcript_preserves_observed_memory_source(tmp_path, monkeypatch):
    """Observed gateway rows must retain original speaker metadata after reload."""
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

    config = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path, config=config)

    sid = "test-observed-memory-source"
    memory_source = {
        "platform": "feishu",
        "chat_id": "oc_group",
        "chat_type": "group",
        "user_id": "ou_alice",
        "user_name": "Alice",
        "user_handle": '<at user_id="ou_alice">Alice</at>',
        "message_id": "om_1",
    }
    store._db.create_session(session_id=sid, source="feishu:oc_group")
    store.append_to_transcript(
        sid,
        {
            "role": "user",
            "content": "[Alice]\nfavorite snack is oreo",
            "timestamp": 1.0,
            "observed": True,
            "platform_message_id": "om_1",
            "memory_source": memory_source,
        },
    )

    history = store.load_transcript(sid)

    assert len(history) == 1
    assert history[0]["observed"] is True
    assert history[0]["memory_source"] == memory_source
