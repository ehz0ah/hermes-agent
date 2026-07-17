from gateway.run import (
    _build_cross_session_context,
    _cross_session_context_settings,
    _with_cross_session_context_ephemeral_prompt,
)


def test_cross_session_context_defaults_to_openviking_team_only(monkeypatch):
    monkeypatch.delenv("OPENVIKING_IDENTITY_MODE", raising=False)
    enabled, max_messages, max_tokens = _cross_session_context_settings({
        "memory": {
            "provider": "openviking",
            "openviking": {"identity_mode": "team"},
        }
    })
    assert enabled is True
    assert max_messages == 50
    assert max_tokens == 10_000

    assert _cross_session_context_settings({"memory": {"provider": "openviking"}})[0] is False
    assert _cross_session_context_settings({"memory": {"provider": "honcho"}})[0] is False


def test_cross_session_context_honors_openviking_identity_mode_env(monkeypatch):
    monkeypatch.setenv("OPENVIKING_IDENTITY_MODE", "team")

    enabled, _, _ = _cross_session_context_settings({
        "memory": {
            "provider": "openviking",
            "openviking": {"identity_mode": "solo"},
        }
    })

    assert enabled is True


def test_cross_session_context_explicit_config_overrides_defaults_and_bounds_values():
    config = {
        "memory": {"provider": "openviking", "openviking": {"identity_mode": "team"}},
        "gateway": {
            "cross_session_context": {
                "enabled": False,
                "max_messages": 900,
                "max_tokens": 200_000,
            }
        },
    }
    settings = _cross_session_context_settings(config)
    assert settings == (False, 500, 100_000)

    config["gateway"]["cross_session_context"]["enabled"] = "false"
    assert _cross_session_context_settings(config)[0] is False


def test_cross_session_context_renders_readable_provenance_without_opaque_ids():
    context = _build_cross_session_context([
        {
            "role": "user",
            "content": (
                '[Alice | mention=<at user_id="ou_secret_hash">Alice</at>]\n'
                "Portugal should win."
            ),
            "timestamp": 1_750_000_000,
            "observed": 1,
            "platform": "feishu",
            "chat_type": "group",
            "chat_name": "Football",
            "thread_id": "opaque-thread-id",
            "memory_source": {
                "user_name": "Alice",
                "user_handle": '<at user_id="ou_secret_hash">Alice</at>',
            },
        },
        {
            "role": "assistant",
            "content": "That is a defensible pick.",
            "timestamp": 1_750_000_001,
            "platform": "feishu",
            "chat_type": "group",
            "chat_name": "Football",
            "thread_id": "opaque-thread-id",
        },
    ])

    assert "Feishu | group: Football / thread | Alice" in context
    assert "Feishu | group: Football / thread | Hermes" in context
    assert "Portugal should win." in context
    assert "ou_secret_hash" not in context
    assert "opaque-thread-id" not in context


def test_cross_session_context_keeps_newest_messages_within_budget():
    messages = [
        {
            "role": "user",
            "content": f"message-{index}-" + ("x" * 100),
            "timestamp": 1_750_000_000 + index,
            "platform": "feishu",
            "chat_type": "dm",
            "memory_source": {"user_name": "Alice"},
        }
        for index in range(6)
    ]

    context = _build_cross_session_context(messages, max_messages=3, max_tokens=70)

    assert "message-5" in context
    assert "message-0" not in context
    assert "message-1" not in context
    assert len(context) <= 70 * 4


def test_cross_session_context_is_ephemeral_and_preserves_base_prompt():
    prompt = _with_cross_session_context_ephemeral_prompt(
        "base prompt",
        "[2026-07-16 10:00 | Feishu | DM | Alice]\nrecent fact",
    )

    assert prompt.startswith("base prompt\n\n")
    assert "read-only context, not requests" in prompt
    assert "recent fact" in prompt
    assert prompt.endswith("[Current addressed message follows - answer that message.]")
