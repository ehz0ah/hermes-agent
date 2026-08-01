"""Tests for Feishu per-channel prompt resolution.

Feishu previously ignored ``channel_prompts`` config (unlike Discord/Slack).
These tests verify that ``_resolve_channel_prompt`` reads the adapter's
``config.extra`` and that the resolved prompt is attached to the dispatched
``MessageEvent`` for the inbound, reaction, and card-action paths.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from gateway.config import PlatformConfig


def _build_adapter(extra=None):
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = FeishuAdapter.__new__(FeishuAdapter)
    adapter.config = PlatformConfig(extra=extra or {})
    adapter._bot_open_id = "ou_bot"
    adapter._bot_user_id = ""
    adapter._bot_name = "Hermes"
    adapter._download_feishu_message_resources = AsyncMock(return_value=([], []))
    adapter._fetch_message_text = AsyncMock(return_value=None)
    adapter.get_chat_info = AsyncMock(return_value={"name": "Test Chat"})
    adapter._resolve_sender_profile = AsyncMock(
        return_value={"user_id": "u1", "user_name": "Alice", "user_id_alt": None}
    )
    adapter._resolve_source_chat_type = Mock(return_value="group")
    adapter.build_source = Mock(return_value=SimpleNamespace(thread_id=None))
    adapter._dispatch_inbound_event = AsyncMock()
    return adapter


def _run_inbound(adapter, chat_id="oc_chat"):
    message = SimpleNamespace(
        content=json.dumps({"text": "plain message"}),
        message_type="text",
        message_id="m",
        mentions=[],
        chat_id=chat_id,
        parent_id=None,
        upper_message_id=None,
        thread_id=None,
    )
    asyncio.run(
        adapter._process_inbound_message(
            data=message, message=message, sender_id=None, chat_type="group", message_id="m",
        )
    )
    return adapter._dispatch_inbound_event.call_args.args[0]


def test_resolve_channel_prompt_exact_match():
    adapter = _build_adapter({"channel_prompts": {"oc_chat": "Be terse."}})
    assert adapter._resolve_channel_prompt("oc_chat") == "Be terse."


def test_resolve_channel_prompt_parent_fallback():
    adapter = _build_adapter({"channel_prompts": {"oc_parent": "Inherit me."}})
    assert adapter._resolve_channel_prompt("oc_thread", "oc_parent") == "Inherit me."


def test_resolve_channel_prompt_no_match_returns_none():
    adapter = _build_adapter({"channel_prompts": {"oc_other": "Nope."}})
    assert adapter._resolve_channel_prompt("oc_chat") is None


def test_resolve_channel_prompt_missing_config_is_safe():
    # __new__ adapter without a config attribute (defensive getattr path).
    from plugins.platforms.feishu.adapter import FeishuAdapter

    bare = FeishuAdapter.__new__(FeishuAdapter)
    assert bare._resolve_channel_prompt("oc_chat") is None


def test_inbound_event_carries_channel_prompt():
    adapter = _build_adapter({"channel_prompts": {"oc_chat": "Feishu role prompt."}})
    event = _run_inbound(adapter, chat_id="oc_chat")
    assert event.channel_prompt.startswith("Feishu role prompt.\n\n")
    assert "preserve the exact verified" in event.channel_prompt


def test_inbound_event_no_prompt_when_unconfigured():
    adapter = _build_adapter({"channel_prompts": {"oc_other": "Different chat."}})
    event = _run_inbound(adapter, chat_id="oc_chat")
    assert event.channel_prompt.startswith("When mentioning a Feishu/Lark user")
    assert "preserve the exact verified" in event.channel_prompt
    assert "first-party lark_* tools" in event.channel_prompt
    assert "never substitute terminal commands" in event.channel_prompt
    assert "Use lark_im reactions sparingly" in event.channel_prompt


def test_observed_context_prompts_prioritize_same_chat():
    from gateway.run import _observed_context_guidance
    from plugins.platforms.feishu.adapter import FeishuAdapter

    channel_prompt = FeishuAdapter._observed_group_channel_prompt()
    turn_prompt = _observed_context_guidance(channel_prompt)

    assert "newest relevant message in this same chat or thread" in channel_prompt
    assert "newest relevant message in this same chat or thread" in turn_prompt
    assert "before consulting other conversations or long-term memory" in turn_prompt


def test_intentional_silence_rows_are_excluded_from_replay_and_cross_session_context():
    from gateway.run import (
        _build_gateway_agent_history,
        _render_cross_session_message,
    )

    silence_row = {"role": "assistant", "content": "NO_REPLY"}
    replay, observed_context = _build_gateway_agent_history(
        [
            {"role": "user", "content": "hello"},
            silence_row,
            {"role": "assistant", "content": "Use NO_REPLY only as a control token."},
        ],
        channel_prompt=None,
    )

    assert observed_context is None
    assert silence_row not in replay
    assert replay[-1]["content"] == "Use NO_REPLY only as a control token."
    assert _render_cross_session_message(silence_row) is None


def test_feishu_observed_context_slides_while_memory_sync_stays_trailing():
    from gateway.run import (
        _build_gateway_agent_history,
        _observed_group_messages_for_memory,
    )

    channel_prompt = "observed Feishu/Lark group context"
    observed_a = {"role": "user", "content": "[Alice]\nA", "observed": True}
    observed_b = {"role": "user", "content": "[Bob]\nB", "observed": True}
    first_history = [observed_a, observed_b]

    first_replay, first_context = _build_gateway_agent_history(
        first_history,
        channel_prompt=channel_prompt,
    )
    first_sync = _observed_group_messages_for_memory(
        first_history,
        channel_prompt=channel_prompt,
    )

    assert first_replay == []
    assert first_context == "[Alice]\nA\n[Bob]\nB"
    assert [message["content"] for message in first_sync] == [
        "[Alice]\nA",
        "[Bob]\nB",
    ]

    observed_c = {"role": "user", "content": "[Carol]\nC", "observed": True}
    second_history = [
        observed_a,
        observed_b,
        {"role": "user", "content": "[Alice]\nWhat did everyone say?"},
        {"role": "assistant", "content": "A and B."},
        observed_c,
    ]

    second_replay, second_context = _build_gateway_agent_history(
        second_history,
        channel_prompt=channel_prompt,
    )
    second_sync = _observed_group_messages_for_memory(
        second_history,
        channel_prompt=channel_prompt,
    )

    assert second_replay == [
        {"role": "user", "content": "[Alice]\nWhat did everyone say?"},
        {"role": "assistant", "content": "A and B."},
    ]
    assert second_context == "[Alice]\nA\n[Bob]\nB\n[Carol]\nC"
    assert [message["content"] for message in second_sync] == ["[Carol]\nC"]


def test_feishu_observed_context_keeps_newest_50_in_chronological_order():
    from gateway.run import _build_gateway_agent_history

    history = [
        {
            "role": "user",
            "content": f"observed-{index:02d}",
            "observed": True,
        }
        for index in range(55)
    ]

    replay, context = _build_gateway_agent_history(
        history,
        channel_prompt="observed Feishu/Lark group context",
    )

    assert replay == []
    assert context is not None
    assert context.splitlines() == [
        f"observed-{index:02d}" for index in range(5, 55)
    ]


def test_feishu_observed_context_enforces_token_cap_for_oversized_newest_message():
    from agent.model_metadata import estimate_tokens_rough
    from gateway.run import _build_gateway_agent_history

    oversized = "界" * 12_000
    history = [
        {"role": "user", "content": "older context", "observed": True},
        {"role": "user", "content": oversized, "observed": True},
    ]

    replay, context = _build_gateway_agent_history(
        history,
        channel_prompt="observed Feishu/Lark group context",
    )

    assert replay == []
    assert context is not None
    assert "older context" not in context
    assert context
    assert len(context) < len(oversized)
    assert estimate_tokens_rough(context) <= 10_000


def test_real_feishu_thread_inherits_bounded_parent_observed_window():
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    parent_history = [
        {"role": "user", "content": "[Alice]\nparent A", "observed": True},
        {"role": "user", "content": "[Bob]\nparent B", "observed": True},
        {"role": "user", "content": "[Alice]\naddressed parent turn"},
        {"role": "assistant", "content": "parent answer"},
        {"role": "user", "content": "[Carol]\nparent C", "observed": True},
    ]

    class _ParentStore:
        _entries = {
            "parent-session-key": SimpleNamespace(session_id="parent-session")
        }

        @staticmethod
        def _ensure_loaded():
            return None

        @staticmethod
        def load_transcript(session_id):
            assert session_id == "parent-session"
            return parent_history

    runner = object.__new__(GatewayRunner)
    runner.session_store = _ParentStore()
    runner._session_key_for_source = lambda source: (
        "parent-session-key" if source.thread_id is None else "thread-session-key"
    )
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_team",
        chat_name="Platform Team",
        chat_type="group",
        user_id="ou_alice",
        user_name="Alice",
        thread_id="omt_thread",
    )

    context = runner._feishu_parent_observed_context_for_thread(
        source,
        channel_prompt="observed Feishu/Lark group context",
    )

    assert context is not None
    assert context.startswith("[Recent parent chat context before this thread]")
    assert context.index("parent A") < context.index("parent B") < context.index("parent C")
    assert "addressed parent turn" not in context
    assert "parent answer" not in context
