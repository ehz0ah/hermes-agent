"""Focused tests for Feishu adaptive group participation."""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource
from plugins.platforms.feishu.adapter import (
    FeishuAdapter,
    FeishuGroupRule,
    _FeishuParticipationCandidate,
    _FeishuParticipationState,
    _apply_yaml_config,
    _build_adapter,
    register,
)
from tests.gateway.feishu_helpers import (
    make_adapter_skeleton,
    make_message,
    make_sender,
    stub_mention,
)


def _group_message(
    text: str,
    *,
    message_id: str = "om_current",
    chat_id: str = "oc_team",
) -> SimpleNamespace:
    message = make_message(
        message_id=message_id,
        chat_type="group",
        chat_id=chat_id,
    )
    message.content = json.dumps({"text": text})
    message.message_type = "text"
    message.thread_id = None
    message.parent_id = None
    message.upper_message_id = None
    message.root_id = None
    return message


def _candidate(
    *,
    message_id: str = "om_current",
    generation: int = 1,
    key: str = "oc_team:thread:-",
    text: str = "Should we move the release?",
) -> _FeishuParticipationCandidate:
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_team",
        chat_name="Platform Team",
        chat_type="group",
        user_id="ou_alice",
        user_id_alt="on_alice",
        user_name="Alice",
        user_handle='<at user_id="ou_alice">Alice</at>',
        message_id=message_id,
    )
    return _FeishuParticipationCandidate(
        data=SimpleNamespace(),
        message=_group_message(text, message_id=message_id),
        sender_id=SimpleNamespace(
            open_id="ou_alice",
            user_id=None,
            union_id="on_alice",
        ),
        message_id=message_id,
        text=text,
        source=source,
        session_source=FeishuAdapter._shared_group_session_source(source),
        key=key,
        generation=generation,
        queued_at=time.monotonic(),
    )


def _adaptive_adapter() -> FeishuAdapter:
    adapter = make_adapter_skeleton(
        require_mention=True,
        group_policy="open",
    )
    adapter._participation_mode = "adaptive"
    adapter._participation_debounce_seconds = 0
    adapter._participation_timeout_seconds = 4
    adapter._participation_recent_messages = 12
    adapter._participation_confidence_threshold = 0.8
    adapter._participation_hot_confidence_threshold = 0.55
    adapter._participation_hot_window_seconds = 180
    adapter._participation_hot_max_human_messages = 2
    adapter._participation_cooldown_seconds = 30
    adapter._sent_message_ids_to_chat = {}
    adapter._ensure_participation_state()
    return adapter


class _TranscriptStore:
    def __init__(self, rows=None):
        self.rows = list(rows or [])
        self.appended = []

    def get_or_create_session(self, _source):
        return SimpleNamespace(session_id="session-team")

    def load_transcript(self, _session_id):
        return list(self.rows)

    def append_to_transcript(self, session_id, message, skip_db=False):
        self.appended.append((session_id, message, skip_db))


class TestParticipationSettings:
    @patch.dict(os.environ, {}, clear=True)
    def test_legacy_require_mention_maps_to_modes(self):
        assert FeishuAdapter._load_settings(
            {"require_mention": True}
        ).participation_mode == "mention_only"
        assert FeishuAdapter._load_settings(
            {"require_mention": False}
        ).participation_mode == "always"

    @patch.dict(os.environ, {}, clear=True)
    def test_invalid_numeric_settings_fall_back_safely(self):
        settings = FeishuAdapter._load_settings(
            {
                "participation": {
                    "mode": "adaptive",
                    "debounce_seconds": "not-a-number",
                    "timeout_seconds": "not-a-number",
                    "recent_messages": "invalid",
                    "confidence_threshold": float("nan"),
                    "hot_confidence_threshold": float("inf"),
                    "hot_window_seconds": -1,
                    "hot_max_human_messages": -1,
                    "cooldown_seconds": -1,
                }
            }
        )

        assert settings.participation_mode == "adaptive"
        assert settings.participation_debounce_seconds == 1.5
        assert settings.participation_timeout_seconds == 30.0
        assert settings.participation_recent_messages == 12
        assert settings.participation_confidence_threshold == 0.8
        assert settings.participation_hot_confidence_threshold == 0.55
        assert settings.participation_hot_window_seconds == 180
        assert settings.participation_hot_max_human_messages == 2
        assert settings.participation_cooldown_seconds == 30

    @patch.dict(os.environ, {}, clear=True)
    def test_timeout_override_is_preserved(self):
        settings = FeishuAdapter._load_settings(
            {"participation": {"mode": "adaptive", "timeout_seconds": 2.5}}
        )

        assert settings.participation_timeout_seconds == 2.5

    @patch.dict(os.environ, {}, clear=True)
    def test_plain_text_aliases_are_configurable_with_safe_default(self):
        default = FeishuAdapter._load_settings({"participation": {}})
        customized = FeishuAdapter._load_settings(
            {"participation": {"aliases": ["Hermes", "小海", "Hermes"]}}
        )

        assert default.participation_aliases == ("Hermes",)
        assert customized.participation_aliases == ("Hermes", "小海")

    def test_per_group_mode_precedes_global_and_legacy_policy(self):
        adapter = _adaptive_adapter()
        adapter._group_rules = {
            "oc_always": FeishuGroupRule(
                policy="open",
                participation_mode="always",
            ),
            "oc_legacy": FeishuGroupRule(
                policy="open",
                require_mention=True,
            ),
        }

        assert adapter._participation_mode_for("oc_team") == "adaptive"
        assert adapter._participation_mode_for("oc_always") == "always"
        assert adapter._participation_mode_for("oc_legacy") == "mention_only"


class TestParticipationAdmission:
    def test_mention_only_accepts_mentions_commands_and_replies_to_hermes(self):
        adapter = make_adapter_skeleton(
            require_mention=True,
            group_policy="open",
        )
        adapter._participation_mode = "mention_only"
        adapter._sent_message_ids_to_chat = {"om_hermes": "oc_team"}
        sender = make_sender()

        plain = _group_message("hello")
        stub_mention(adapter, False)
        assert adapter._admit(sender, plain) == "group_policy_rejected"

        command = _group_message("/status")
        assert adapter._admit(sender, command) is None

        reply = _group_message("following up")
        reply.parent_id = "om_hermes"
        assert adapter._admit(sender, reply) is None

        mentioned = _group_message("Hermes?")
        stub_mention(adapter, True)
        assert adapter._admit(sender, mentioned) is None

        plain_alias = _group_message("welcome back hermes")
        stub_mention(adapter, False)
        assert adapter._admit(sender, plain_alias) is None

        alias_prefix_only = _group_message("welcome back hermesbot")
        assert adapter._admit(sender, alias_prefix_only) == "group_policy_rejected"

        adapter._bot_name = "TeamMate"
        hydrated_name = _group_message("welcome back teammate")
        assert adapter._admit(sender, hydrated_name) is None

    def test_always_accepts_plain_group_text(self):
        adapter = make_adapter_skeleton(
            require_mention=True,
            group_policy="open",
        )
        adapter._participation_mode = "always"
        adapter._sent_message_ids_to_chat = {}
        stub_mention(adapter, False)

        assert adapter._admit(make_sender(), _group_message("hello")) is None

    def test_adaptive_routes_only_unaddressed_human_text_to_classifier(self):
        adapter = _adaptive_adapter()
        sender = make_sender()
        message = _group_message("Could someone review this?")
        stub_mention(adapter, False)

        reason = adapter._admit(sender, message)

        assert reason == "group_policy_rejected"
        assert adapter._should_classify_unaddressed_group_message(
            sender=sender,
            message=message,
            reason=reason,
        )
        assert not adapter._should_classify_unaddressed_group_message(
            sender=make_sender(sender_type="bot"),
            message=message,
            reason=reason,
        )


class TestParticipationDecisionParsing:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (
                '{"decision":"candidate","confidence":0.91}',
                ("candidate", 0.91),
            ),
            (
                "```json\n"
                '{"decision":"silent","confidence":1}'
                "\n```",
                ("silent", 1.0),
            ),
            (
                '{"decision":"uncertain","confidence":0.52}',
                ("uncertain", 0.52),
            ),
        ],
    )
    def test_accepts_only_strict_json_contract(self, raw, expected):
        assert FeishuAdapter._parse_participation_decision(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "speak",
            "[]",
            '{"decision":"maybe","confidence":0.9}',
            '{"decision":"wake","confidence":0.9}',
            '{"decision":"candidate","confidence":true}',
            '{"decision":"candidate","confidence":1.1}',
            '{"decision":"candidate","confidence":0.9,"reason_code":"extra"}',
        ],
    )
    def test_rejects_malformed_or_ambiguous_output(self, raw):
        assert FeishuAdapter._parse_participation_decision(raw) is None

    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {
            "AUXILIARY_FEISHU_PARTICIPATION_PROVIDER": "custom",
            "AUXILIARY_FEISHU_PARTICIPATION_MODEL": "deepseek-v4-flash",
            "AUXILIARY_FEISHU_PARTICIPATION_BASE_URL": (
                "https://example.invalid/v1"
            ),
            "AUXILIARY_FEISHU_PARTICIPATION_API_KEY": "secret-test-value",
        },
        clear=False,
    )
    async def test_classifier_uses_bounded_same_chat_context(self):
        adapter = _adaptive_adapter()
        adapter._participation_recent_messages = 2
        adapter._session_store = _TranscriptStore(
            [
                {"role": "user", "content": "old", "timestamp": "1"},
                {"role": "tool", "content": "ignore", "timestamp": "2"},
                {"role": "assistant", "content": "recent answer", "timestamp": "3"},
                {"role": "user", "content": "recent question", "timestamp": "4"},
            ]
        )
        response = object()

        with (
            patch(
                "agent.auxiliary_client.async_call_llm",
                new=AsyncMock(return_value=response),
            ) as call,
            patch(
                "agent.auxiliary_client.extract_content_or_reasoning",
                return_value='{"decision":"candidate","confidence":0.9}',
            ),
        ):
            assert await adapter._classify_participation(_candidate())

        messages = call.await_args.kwargs["messages"]
        classifier_prompt = messages[0]["content"]
        assert "cross-conversation context" in classifier_prompt
        assert "long-term memory" in classifier_prompt
        assert "do not require the answer to appear" in classifier_prompt
        assert "colleague preferences" in classifier_prompt
        assert "do not dismiss it merely because it is personal" in classifier_prompt
        assert '"candidate", "uncertain", or "silent"' in classifier_prompt
        assert "social inclusion" in classifier_prompt
        assert "group, team, everyone, or the chat" in classifier_prompt
        assert "written reply is unnecessary" in classifier_prompt
        assert "choose uncertain rather than silent" in classifier_prompt
        assert "routine logistical information" in classifier_prompt
        assert "expressive, affiliative message that recognizes an individual" in classifier_prompt
        assert "instrumental requests, questions, and assigned work" in classifier_prompt
        assert "conversation_state.hot is true" in classifier_prompt
        assert "semantically answers or continues Hermes's recent turn" in classifier_prompt
        assert "directed at another person" in classifier_prompt
        assert "never wake merely to announce" in classifier_prompt
        assert "high confidence" in classifier_prompt
        assert "which option Alice preferred" in classifier_prompt
        assert "structured hot-state evidence" in classifier_prompt
        assert call.await_args.kwargs["provider"] == "custom"
        assert call.await_args.kwargs["model"] == "deepseek-v4-flash"
        assert (
            call.await_args.kwargs["base_url"]
            == "https://example.invalid/v1"
        )
        assert call.await_args.kwargs["api_key"] == "secret-test-value"
        assert call.await_args.kwargs["timeout"] == 4
        payload = json.loads(messages[1]["content"])
        assert payload["chat"]["chat_name"] == "Platform Team"
        assert payload["sender"]["name"] == "Alice"
        assert payload["assistant_identity"]["name"] == "Hermes"
        assert "Hermes" in payload["assistant_identity"]["aliases"]
        assert payload["recent_dialogue"] == [
            {
                "role": "assistant",
                "content": "recent answer",
                "timestamp": "3",
            },
            {
                "role": "user",
                "content": "recent question",
                "timestamp": "4",
            },
        ]
        assert payload["conversation_state"]["same_chat_and_thread"] is True
        assert payload["conversation_state"]["reply_to_hermes"] is False

    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {
            "BYTEPLUS_FAST_LLM_MODEL": "deepseek-v4-flash",
            "BYTEPLUS_API_BASE": "https://example.invalid/v1",
            "BYTEPLUS_API_KEY": "secret-test-value",
        },
        clear=True,
    )
    async def test_classifier_bridges_byteplus_without_feishu_yaml(self):
        adapter = _adaptive_adapter()
        response = object()

        with (
            patch(
                "agent.auxiliary_client.async_call_llm",
                new=AsyncMock(return_value=response),
            ) as call,
            patch(
                "agent.auxiliary_client.extract_content_or_reasoning",
                return_value='{"decision":"silent","confidence":0.95}',
            ),
        ):
            assert not await adapter._classify_participation(_candidate())

        assert call.await_args.kwargs["provider"] == "custom"
        assert call.await_args.kwargs["model"] == "deepseek-v4-flash"
        assert (
            call.await_args.kwargs["base_url"]
            == "https://example.invalid/v1"
        )
        assert call.await_args.kwargs["api_key"] == "secret-test-value"

    @pytest.mark.asyncio
    async def test_low_confidence_and_malformed_outputs_fail_silent(self):
        adapter = _adaptive_adapter()
        adapter._session_store = _TranscriptStore()
        response = object()

        with (
            patch(
                "agent.auxiliary_client.async_call_llm",
                new=AsyncMock(return_value=response),
            ),
            patch(
                "agent.auxiliary_client.extract_content_or_reasoning",
                return_value='{"decision":"candidate","confidence":0.79}',
            ),
        ):
            assert not await adapter._classify_participation(_candidate())

        with (
            patch(
                "agent.auxiliary_client.async_call_llm",
                new=AsyncMock(return_value=response),
            ),
            patch(
                "agent.auxiliary_client.extract_content_or_reasoning",
                return_value="not-json",
            ),
        ):
            assert not await adapter._classify_participation(_candidate())

    @pytest.mark.asyncio
    async def test_classifier_enforces_one_end_to_end_deadline(self):
        adapter = _adaptive_adapter()
        adapter._participation_timeout_seconds = 0.01
        adapter._session_store = _TranscriptStore()

        async def never_finishes(**_kwargs):
            await asyncio.sleep(10)

        with patch(
            "agent.auxiliary_client.async_call_llm",
            side_effect=never_finishes,
        ):
            started_at = time.monotonic()
            with pytest.raises(TimeoutError):
                await adapter._classify_participation(_candidate())

        assert time.monotonic() - started_at < 0.5


class TestParticipationLifecycle:
    @pytest.mark.asyncio
    async def test_speak_dispatches_once_without_observed_duplicate(self):
        adapter = _adaptive_adapter()
        candidate = _candidate()
        adapter._participation_generations[candidate.key] = candidate.generation
        adapter._participation_pending[candidate.key] = candidate
        adapter._classify_participation = AsyncMock(return_value=True)
        adapter._process_inbound_message = AsyncMock()
        adapter._persist_observed_candidate = AsyncMock()

        await adapter._run_participation_candidate(candidate)

        adapter._process_inbound_message.assert_awaited_once()
        assert (
            adapter._process_inbound_message.await_args.kwargs[
                "dispatch_immediately"
            ]
            is True
        )
        assert (
            adapter._process_inbound_message.await_args.kwargs[
                "adaptive_participation"
            ]
            is True
        )
        participation_latency = (
            adapter._process_inbound_message.await_args.kwargs[
                "participation_latency"
            ]
        )
        assert participation_latency["debounce_seconds"] >= 0
        assert participation_latency["classifier_seconds"] >= 0
        adapter._persist_observed_candidate.assert_not_awaited()
        assert candidate.key not in adapter._participation_pending

    @pytest.mark.asyncio
    async def test_silent_or_cooldown_candidate_is_persisted_observed(self):
        adapter = _adaptive_adapter()
        adapter.config = SimpleNamespace(
            extra={"_turn_latency_enabled": True}
        )
        candidate = _candidate()
        adapter._participation_generations[candidate.key] = candidate.generation
        adapter._participation_pending[candidate.key] = candidate
        adapter._classify_participation = AsyncMock(return_value=False)
        adapter._process_inbound_message = AsyncMock()
        adapter._persist_observed_candidate = AsyncMock()

        with patch(
            "gateway.turn_latency.emit_participation_only"
        ) as emit_latency:
            await adapter._run_participation_candidate(candidate)

        adapter._process_inbound_message.assert_not_awaited()
        adapter._persist_observed_candidate.assert_awaited_once_with(candidate)
        assert (
            emit_latency.call_args.kwargs["outcome"]
            == "participation_silent"
        )

        next_candidate = _candidate(
            message_id="om_next",
            generation=2,
        )
        adapter._participation_generations[candidate.key] = 2
        adapter._participation_pending[candidate.key] = next_candidate
        adapter._classify_participation = AsyncMock(return_value=True)
        adapter._session_store = _TranscriptStore(
            [
                {
                    "role": "assistant",
                    "content": "recent Hermes reply",
                    "timestamp": datetime.now().isoformat(),
                }
            ]
        )
        adapter._persist_observed_candidate.reset_mock()

        with patch(
            "gateway.turn_latency.emit_participation_only"
        ) as emit_latency:
            await adapter._run_participation_candidate(next_candidate)

        adapter._process_inbound_message.assert_not_awaited()
        adapter._persist_observed_candidate.assert_awaited_once_with(
            next_candidate
        )
        assert (
            emit_latency.call_args.kwargs["outcome"]
            == "participation_cooldown"
        )

    @pytest.mark.asyncio
    async def test_generation_change_cannot_remove_or_dispatch_newer_candidate(self):
        adapter = _adaptive_adapter()
        old = _candidate()
        new = _candidate(message_id="om_new", generation=2)
        adapter._participation_generations[old.key] = 1
        adapter._participation_pending[old.key] = old
        adapter._process_inbound_message = AsyncMock()
        adapter._persist_observed_candidate = AsyncMock()

        async def supersede_while_classifying(_candidate, **_kwargs):
            adapter._participation_generations[old.key] = 2
            adapter._participation_pending[old.key] = new
            return True

        adapter._classify_participation = supersede_while_classifying

        await adapter._run_participation_candidate(old)

        assert adapter._participation_pending[old.key] is new
        adapter._process_inbound_message.assert_not_awaited()
        adapter._persist_observed_candidate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_text_candidate_does_not_invalidate_pending_text(self):
        adapter = _adaptive_adapter()
        pending = _candidate()
        adapter._participation_pending[pending.key] = pending
        adapter._participation_generations[pending.key] = pending.generation
        adapter._build_participation_candidate = AsyncMock(return_value=None)

        await adapter._queue_participation_candidate(
            data=SimpleNamespace(),
            message=_group_message("ignored", message_id="om_media"),
            sender_id=SimpleNamespace(open_id="ou_alice"),
            message_id="om_media",
            is_bot=False,
        )

        assert adapter._participation_pending[pending.key] is pending
        assert adapter._participation_generations[pending.key] == 1

    @pytest.mark.asyncio
    async def test_direct_trigger_flushes_pending_context_before_dispatch(self):
        adapter = _adaptive_adapter()
        candidate = _candidate()
        sleeper = asyncio.create_task(asyncio.sleep(60))
        adapter._participation_pending[candidate.key] = candidate
        adapter._participation_tasks[candidate.key] = sleeper
        adapter._participation_generations[candidate.key] = 1
        adapter._persist_observed_candidate = AsyncMock()

        await adapter._flush_pending_participation(candidate.key)

        assert sleeper.cancelled()
        adapter._persist_observed_candidate.assert_awaited_once_with(candidate)
        assert candidate.key not in adapter._participation_pending
        assert candidate.key not in adapter._participation_tasks


class TestParticipationState:
    @pytest.mark.asyncio
    async def test_recent_same_sender_continuation_is_hot(self):
        adapter = _adaptive_adapter()
        adapter._session_store = _TranscriptStore(
            [
                {
                    "role": "assistant",
                    "content": "What do you think?",
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "role": "user",
                    "content": "I prefer the first option",
                    "timestamp": datetime.now().isoformat(),
                    "memory_source": {
                        "user_id_alt": "on_alice",
                        "user_name": "Alice",
                    },
                },
            ]
        )

        state = await adapter._participation_state(_candidate())

        assert state.hot is True
        assert state.seconds_since_hermes_reply is not None
        assert state.seconds_since_hermes_reply < 5
        assert state.human_messages_since_hermes_reply == 2
        assert state.same_sender_as_last_human is True
        assert state.recent_dialogue[-1]["sender"] == "Alice"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "rows",
        [
            [],
            [
                {
                    "role": "assistant",
                    "content": "old reply",
                    "timestamp": (
                        datetime.now() - timedelta(minutes=10)
                    ).isoformat(),
                }
            ],
            [
                {
                    "role": "assistant",
                    "content": "recent reply",
                    "timestamp": datetime.now().isoformat(),
                },
                {"role": "user", "content": "one"},
                {"role": "user", "content": "two"},
            ],
        ],
    )
    async def test_missing_stale_or_crowded_context_is_cold(self, rows):
        adapter = _adaptive_adapter()
        adapter._session_store = _TranscriptStore(rows)

        state = await adapter._participation_state(_candidate())

        assert state.hot is False

    @pytest.mark.asyncio
    async def test_hot_state_uses_lower_wake_threshold(self):
        adapter = _adaptive_adapter()
        response = object()
        hot_state = _FeishuParticipationState(
            recent_dialogue=[],
            seconds_since_hermes_reply=10,
            human_messages_since_hermes_reply=1,
            same_sender_as_last_human=True,
            hot=True,
        )
        cold_state = _FeishuParticipationState(
            recent_dialogue=[],
            seconds_since_hermes_reply=None,
            human_messages_since_hermes_reply=0,
            same_sender_as_last_human=False,
            hot=False,
        )

        with (
            patch(
                "agent.auxiliary_client.async_call_llm",
                new=AsyncMock(return_value=response),
            ),
            patch(
                "agent.auxiliary_client.extract_content_or_reasoning",
                return_value='{"decision":"candidate","confidence":0.6}',
            ),
        ):
            assert await adapter._classify_participation(
                _candidate(),
                state=hot_state,
            )
            assert not await adapter._classify_participation(
                _candidate(),
                state=cold_state,
            )

    @pytest.mark.asyncio
    async def test_uncertain_decision_defers_to_main_agent_in_hot_or_cold_chat(self):
        adapter = _adaptive_adapter()
        response = object()
        states = [
            _FeishuParticipationState([], 10, 1, True, True),
            _FeishuParticipationState([], None, 0, False, False),
        ]

        with (
            patch(
                "agent.auxiliary_client.async_call_llm",
                new=AsyncMock(return_value=response),
            ),
            patch(
                "agent.auxiliary_client.extract_content_or_reasoning",
                return_value='{"decision":"uncertain","confidence":0.5}',
            ),
        ):
            for state in states:
                assert await adapter._classify_participation(
                    _candidate(), state=state
                )


class TestParticipationIntegrationDefaults:
    @patch.dict(
        os.environ,
        {
            "BYTEPLUS_FAST_LLM_MODEL": "deepseek-v4-flash",
            "BYTEPLUS_API_BASE": "https://example.invalid/v1",
            "BYTEPLUS_API_KEY": "secret-test-value",
        },
        clear=True,
    )
    def test_byteplus_fast_model_bridges_to_dedicated_auxiliary_task(self):
        _apply_yaml_config({}, {})

        assert os.environ["AUXILIARY_FEISHU_PARTICIPATION_PROVIDER"] == "custom"
        assert (
            os.environ["AUXILIARY_FEISHU_PARTICIPATION_MODEL"]
            == "deepseek-v4-flash"
        )
        assert (
            os.environ["AUXILIARY_FEISHU_PARTICIPATION_BASE_URL"]
            == "https://example.invalid/v1"
        )
        assert (
            os.environ["AUXILIARY_FEISHU_PARTICIPATION_API_KEY"]
            == "secret-test-value"
        )

    def test_register_declares_auxiliary_task_and_platform(self):
        ctx = SimpleNamespace(
            register_auxiliary_task=Mock(),
            register_platform=Mock(),
        )

        register(ctx)

        ctx.register_auxiliary_task.assert_called_once()
        assert (
            ctx.register_auxiliary_task.call_args.kwargs["key"]
            == "feishu_participation"
        )
        assert (
            ctx.register_auxiliary_task.call_args.kwargs["defaults"]["timeout"]
            == 30
        )
        ctx.register_platform.assert_called_once()

    def test_feishu_typing_is_quiet_by_default_but_explicit_override_wins(self):
        with patch(
            "plugins.platforms.feishu.adapter.FeishuAdapter",
            side_effect=lambda config: config,
        ):
            default_config = _build_adapter(PlatformConfig())
            explicit_config = _build_adapter(
                PlatformConfig(
                    typing_indicator=True,
                    extra={"typing_indicator": True},
                )
            )

        assert default_config.typing_indicator is False
        assert explicit_config.typing_indicator is True
