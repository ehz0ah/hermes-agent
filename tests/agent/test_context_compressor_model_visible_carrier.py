"""Regression coverage for the model-visible replay half of issue #131104."""

from types import SimpleNamespace
from unittest.mock import patch

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    ContextCompressor,
    SUMMARY_PREFIX,
    _MERGED_PRIOR_CONTEXT_HEADER,
    _MERGED_SUMMARY_DELIMITER,
    _SUMMARY_END_MARKER,
)
from agent.turn_context import build_api_messages


def _compressor() -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        return ContextCompressor(model="test/model", quiet_mode=True)


def _wire_agent() -> SimpleNamespace:
    agent = SimpleNamespace(
        _current_turn_timestamp=1_700_000_000.0,
        api_mode="chat_completions",
        base_url="https://example.test/v1",
        model="test/model",
        ephemeral_system_prompt="",
    )
    agent._copy_reasoning_content_for_api = lambda _source, _target: None
    agent._should_sanitize_tool_calls = lambda: False
    return agent


def _wire(messages: list[dict]) -> list[dict]:
    wire, _ = build_api_messages(
        _wire_agent(),
        messages,
        current_turn_user_idx=len(messages) - 1,
        ext_prefetch_cache="",
        plugin_user_context="",
        moa_config=None,
        active_system_prompt="",
    )
    return wire


def test_new_merged_assistant_carrier_does_not_prime_header_replay():
    """The prior reply stays model-visible, but the imitable frame does not."""
    previous_reply = "The migration is complete."
    carrier = {
        "role": "assistant",
        "content": previous_reply,
        # A stale sidecar must not override the rewritten carrier on the wire.
        "api_content": previous_reply,
    }
    summary = f"{SUMMARY_PREFIX}\nThe migration completed successfully."

    _compressor()._merge_summary_into_tail_row(
        carrier, summary, summary_role="assistant", force_user_leading=False
    )
    wire = _wire([carrier, {"role": "user", "content": "What should I do next?"}])
    model_content = wire[0]["content"]

    assert model_content.startswith(previous_reply)
    assert _MERGED_PRIOR_CONTEXT_HEADER not in model_content
    assert _MERGED_SUMMARY_DELIMITER in model_content
    assert summary in model_content
    assert model_content.rstrip().endswith(_SUMMARY_END_MARKER)
    assert ContextCompressor._is_context_summary_content(model_content) is True


def test_legacy_merged_assistant_carrier_strips_header_from_wire_copy():
    """Existing persisted carriers self-heal without rewriting durable history."""
    previous_reply = "The migration is complete."
    legacy_content = (
        f"{_MERGED_PRIOR_CONTEXT_HEADER}\n{previous_reply}\n\n"
        f"{_MERGED_SUMMARY_DELIMITER}\n\n{SUMMARY_PREFIX}\n"
        f"The migration completed successfully.\n\n{_SUMMARY_END_MARKER}"
    )
    carrier = {
        "role": "assistant",
        "content": legacy_content,
        "api_content": legacy_content,
        COMPRESSED_SUMMARY_METADATA_KEY: True,
    }

    wire = _wire([carrier, {"role": "user", "content": "What should I do next?"}])

    assert wire[0]["content"].startswith(previous_reply)
    assert _MERGED_PRIOR_CONTEXT_HEADER not in wire[0]["content"]
    assert _MERGED_SUMMARY_DELIMITER in wire[0]["content"]
    assert carrier["content"] == legacy_content
    assert carrier["api_content"] == legacy_content


def test_merged_user_carrier_keeps_prior_context_safety_header():
    """The assistant-only fix must not weaken user-carrier instruction framing."""
    carrier = {"role": "user", "content": "Earlier user request."}
    summary = f"{SUMMARY_PREFIX}\nThe earlier request was completed."

    _compressor()._merge_summary_into_tail_row(
        carrier, summary, summary_role="user", force_user_leading=False
    )

    assert carrier["content"].startswith(_MERGED_PRIOR_CONTEXT_HEADER)
    assert "Earlier user request." in carrier["content"]
    assert ContextCompressor._is_context_summary_content(carrier["content"]) is True


def test_plain_assistant_text_that_mentions_header_is_unchanged():
    """Content is healed only when it has the full merged-carrier structure."""
    plain = f"{_MERGED_PRIOR_CONTEXT_HEADER}\nThis is quoted documentation."
    carrier = {"role": "assistant", "content": plain, "api_content": plain}

    wire = _wire([carrier, {"role": "user", "content": "Continue."}])

    assert wire[0]["content"] == plain


def test_legacy_multimodal_assistant_carrier_strips_only_header_part():
    """Legacy list content keeps its original text and media ordering."""
    previous_reply = {"type": "text", "text": "I inspected the screenshot."}
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
    summary = {
        "type": "text",
        "text": (
            f"\n\n{_MERGED_SUMMARY_DELIMITER}\n\n{SUMMARY_PREFIX}\n"
            f"The screenshot was inspected.\n\n{_SUMMARY_END_MARKER}"
        ),
    }
    content = [
        {"type": "text", "text": f"{_MERGED_PRIOR_CONTEXT_HEADER}\n"},
        previous_reply,
        image,
        summary,
    ]

    wire = _wire([
        {"role": "assistant", "content": content, COMPRESSED_SUMMARY_METADATA_KEY: True},
        {"role": "user", "content": "Continue."},
    ])

    assert wire[0]["content"] == [previous_reply, image, summary]
    assert content[0]["text"].startswith(_MERGED_PRIOR_CONTEXT_HEADER)
