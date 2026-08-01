from __future__ import annotations

import json
from pathlib import Path
import re
from unittest.mock import ANY

import pytest

from tools.lark_service import LarkApiResult, LarkService, LarkServiceError
from tools.lark_tools import (
    _BITABLE_ACTIONS,
    _CALENDAR_ACTIONS,
    _DOC_ACTIONS,
    _DRIVE_ACTIONS,
    _execute,
    _IM_ACTIONS,
    _MEETING_ACTIONS,
    _PEOPLE_ACTIONS,
    _schema,
    _TASK_ACTIONS,
    _TOOLS,
    _WIKI_ACTIONS,
)
from tools.registry import registry


class _FakeService:
    def __init__(self):
        self.calls = []

    def request(self, method, uri, **kwargs):
        self.calls.append((method, uri, kwargs))
        return LarkApiResult({"ok": True}, "req-1")

    def request_pages(self, method, uri, **kwargs):
        self.calls.append((method, uri, kwargs))
        return LarkApiResult({"items": []}, "req-2")

    def scope_audit(self, scopes):
        return {
            "available": [],
            "missing": [],
            "unverified": sorted(scopes),
        }

    def auth_audit(self):
        return {
            "tenant_token": "configured",
            "user_token": "missing",
        }

    def upload_task_attachment(self, **kwargs):
        self.calls.append(("UPLOAD_TASK_ATTACHMENT", "", kwargs))
        return LarkApiResult({"attachment": {"guid": "att-1"}}, "req-upload")

    def upload_file(self, **kwargs):
        self.calls.append(("UPLOAD_FILE", "", kwargs))
        return LarkApiResult({"file_token": "file-1"}, "req-upload-file")

    def download_file(self, **kwargs):
        self.calls.append(("DOWNLOAD_FILE", "", kwargs))
        return LarkApiResult({"path": kwargs["destination"]}, "req-download-file")

    def add_managed_reaction(self, **kwargs):
        self.calls.append(("ADD_REACTION", "", kwargs))
        return LarkApiResult(
            {
                "reaction_id": "reaction_1",
                "reaction_type": {"emoji_type": kwargs["emoji_type"]},
            },
            "req-reaction",
        )

    def remove_managed_reaction(self, **kwargs):
        self.calls.append(("REMOVE_REACTION", "", kwargs))
        return LarkApiResult(
            {"removed": True, "message_id": kwargs["message_id"]},
            "req-remove-reaction",
        )


@pytest.fixture
def fake_service(monkeypatch):
    service = _FakeService()
    monkeypatch.setattr(
        LarkService,
        "from_environment",
        classmethod(lambda cls: service),
    )
    return service


def _payload(result):
    return json.loads(result)


def test_lark_tools_are_registered_in_feishu_toolset_only():
    from toolsets import TOOLSETS

    names = {
        "lark_people",
        "lark_im",
        "lark_docs",
        "lark_wiki",
        "lark_drive",
        "lark_calendar",
        "lark_bitable",
        "lark_permissions",
    }
    assert names <= set(TOOLSETS["hermes-feishu"]["tools"])
    assert "lark_tasks" not in TOOLSETS["hermes-feishu"]["tools"]
    assert "lark_meetings" not in TOOLSETS["hermes-feishu"]["tools"]
    for toolset, definition in TOOLSETS.items():
        if toolset != "hermes-feishu":
            assert not names.intersection(definition["tools"])


def test_deferred_domains_and_meeting_rooms_are_not_exposed():
    from toolsets import TOOLSETS

    feishu_tools = set(TOOLSETS["hermes-feishu"]["tools"])
    assert "lark_tasks" not in feishu_tools
    assert "lark_meetings" not in feishu_tools
    assert "list_rooms" not in _TOOLS["lark_calendar"][1]
    assert "list_rooms" in _CALENDAR_ACTIONS


def test_action_schema_explains_required_parameters():
    schema = _schema(
        "lark_im",
        "Lark IM",
        {
            "list": _IM_ACTIONS["list_chats"],
            "reply": _IM_ACTIONS["reply"],
        },
    )

    description = schema["parameters"]["properties"]["params"]["description"]
    assert "list(no required params)" in description
    assert "reply(message_id, text)" in description


def test_people_resolution_requires_email_or_mobile(fake_service):
    result = _execute(
        "lark_people",
        _PEOPLE_ACTIONS,
        {"action": "resolve_users", "params": {}},
        {},
    )

    assert "error" in _payload(result)
    assert "emails or mobiles" in result
    assert fake_service.calls == []


def test_thread_context_uses_thread_container(fake_service):
    result = _execute(
        "lark_im",
        _IM_ACTIONS,
        {
            "action": "thread_context",
            "params": {"thread_id": "omt_1", "page_size": 50},
        },
        {},
    )

    assert "error" not in _payload(result)
    _, _, kwargs = fake_service.calls[0]
    assert kwargs["queries"] == {
        "page_size": 50,
        "container_id_type": "thread",
        "container_id": "omt_1",
    }


@pytest.mark.parametrize(
    ("tool_name", "actions", "action", "params", "method", "uri"),
    [
        (
            "lark_people",
            _PEOPLE_ACTIONS,
            "get_profile",
            {"user_id": "ou_1"},
            "GET",
            "/open-apis/contact/v3/users/:user_id",
        ),
        (
            "lark_im",
            _IM_ACTIONS,
            "get_chat",
            {"chat_id": "oc_1"},
            "GET",
            "/open-apis/im/v1/chats/:chat_id",
        ),
        (
            "lark_docs",
            _DOC_ACTIONS,
            "read",
            {"document_id": "doc_1"},
            "GET",
            "/open-apis/docx/v1/documents/:document_id/raw_content",
        ),
        (
            "lark_wiki",
            _WIKI_ACTIONS,
            "resolve_node",
            {"token": "wik_1", "obj_type": "docx"},
            "GET",
            "/open-apis/wiki/v2/spaces/get_node",
        ),
        (
            "lark_drive",
            _DRIVE_ACTIONS,
            "get_metadata",
            {"request_docs": [{"doc_token": "doc_1", "doc_type": "docx"}]},
            "POST",
            "/open-apis/drive/v1/metas/batch_query",
        ),
        (
            "lark_calendar",
            _CALENDAR_ACTIONS,
            "agenda",
            {"calendar_id": "primary", "page_size": 50},
            "GET",
            "/open-apis/calendar/v4/calendars/:calendar_id/events",
        ),
        (
            "lark_tasks",
            _TASK_ACTIONS,
            "get",
            {"task_guid": "task_1"},
            "GET",
            "/open-apis/task/v2/tasks/:task_guid",
        ),
        (
            "lark_bitable",
            _BITABLE_ACTIONS,
            "list_fields",
            {"app_token": "app_1", "table_id": "tbl_1"},
            "GET",
            "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/fields",
        ),
        (
            "lark_meetings",
            _MEETING_ACTIONS,
            "transcript",
            {"minute_token": "min_1", "need_speaker": True},
            "GET",
            "/open-apis/minutes/v1/minutes/:minute_token/transcript",
        ),
    ],
)
def test_each_lark_domain_routes_to_its_official_endpoint(
    fake_service,
    tool_name,
    actions,
    action,
    params,
    method,
    uri,
):
    payload = _payload(
        _execute(
            tool_name,
            actions,
            {"action": action, "params": params},
            {},
        )
    )

    assert payload["success"] is True
    assert fake_service.calls[0][0:2] == (method, uri)


def test_action_specs_match_installed_lark_sdk_request_contracts():
    lark = pytest.importorskip("lark_oapi")
    sdk_root = Path(next(iter(lark.__path__))) / "api"
    contracts: dict[tuple[str, str], list[tuple[set[str], set[str]]]] = {}

    for path in sdk_root.rglob("*_request.py"):
        source = path.read_text(encoding="utf-8")
        uri = re.search(r'\.uri = ["\']([^"\']+)', source)
        method = re.search(r"\.http_method = HttpMethod\.([A-Z]+)", source)
        if not uri or not method:
            continue
        token_types = set(re.findall(r"AccessTokenType\.([A-Z]+)", source))
        query_keys = set(re.findall(r'\.add_query\(["\']([^"\']+)', source))
        contracts.setdefault((method.group(1), uri.group(1)), []).append(
            (token_types, query_keys)
        )

    for tool_name, (_, actions) in _TOOLS.items():
        for action, spec in actions.items():
            matches = contracts.get((spec.method, spec.uri))
            assert matches, (
                f"{tool_name}.{action} has no matching request in the installed "
                f"Lark SDK: {spec.method} {spec.uri}"
            )
            expected_token = spec.auth.upper()
            assert any(expected_token in tokens for tokens, _ in matches), (
                f"{tool_name}.{action} uses {spec.auth} auth, but the installed "
                f"Lark SDK only permits "
                f"{sorted(set().union(*(tokens for tokens, _ in matches)))}"
            )
            supported_queries = set().union(*(queries for _, queries in matches))
            assert set(spec.query_keys) <= supported_queries, (
                f"{tool_name}.{action} advertises unsupported query parameters: "
                f"{sorted(set(spec.query_keys) - supported_queries)}"
            )


def test_lark_actions_declare_current_official_scopes():
    expected = {
        "im.list_chats": (_IM_ACTIONS["list_chats"], ("im:chat:read",)),
        "im.send": (_IM_ACTIONS["send"], ("im:message:send_as_bot",)),
        "docs.create": (
            _DOC_ACTIONS["create"],
            ("docx:document:create",),
        ),
        "docs.append_blocks": (
            _DOC_ACTIONS["append_blocks"],
            ("docx:document:write_only",),
        ),
        "docs.list_comments": (
            _DOC_ACTIONS["list_comments"],
            ("docs:document.comment:read",),
        ),
        "wiki.list_spaces": (
            _WIKI_ACTIONS["list_spaces"],
            ("wiki:space:retrieve",),
        ),
        "wiki.create_node": (
            _WIKI_ACTIONS["create_node"],
            ("wiki:node:create",),
        ),
        "drive.list": (
            _DRIVE_ACTIONS["list"],
            ("space:document:retrieve",),
        ),
        "drive.upload": (
            _DRIVE_ACTIONS["upload"],
            ("drive:file:upload",),
        ),
        "drive.download": (
            _DRIVE_ACTIONS["download"],
            ("drive:file:download",),
        ),
        "calendar.agenda": (
            _CALENDAR_ACTIONS["agenda"],
            ("calendar:calendar.event:read",),
        ),
        "calendar.create_event": (
            _CALENDAR_ACTIONS["create_event"],
            ("calendar:calendar.event:create",),
        ),
        "bitable.query_records": (
            _BITABLE_ACTIONS["query_records"],
            ("base:record:retrieve",),
        ),
        "bitable.create_record": (
            _BITABLE_ACTIONS["create_record"],
            ("base:record:create",),
        ),
        "meetings.transcript": (
            _MEETING_ACTIONS["transcript"],
            ("minutes:minutes:readonly",),
        ),
    }

    for label, (spec, scopes) in expected.items():
        assert spec.scopes == scopes, label


def test_drive_upload_uses_file_api_and_requires_approval(
    fake_service,
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "report.txt"
    source.write_text("report", encoding="utf-8")
    monkeypatch.setattr(
        "tools.lark_tools.request_tool_approval",
        lambda *args, **kwargs: {"approved": True},
    )

    payload = _payload(
        _execute(
            "lark_drive",
            _DRIVE_ACTIONS,
            {
                "action": "upload",
                "params": {
                    "file_path": str(source),
                    "parent_type": "explorer",
                    "parent_node": "fld_1",
                },
            },
            {"session_id": "sid", "user_task": "upload report"},
        )
    )

    assert payload["success"] is True
    assert fake_service.calls == [
        (
            "UPLOAD_FILE",
            "",
            {
                "file_path": str(source),
                "parent_type": "explorer",
                "parent_node": "fld_1",
                "idempotency_key": ANY,
            },
        )
    ]


def test_drive_download_uses_file_api(fake_service, tmp_path):
    destination = tmp_path / "report.txt"

    payload = _payload(
        _execute(
            "lark_drive",
            _DRIVE_ACTIONS,
            {
                "action": "download",
                "params": {
                    "file_token": "file_1",
                    "destination": str(destination),
                    "version": "42",
                },
            },
            {},
        )
    )

    assert payload["success"] is True
    assert fake_service.calls == [
        (
            "DOWNLOAD_FILE",
            "",
            {
                "file_token": "file_1",
                "destination": str(destination),
                "version": "42",
            },
        )
    ]


def test_search_messages_filters_bounded_container_history(fake_service):
    def request_pages(method, uri, **kwargs):
        fake_service.calls.append((method, uri, kwargs))
        return LarkApiResult(
            {
                "items": [
                    {"message_id": "om_1", "body": {"content": "Project Atlas"}},
                    {"message_id": "om_2", "body": {"content": "Lunch"}},
                    {"message_id": "om_3", "body": {"content": "atlas launch"}},
                ]
            },
            "req-search",
        )

    fake_service.request_pages = request_pages
    payload = _payload(
        _execute(
            "lark_im",
            _IM_ACTIONS,
            {
                "action": "search_messages",
                "params": {
                    "container_id": "oc_chat",
                    "container_id_type": "chat",
                    "query": "ATLAS",
                    "limit": 1,
                },
            },
            {},
        )
    )

    assert payload["data"] == {
        "items": [
            {"message_id": "om_1", "body": {"content": "Project Atlas"}}
        ],
        "matched": 2,
        "scanned": 3,
    }
    assert fake_service.calls[0][2]["queries"] == {
        "container_id": "oc_chat",
        "container_id_type": "chat",
    }


def test_calendar_recommend_times_uses_freebusy_and_computes_slots(fake_service):
    def request(method, uri, **kwargs):
        fake_service.calls.append((method, uri, kwargs))
        return LarkApiResult(
            {
                "freebusy_lists": [
                    {
                        "user_id": "ou_1",
                        "freebusy_items": [
                            {
                                "start_time": "2026-07-29T10:00:00+08:00",
                                "end_time": "2026-07-29T11:00:00+08:00",
                            }
                        ],
                    }
                ]
            },
            "req-freebusy",
        )

    fake_service.request = request
    payload = _payload(
        _execute(
            "lark_calendar",
            _CALENDAR_ACTIONS,
            {
                "action": "recommend_times",
                "params": {
                    "time_min": "2026-07-29T09:00:00+08:00",
                    "time_max": "2026-07-29T12:00:00+08:00",
                    "user_ids": ["ou_1"],
                    "duration_minutes": 30,
                    "max_suggestions": 3,
                    "user_id_type": "open_id",
                },
            },
            {},
        )
    )

    assert payload["data"]["suggestions"] == [
        {
            "start_time": "2026-07-29T09:00:00+08:00",
            "end_time": "2026-07-29T09:30:00+08:00",
        },
        {
            "start_time": "2026-07-29T09:30:00+08:00",
            "end_time": "2026-07-29T10:00:00+08:00",
        },
        {
            "start_time": "2026-07-29T11:00:00+08:00",
            "end_time": "2026-07-29T11:30:00+08:00",
        },
    ]
    _, _, kwargs = fake_service.calls[0]
    assert kwargs["queries"] == {"user_id_type": "open_id"}
    assert kwargs["body"] == {
        "time_min": "2026-07-29T09:00:00+08:00",
        "time_max": "2026-07-29T12:00:00+08:00",
        "user_ids": ["ou_1"],
    }


def test_task_assignment_uses_update_members_contract(
    fake_service,
    monkeypatch,
):
    monkeypatch.setattr(
        "tools.lark_tools.request_tool_approval",
        lambda *args, **kwargs: {"approved": True},
    )
    payload = _payload(
        _execute(
            "lark_tasks",
            _TASK_ACTIONS,
            {
                "action": "assign",
                "params": {
                    "task_guid": "task_1",
                    "members": [{"id": "ou_1", "type": "user", "role": "assignee"}],
                },
            },
            {},
        )
    )

    assert payload["success"] is True
    _, _, kwargs = fake_service.calls[0]
    assert kwargs["body"] == {
        "task": {
            "members": [{"id": "ou_1", "type": "user", "role": "assignee"}]
        },
        "update_fields": ["members"],
    }


@pytest.mark.parametrize(
    ("tool_name", "actions", "action", "params", "expected_body"),
    [
        (
            "lark_docs",
            _DOC_ACTIONS,
            "update_block",
            {
                "document_id": "doc_1",
                "block_id": "blk_1",
                "document_revision_id": 3,
                "update": {
                    "update_text": {
                        "elements": [
                            {"text_run": {"content": "Updated text"}},
                        ]
                    }
                },
            },
            {
                "update_text": {
                    "elements": [
                        {"text_run": {"content": "Updated text"}},
                    ]
                }
            },
        ),
        (
            "lark_calendar",
            _CALENDAR_ACTIONS,
            "update_event",
            {
                "calendar_id": "primary",
                "event_id": "event_1",
                "user_id_type": "open_id",
                "update": {"summary": "Updated meeting"},
            },
            {"summary": "Updated meeting"},
        ),
        (
            "lark_tasks",
            _TASK_ACTIONS,
            "update",
            {
                "task_guid": "task_1",
                "task": {"summary": "Updated task"},
                "update_fields": ["summary"],
            },
            {
                "task": {"summary": "Updated task"},
                "update_fields": ["summary"],
            },
        ),
    ],
)
def test_update_actions_follow_official_body_contracts(
    fake_service,
    monkeypatch,
    tool_name,
    actions,
    action,
    params,
    expected_body,
):
    monkeypatch.setattr(
        "tools.lark_tools.request_tool_approval",
        lambda *args, **kwargs: {"approved": True},
    )

    payload = _payload(
        _execute(
            tool_name,
            actions,
            {"action": action, "params": params},
            {},
        )
    )

    assert payload["success"] is True
    _, _, kwargs = fake_service.calls[0]
    assert kwargs["body"] == expected_body
    assert "update" not in kwargs["body"]


def test_task_update_requires_task_object(fake_service):
    result = _execute(
        "lark_tasks",
        _TASK_ACTIONS,
        {
            "action": "update",
            "params": {
                "task_guid": "task_1",
                "update_fields": ["summary"],
            },
        },
        {},
    )

    assert "Missing required parameters for update: task" in result
    assert fake_service.calls == []


def test_meeting_search_uses_official_post_contract(fake_service):
    payload = _payload(
        _execute(
            "lark_meetings",
            _MEETING_ACTIONS,
            {
                "action": "search",
                "params": {
                    "query": "quarterly review",
                    "meeting_filter": {
                        "start_time": "1751328000",
                        "end_time": "1754006400",
                    },
                    "page_size": 20,
                },
            },
            {},
        )
    )

    assert payload["success"] is True
    method, uri, kwargs = fake_service.calls[0]
    assert method == "POST"
    assert uri == "/open-apis/vc/v1/meetings/search"
    assert kwargs["queries"] == {"page_size": 20}
    assert kwargs["body"] == {
        "query": "quarterly review",
        "meeting_filter": {
            "start_time": "1751328000",
            "end_time": "1754006400",
        },
    }
    assert kwargs["auth"] == "user"


def test_minute_search_requires_user_oauth(fake_service):
    payload = _payload(
        _execute(
            "lark_meetings",
            _MEETING_ACTIONS,
            {
                "action": "search_minutes",
                "params": {"query": "quarterly review"},
            },
            {},
        )
    )

    assert payload["success"] is True
    assert fake_service.calls[0][2]["auth"] == "user"


def test_task_attachment_uses_sdk_upload_path(fake_service, monkeypatch):
    monkeypatch.setattr(
        "tools.lark_tools.request_tool_approval",
        lambda *args, **kwargs: {"approved": True},
    )
    payload = _payload(
        _execute(
            "lark_tasks",
            _TASK_ACTIONS,
            {
                "action": "upload_attachment",
                "params": {
                    "file_path": "/tmp/report.txt",
                    "resource_type": "task",
                    "resource_id": "task_1",
                },
            },
            {},
        )
    )

    assert payload["success"] is True
    assert fake_service.calls[0][0] == "UPLOAD_TASK_ATTACHMENT"
    assert fake_service.calls[0][2]["resource_id"] == "task_1"


def test_meeting_action_items_extracts_minute_todos(fake_service):
    def request(method, uri, **kwargs):
        fake_service.calls.append((method, uri, kwargs))
        return LarkApiResult(
            {"minute_todos": [{"todo_id": "todo_1", "content": "Ship it"}]},
            "req-minute",
        )

    fake_service.request = request
    payload = _payload(
        _execute(
            "lark_meetings",
            _MEETING_ACTIONS,
            {
                "action": "action_items",
                "params": {"minute_token": "min_1"},
            },
            {},
        )
    )

    assert payload["data"] == {
        "action_items": [{"todo_id": "todo_1", "content": "Ship it"}]
    }


def test_send_formats_mentions_and_honors_approval(fake_service, monkeypatch):
    monkeypatch.setattr(
        "tools.lark_tools.request_tool_approval",
        lambda *args, **kwargs: {"approved": True},
    )
    result = _execute(
        "lark_im",
        _IM_ACTIONS,
        {
            "action": "send",
            "params": {
                "receive_id": "oc_chat",
                "receive_id_type": "chat_id",
                "text": "hello",
                "mentions": [{"user_id": "ou_user", "name": "Alice"}],
            },
        },
        {"session_id": "sid", "user_task": "say hello"},
    )

    assert "error" not in _payload(result)
    _, _, kwargs = fake_service.calls[0]
    content = json.loads(kwargs["body"]["content"])
    assert content["text"] == '<at user_id="ou_user">Alice</at> hello'
    assert kwargs["queries"]["receive_id_type"] == "chat_id"
    assert kwargs["idempotency_key"]
    assert kwargs["retries"] == 0


def test_react_uses_safe_alias_without_interactive_approval(
    fake_service,
    monkeypatch,
):
    monkeypatch.setattr(
        "tools.lark_tools.request_tool_approval",
        lambda *args, **kwargs: pytest.fail("reactions must not prompt"),
    )

    payload = _payload(
        _execute(
            "lark_im",
            _IM_ACTIONS,
            {
                "action": "react",
                "params": {"message_id": "om_1", "emoji": "👍"},
            },
            {"session_id": "sid"},
        )
    )

    assert payload["success"] is True
    method, _, kwargs = fake_service.calls[0]
    assert method == "ADD_REACTION"
    assert kwargs["message_id"] == "om_1"
    assert kwargs["emoji_type"] == "THUMBSUP"
    assert kwargs["scopes"] == ("im:message.reactions:write",)
    assert kwargs["idempotency_key"]


def test_react_rejects_emoji_outside_allowlist(fake_service):
    payload = _payload(
        _execute(
            "lark_im",
            _IM_ACTIONS,
            {
                "action": "react",
                "params": {"message_id": "om_1", "emoji": "Typing"},
            },
            {},
        )
    )

    assert "emoji must be one of" in payload["error"]
    assert fake_service.calls == []


def test_remove_reaction_uses_only_service_managed_handle(fake_service):
    payload = _payload(
        _execute(
            "lark_im",
            _IM_ACTIONS,
            {
                "action": "remove_reaction",
                "params": {"message_id": "om_1"},
            },
            {},
        )
    )

    assert payload["success"] is True
    method, _, kwargs = fake_service.calls[0]
    assert method == "REMOVE_REACTION"
    assert kwargs == {
        "message_id": "om_1",
        "scopes": ("im:message.reactions:write",),
    }


def test_write_denial_never_calls_lark(fake_service, monkeypatch):
    monkeypatch.setattr(
        "tools.lark_tools.request_tool_approval",
        lambda *args, **kwargs: {"approved": False, "message": "denied"},
    )

    result = _execute(
        "lark_im",
        _IM_ACTIONS,
        {
            "action": "send",
            "params": {"receive_id": "oc_chat", "text": "hello"},
        },
        {},
    )

    assert _payload(result)["error"] == "denied"
    assert fake_service.calls == []


def test_scope_failure_returns_actionable_diagnostics(monkeypatch):
    class _DeniedService(_FakeService):
        def request(self, method, uri, **kwargs):
            raise LarkServiceError(
                "permission denied",
                code=99991663,
                missing_scopes=("im:message",),
            )

    denied = _DeniedService()
    monkeypatch.setattr(
        LarkService,
        "from_environment",
        classmethod(lambda cls: denied),
    )
    monkeypatch.setattr(
        "tools.lark_tools.request_tool_approval",
        lambda *args, **kwargs: {"approved": True},
    )

    payload = _payload(
        _execute(
            "lark_im",
            _IM_ACTIONS,
            {
                "action": "send",
                "params": {"receive_id": "oc_chat", "text": "hello"},
            },
            {},
        )
    )

    assert payload["error"] == "permission denied"
    assert payload["missing_scopes"] == ["im:message"]
    assert "publish" in payload["resolution"]


def test_permission_audit_does_not_expose_credentials(fake_service, monkeypatch):
    entry = registry.get_entry("lark_permissions")
    assert entry is not None

    result = entry.handler({})

    payload = _payload(result)
    assert payload["success"] is True
    assert "unverified" in payload
    assert payload["authentication"] == {
        "tenant_token": "configured",
        "user_token": "missing",
    }
    assert payload["active_tools"] == sorted(
        {
            "lark_people",
            "lark_im",
            "lark_docs",
            "lark_wiki",
            "lark_drive",
            "lark_calendar",
            "lark_bitable",
        }
    )
    assert payload["deferred_domains"] == {
        "calendar_rooms": "Deferred until meeting-room permissions are approved.",
        "lark_meetings": "Deferred for a later release.",
        "lark_tasks": "Deferred because the team does not use native Feishu Tasks.",
    }
    assert "secret" not in result.lower()
