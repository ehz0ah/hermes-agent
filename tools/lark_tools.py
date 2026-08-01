"""Compact first-party Feishu/Lark tools backed by the official SDK.

The model sees one action-oriented tool per product domain instead of dozens
of low-level endpoint schemas.  Endpoint definitions are intentionally kept in
this module so supported actions, permission diagnostics, approval behavior,
and idempotency are reviewed together.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from typing import Any, Mapping, Sequence

from tools.approval import request_tool_approval
from tools.lark_service import (
    LarkApiResult,
    LarkService,
    LarkServiceError,
    lark_sdk_available,
)
from tools.registry import registry, tool_error, tool_result


@dataclass(frozen=True)
class ActionSpec:
    method: str
    uri: str
    required: tuple[str, ...] = ()
    path_keys: tuple[str, ...] = ()
    query_keys: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()
    write: bool = False
    approval_required: bool = True
    paginate: bool = False
    custom: str = ""
    auth: str = "tenant"


_PEOPLE_ACTIONS = {
    "resolve_users": ActionSpec(
        "POST",
        "/open-apis/contact/v3/users/batch_get_id",
        scopes=("contact:user.id:readonly",),
        custom="resolve_users",
    ),
    "get_profile": ActionSpec(
        "GET",
        "/open-apis/contact/v3/users/:user_id",
        required=("user_id",),
        path_keys=("user_id",),
        query_keys=("user_id_type", "department_id_type"),
        scopes=("contact:user.base:readonly",),
    ),
    "list_department_users": ActionSpec(
        "GET",
        "/open-apis/contact/v3/users/find_by_department",
        required=("department_id",),
        query_keys=(
            "department_id",
            "department_id_type",
            "user_id_type",
            "page_size",
        ),
        scopes=("contact:user.base:readonly",),
        paginate=True,
    ),
    "get_department": ActionSpec(
        "GET",
        "/open-apis/contact/v3/departments/:department_id",
        required=("department_id",),
        path_keys=("department_id",),
        query_keys=("department_id_type", "user_id_type"),
        scopes=("contact:department.base:readonly",),
    ),
    "list_group_members": ActionSpec(
        "GET",
        "/open-apis/im/v1/chats/:chat_id/members",
        required=("chat_id",),
        path_keys=("chat_id",),
        query_keys=("member_id_type", "page_size"),
        scopes=("im:chat.members:read",),
        paginate=True,
    ),
}

_IM_ACTIONS = {
    "search_messages": ActionSpec(
        "GET",
        "/open-apis/im/v1/messages",
        required=("container_id", "query"),
        query_keys=(
            "container_id_type",
            "container_id",
            "start_time",
            "end_time",
            "sort_type",
            "page_size",
            "only_thread_root_messages",
        ),
        scopes=("im:message:readonly",),
        paginate=True,
        custom="search_messages",
    ),
    "list_messages": ActionSpec(
        "GET",
        "/open-apis/im/v1/messages",
        required=("container_id",),
        query_keys=(
            "container_id_type",
            "container_id",
            "start_time",
            "end_time",
            "sort_type",
            "page_size",
            "only_thread_root_messages",
        ),
        scopes=("im:message:readonly",),
        paginate=True,
    ),
    "get_message": ActionSpec(
        "GET",
        "/open-apis/im/v1/messages/:message_id",
        required=("message_id",),
        path_keys=("message_id",),
        scopes=("im:message:readonly",),
    ),
    "thread_context": ActionSpec(
        "GET",
        "/open-apis/im/v1/messages",
        required=("thread_id",),
        query_keys=("page_size", "sort_type"),
        scopes=("im:message:readonly",),
        paginate=True,
        custom="thread_context",
    ),
    "list_chats": ActionSpec(
        "GET",
        "/open-apis/im/v1/chats",
        query_keys=("user_id_type", "sort_type", "page_size"),
        scopes=("im:chat:read",),
        paginate=True,
    ),
    "get_chat": ActionSpec(
        "GET",
        "/open-apis/im/v1/chats/:chat_id",
        required=("chat_id",),
        path_keys=("chat_id",),
        scopes=("im:chat:read",),
    ),
    "send": ActionSpec(
        "POST",
        "/open-apis/im/v1/messages",
        required=("receive_id", "text"),
        query_keys=("receive_id_type",),
        scopes=("im:message:send_as_bot",),
        write=True,
        custom="send",
    ),
    "reply": ActionSpec(
        "POST",
        "/open-apis/im/v1/messages/:message_id/reply",
        required=("message_id", "text"),
        path_keys=("message_id",),
        scopes=("im:message:send_as_bot",),
        write=True,
        custom="reply",
    ),
    "react": ActionSpec(
        "POST",
        "/open-apis/im/v1/messages/:message_id/reactions",
        required=("message_id", "emoji"),
        path_keys=("message_id",),
        scopes=("im:message.reactions:write",),
        write=True,
        approval_required=False,
        custom="react",
    ),
    "remove_reaction": ActionSpec(
        "DELETE",
        "/open-apis/im/v1/messages/:message_id/reactions/:reaction_id",
        required=("message_id",),
        path_keys=("message_id",),
        scopes=("im:message.reactions:write",),
        write=True,
        approval_required=False,
        custom="remove_reaction",
    ),
}

_SAFE_REACTION_EMOJIS = {
    "APPLAUSE",
    "CLAP",
    "EYES",
    "FIRE",
    "HEART",
    "LAUGH",
    "OK",
    "PARTY",
    "SMILE",
    "THANKS",
    "THUMBSUP",
}
_REACTION_ALIASES = {
    "👏": "APPLAUSE",
    "👀": "EYES",
    "🔥": "FIRE",
    "❤": "HEART",
    "❤️": "HEART",
    "😂": "LAUGH",
    "🤣": "LAUGH",
    "👌": "OK",
    "🎉": "PARTY",
    "😊": "SMILE",
    "🙏": "THANKS",
    "👍": "THUMBSUP",
}

_DOC_ACTIONS = {
    "search": ActionSpec(
        "POST",
        "/open-apis/search/v2/doc_wiki/search",
        required=("query",),
        scopes=("search:docs:read",),
    ),
    "read": ActionSpec(
        "GET",
        "/open-apis/docx/v1/documents/:document_id/raw_content",
        required=("document_id",),
        path_keys=("document_id",),
        scopes=("docx:document:readonly",),
    ),
    "get": ActionSpec(
        "GET",
        "/open-apis/docx/v1/documents/:document_id",
        required=("document_id",),
        path_keys=("document_id",),
        scopes=("docx:document:readonly",),
    ),
    "create": ActionSpec(
        "POST",
        "/open-apis/docx/v1/documents",
        required=("title",),
        scopes=("docx:document:create",),
        write=True,
    ),
    "list_blocks": ActionSpec(
        "GET",
        "/open-apis/docx/v1/documents/:document_id/blocks",
        required=("document_id",),
        path_keys=("document_id",),
        query_keys=("page_size", "document_revision_id"),
        scopes=("docx:document:readonly",),
        paginate=True,
    ),
    "append_blocks": ActionSpec(
        "POST",
        "/open-apis/docx/v1/documents/:document_id/blocks/:block_id/children",
        required=("document_id", "block_id", "children"),
        path_keys=("document_id", "block_id"),
        scopes=("docx:document:write_only",),
        write=True,
    ),
    "update_block": ActionSpec(
        "PATCH",
        "/open-apis/docx/v1/documents/:document_id/blocks/:block_id",
        required=("document_id", "block_id", "update"),
        path_keys=("document_id", "block_id"),
        query_keys=("document_revision_id", "client_token", "user_id_type"),
        scopes=("docx:document:write_only",),
        write=True,
        custom="unwrap_update",
    ),
    "list_comments": ActionSpec(
        "GET",
        "/open-apis/drive/v1/files/:file_token/comments",
        required=("file_token",),
        path_keys=("file_token",),
        query_keys=("file_type", "is_solved", "page_size", "user_id_type"),
        scopes=("docs:document.comment:read",),
        paginate=True,
    ),
    "add_comment": ActionSpec(
        "POST",
        "/open-apis/drive/v1/files/:file_token/comments",
        required=("file_token", "file_type", "reply_list"),
        path_keys=("file_token",),
        query_keys=("file_type", "user_id_type"),
        scopes=("docs:document.comment:create",),
        write=True,
    ),
}

_WIKI_ACTIONS = {
    "search": ActionSpec(
        "POST",
        "/open-apis/wiki/v1/nodes/search",
        required=("query",),
        scopes=("wiki:wiki:readonly",),
        auth="user",
    ),
    "list_spaces": ActionSpec(
        "GET",
        "/open-apis/wiki/v2/spaces",
        query_keys=("page_size",),
        scopes=("wiki:space:retrieve",),
        paginate=True,
    ),
    "get_space": ActionSpec(
        "GET",
        "/open-apis/wiki/v2/spaces/:space_id",
        required=("space_id",),
        path_keys=("space_id",),
        scopes=("wiki:space:read",),
    ),
    "list_nodes": ActionSpec(
        "GET",
        "/open-apis/wiki/v2/spaces/:space_id/nodes",
        required=("space_id",),
        path_keys=("space_id",),
        query_keys=("parent_node_token", "page_size"),
        scopes=("wiki:node:retrieve",),
        paginate=True,
    ),
    "resolve_node": ActionSpec(
        "GET",
        "/open-apis/wiki/v2/spaces/get_node",
        required=("token", "obj_type"),
        query_keys=("token", "obj_type"),
        scopes=("wiki:node:read",),
    ),
    "create_node": ActionSpec(
        "POST",
        "/open-apis/wiki/v2/spaces/:space_id/nodes",
        required=("space_id", "obj_type", "node_type", "title"),
        path_keys=("space_id",),
        scopes=("wiki:node:create",),
        write=True,
    ),
    "move_node": ActionSpec(
        "POST",
        "/open-apis/wiki/v2/spaces/:space_id/nodes/:node_token/move",
        required=("space_id", "node_token", "target_parent_token"),
        path_keys=("space_id", "node_token"),
        scopes=("wiki:node:move",),
        write=True,
    ),
}

_DRIVE_ACTIONS = {
    "list": ActionSpec(
        "GET",
        "/open-apis/drive/v1/files",
        query_keys=("folder_token", "order_by", "direction", "page_size"),
        scopes=("space:document:retrieve",),
        paginate=True,
    ),
    "get_metadata": ActionSpec(
        "POST",
        "/open-apis/drive/v1/metas/batch_query",
        required=("request_docs",),
        scopes=("drive:drive.metadata:readonly",),
    ),
    "create_folder": ActionSpec(
        "POST",
        "/open-apis/drive/v1/files/create_folder",
        required=("name", "folder_token"),
        scopes=("space:folder:create",),
        write=True,
    ),
    "move": ActionSpec(
        "POST",
        "/open-apis/drive/v1/files/:file_token/move",
        required=("file_token", "type", "folder_token"),
        path_keys=("file_token",),
        scopes=("space:document:move",),
        write=True,
    ),
    "upload": ActionSpec(
        "POST",
        "/open-apis/drive/v1/files/upload_all",
        required=("file_path", "parent_type", "parent_node"),
        scopes=("drive:file:upload",),
        write=True,
        custom="upload",
    ),
    "download": ActionSpec(
        "GET",
        "/open-apis/drive/v1/files/:file_token/download",
        required=("file_token", "destination"),
        scopes=("drive:file:download",),
        custom="download",
    ),
    "list_permissions": ActionSpec(
        "GET",
        "/open-apis/drive/v1/permissions/:token/members",
        required=("token", "type"),
        path_keys=("token",),
        query_keys=("type", "perm_type", "fields"),
        scopes=("docs:permission.member:retrieve",),
    ),
    "add_permission": ActionSpec(
        "POST",
        "/open-apis/drive/v1/permissions/:token/members",
        required=("token", "type", "member_type", "member_id", "perm"),
        path_keys=("token",),
        query_keys=("type",),
        scopes=("docs:permission.member:create",),
        write=True,
    ),
}

_CALENDAR_ACTIONS = {
    "primary": ActionSpec(
        "POST",
        "/open-apis/calendar/v4/calendars/primary",
        query_keys=("user_id_type",),
        scopes=("calendar:calendar:read",),
    ),
    "agenda": ActionSpec(
        "GET",
        "/open-apis/calendar/v4/calendars/:calendar_id/events",
        required=("calendar_id",),
        path_keys=("calendar_id",),
        query_keys=("start_time", "end_time", "page_size", "user_id_type"),
        scopes=("calendar:calendar.event:read",),
        paginate=True,
    ),
    "search_events": ActionSpec(
        "POST",
        "/open-apis/calendar/v4/calendars/:calendar_id/events/search",
        required=("calendar_id", "query"),
        path_keys=("calendar_id",),
        scopes=("calendar:calendar.event:read",),
    ),
    "free_busy": ActionSpec(
        "POST",
        "/open-apis/calendar/v4/freebusy/batch",
        required=("time_min", "time_max", "user_ids"),
        query_keys=("user_id_type",),
        scopes=("calendar:calendar.free_busy:read",),
    ),
    "recommend_times": ActionSpec(
        "POST",
        "/open-apis/calendar/v4/freebusy/batch",
        required=("time_min", "time_max", "user_ids", "duration_minutes"),
        query_keys=("user_id_type",),
        scopes=("calendar:calendar.free_busy:read",),
        custom="recommend_times",
    ),
    "list_rooms": ActionSpec(
        "GET",
        "/open-apis/vc/v1/rooms",
        query_keys=("room_level_id", "page_size"),
        scopes=("vc:room:readonly",),
        paginate=True,
    ),
    "create_event": ActionSpec(
        "POST",
        "/open-apis/calendar/v4/calendars/:calendar_id/events",
        required=("calendar_id", "summary", "start_time", "end_time"),
        path_keys=("calendar_id",),
        query_keys=("user_id_type",),
        scopes=("calendar:calendar.event:create",),
        write=True,
    ),
    "update_event": ActionSpec(
        "PATCH",
        "/open-apis/calendar/v4/calendars/:calendar_id/events/:event_id",
        required=("calendar_id", "event_id", "update"),
        path_keys=("calendar_id", "event_id"),
        query_keys=("user_id_type",),
        scopes=("calendar:calendar.event:update",),
        write=True,
        custom="unwrap_update",
    ),
    "add_attendees": ActionSpec(
        "POST",
        "/open-apis/calendar/v4/calendars/:calendar_id/events/:event_id/attendees",
        required=("calendar_id", "event_id", "attendees"),
        path_keys=("calendar_id", "event_id"),
        query_keys=("user_id_type",),
        scopes=("calendar:calendar.event:update",),
        write=True,
    ),
}

_ACTIVE_CALENDAR_ACTIONS = {
    action: spec
    for action, spec in _CALENDAR_ACTIONS.items()
    if action != "list_rooms"
}

_TASK_ACTIONS = {
    "list": ActionSpec(
        "GET",
        "/open-apis/task/v2/tasks",
        query_keys=(
            "page_size",
            "completed",
            "type",
            "agent_task_status",
            "user_id_type",
        ),
        scopes=("task:task:read",),
        paginate=True,
    ),
    "search": ActionSpec(
        "POST",
        "/open-apis/task/v2/tasks/search",
        required=("query",),
        scopes=("task:task:read",),
        auth="user",
    ),
    "get": ActionSpec(
        "GET",
        "/open-apis/task/v2/tasks/:task_guid",
        required=("task_guid",),
        path_keys=("task_guid",),
        query_keys=("user_id_type",),
        scopes=("task:task:read",),
    ),
    "create": ActionSpec(
        "POST",
        "/open-apis/task/v2/tasks",
        required=("summary",),
        query_keys=("user_id_type",),
        scopes=("task:task:write",),
        write=True,
    ),
    "update": ActionSpec(
        "PATCH",
        "/open-apis/task/v2/tasks/:task_guid",
        required=("task_guid", "task", "update_fields"),
        path_keys=("task_guid",),
        query_keys=("user_id_type",),
        scopes=("task:task:write",),
        write=True,
        custom="update_task",
    ),
    "assign": ActionSpec(
        "PATCH",
        "/open-apis/task/v2/tasks/:task_guid",
        required=("task_guid", "members"),
        path_keys=("task_guid",),
        query_keys=("user_id_type",),
        scopes=("task:task:write",),
        write=True,
        custom="assign_task",
    ),
    "create_subtask": ActionSpec(
        "POST",
        "/open-apis/task/v2/tasks/:task_guid/subtasks",
        required=("task_guid", "summary"),
        path_keys=("task_guid",),
        query_keys=("user_id_type",),
        scopes=("task:task:write",),
        write=True,
    ),
    "list_tasklists": ActionSpec(
        "GET",
        "/open-apis/task/v2/tasklists",
        query_keys=("page_size",),
        scopes=("task:tasklist:read",),
        paginate=True,
    ),
    "create_tasklist": ActionSpec(
        "POST",
        "/open-apis/task/v2/tasklists",
        required=("name",),
        scopes=("task:tasklist:write",),
        write=True,
    ),
    "add_comment": ActionSpec(
        "POST",
        "/open-apis/task/v2/comments",
        required=("content", "resource_type", "resource_id"),
        scopes=("task:comment:write",),
        write=True,
    ),
    "list_attachments": ActionSpec(
        "GET",
        "/open-apis/task/v2/attachments",
        required=("resource_type", "resource_id"),
        query_keys=("resource_type", "resource_id", "page_size"),
        scopes=("task:attachment:read",),
        paginate=True,
    ),
    "upload_attachment": ActionSpec(
        "POST",
        "/open-apis/task/v2/attachments/upload",
        required=("file_path", "resource_type", "resource_id"),
        query_keys=("user_id_type",),
        scopes=("task:attachment:write",),
        write=True,
        custom="upload_task_attachment",
    ),
}

_BITABLE_ACTIONS = {
    "get_app": ActionSpec(
        "GET",
        "/open-apis/bitable/v1/apps/:app_token",
        required=("app_token",),
        path_keys=("app_token",),
        scopes=("base:app:read",),
    ),
    "list_tables": ActionSpec(
        "GET",
        "/open-apis/bitable/v1/apps/:app_token/tables",
        required=("app_token",),
        path_keys=("app_token",),
        query_keys=("page_size",),
        scopes=("base:table:read",),
        paginate=True,
    ),
    "list_fields": ActionSpec(
        "GET",
        "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/fields",
        required=("app_token", "table_id"),
        path_keys=("app_token", "table_id"),
        query_keys=("page_size",),
        scopes=("base:field:read",),
        paginate=True,
    ),
    "list_views": ActionSpec(
        "GET",
        "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/views",
        required=("app_token", "table_id"),
        path_keys=("app_token", "table_id"),
        query_keys=("page_size",),
        scopes=("base:view:read",),
        paginate=True,
    ),
    "query_records": ActionSpec(
        "POST",
        "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/records/search",
        required=("app_token", "table_id"),
        path_keys=("app_token", "table_id"),
        query_keys=("user_id_type", "page_size"),
        scopes=("base:record:retrieve",),
    ),
    "create_record": ActionSpec(
        "POST",
        "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/records",
        required=("app_token", "table_id", "fields"),
        path_keys=("app_token", "table_id"),
        query_keys=("user_id_type",),
        scopes=("base:record:create",),
        write=True,
    ),
    "update_record": ActionSpec(
        "PUT",
        "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/records/:record_id",
        required=("app_token", "table_id", "record_id", "fields"),
        path_keys=("app_token", "table_id", "record_id"),
        query_keys=("user_id_type",),
        scopes=("base:record:update",),
        write=True,
    ),
    "batch_create": ActionSpec(
        "POST",
        "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/records/batch_create",
        required=("app_token", "table_id", "records"),
        path_keys=("app_token", "table_id"),
        query_keys=("user_id_type",),
        scopes=("base:record:create",),
        write=True,
    ),
    "batch_update": ActionSpec(
        "POST",
        "/open-apis/bitable/v1/apps/:app_token/tables/:table_id/records/batch_update",
        required=("app_token", "table_id", "records"),
        path_keys=("app_token", "table_id"),
        query_keys=("user_id_type",),
        scopes=("base:record:update",),
        write=True,
    ),
}

_MEETING_ACTIONS = {
    "search": ActionSpec(
        "POST",
        "/open-apis/vc/v1/meetings/search",
        query_keys=("page_size", "page_token"),
        scopes=("vc:meeting:readonly",),
        paginate=True,
        auth="user",
    ),
    "get": ActionSpec(
        "GET",
        "/open-apis/vc/v1/meetings/:meeting_id",
        required=("meeting_id",),
        path_keys=("meeting_id",),
        scopes=("vc:meeting:readonly",),
    ),
    "participants": ActionSpec(
        "GET",
        "/open-apis/vc/v1/participant_list",
        required=("meeting_start_time", "meeting_end_time"),
        query_keys=(
            "meeting_start_time",
            "meeting_end_time",
            "meeting_status",
            "meeting_no",
            "user_id",
            "room_id",
            "page_size",
            "webinar_user_role",
            "user_id_type",
        ),
        scopes=("vc:meeting:readonly",),
        paginate=True,
    ),
    "search_minutes": ActionSpec(
        "POST",
        "/open-apis/minutes/v1/minutes/search",
        required=("query",),
        scopes=("minutes:minutes:readonly",),
        auth="user",
    ),
    "get_minute": ActionSpec(
        "GET",
        "/open-apis/minutes/v1/minutes/:minute_token",
        required=("minute_token",),
        path_keys=("minute_token",),
        scopes=("minutes:minutes:readonly",),
    ),
    "transcript": ActionSpec(
        "GET",
        "/open-apis/minutes/v1/minutes/:minute_token/transcript",
        required=("minute_token",),
        path_keys=("minute_token",),
        query_keys=("need_speaker",),
        scopes=("minutes:minutes:readonly",),
    ),
    "artifacts": ActionSpec(
        "GET",
        "/open-apis/minutes/v1/minutes/:minute_token/artifacts",
        required=("minute_token",),
        path_keys=("minute_token",),
        scopes=("minutes:minutes:readonly",),
    ),
    "action_items": ActionSpec(
        "GET",
        "/open-apis/minutes/v1/minutes/:minute_token/artifacts",
        required=("minute_token",),
        path_keys=("minute_token",),
        scopes=("minutes:minutes:readonly",),
        custom="action_items",
    ),
}

_TOOLS: dict[str, tuple[str, Mapping[str, ActionSpec]]] = {
    "lark_people": (
        "Resolve people and inspect profiles, departments, or group members.",
        _PEOPLE_ACTIONS,
    ),
    "lark_im": (
        "Read chats and messages or explicitly send, reply, mention, and DM.",
        _IM_ACTIONS,
    ),
    "lark_docs": (
        "Search, read, create, update, and comment on Feishu documents.",
        _DOC_ACTIONS,
    ),
    "lark_wiki": (
        "Search and manage Wiki spaces and document nodes.",
        _WIKI_ACTIONS,
    ),
    "lark_drive": (
        "List, upload, download, move, and share Drive files.",
        _DRIVE_ACTIONS,
    ),
    "lark_calendar": (
        "Inspect agendas and availability or manage events and attendees.",
        _ACTIVE_CALENDAR_ACTIONS,
    ),
    "lark_bitable": (
        "Inspect and update Bitable apps, tables, fields, views, and records.",
        _BITABLE_ACTIONS,
    ),
}

_DEFERRED_DOMAINS = {
    "calendar_rooms": "Deferred until meeting-room permissions are approved.",
    "lark_meetings": "Deferred for a later release.",
    "lark_tasks": "Deferred because the team does not use native Feishu Tasks.",
}

_ALL_REQUIRED_SCOPES = tuple(
    sorted(
        {
            scope
            for _, actions in _TOOLS.values()
            for spec in actions.values()
            for scope in spec.scopes
        }
    )
)


def _schema(name: str, description: str, actions: Mapping[str, ActionSpec]) -> dict:
    action_help = ", ".join(sorted(actions))
    parameter_help = "; ".join(
        f"{action}({', '.join(spec.required) if spec.required else 'no required params'})"
        for action, spec in sorted(actions.items())
    )
    return {
        "name": name,
        "description": f"{description} Supported actions: {action_help}.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": sorted(actions),
                    "description": "Domain action to execute.",
                },
                "params": {
                    "type": "object",
                    "description": (
                        "Action parameters. Use the identifiers and field names "
                        "returned by earlier read actions. Required parameters "
                        f"by action: {parameter_help}."
                    ),
                    "additionalProperties": True,
                },
                "idempotency_key": {
                    "type": "string",
                    "description": (
                        "Optional stable key for send/create/batch retries. "
                        "Hermes derives one from the current user request when omitted."
                    ),
                },
            },
            "required": ["action"],
        },
    }


def _mention_tag(user_id: str, name: str) -> str:
    safe_name = name or user_id
    return f'<at user_id="{user_id}">{safe_name}</at>'


def _message_text(params: Mapping[str, Any]) -> str:
    mentions = params.get("mentions")
    prefix = ""
    if isinstance(mentions, list):
        tags = []
        for mention in mentions:
            if not isinstance(mention, Mapping):
                continue
            user_id = str(mention.get("user_id") or "").strip()
            if user_id:
                tags.append(_mention_tag(user_id, str(mention.get("name") or "")))
        if tags:
            prefix = " ".join(tags) + " "
    return prefix + str(params.get("text") or "")


def _reaction_emoji_type(value: Any) -> str:
    raw = str(value or "").strip()
    normalized = _REACTION_ALIASES.get(raw, raw.upper())
    if normalized not in _SAFE_REACTION_EMOJIS:
        raise ValueError(
            "emoji must be one of: "
            + ", ".join(sorted(_SAFE_REACTION_EMOJIS))
        )
    return normalized


def _idempotency_key(
    tool_name: str,
    action: str,
    params: Mapping[str, Any],
    args: Mapping[str, Any],
    kwargs: Mapping[str, Any],
) -> str:
    supplied = str(args.get("idempotency_key") or "").strip()
    if supplied:
        return supplied
    seed = {
        "tool": tool_name,
        "action": action,
        "params": params,
        "session_id": kwargs.get("session_id", ""),
        "user_task": kwargs.get("user_task", ""),
    }
    encoded = json.dumps(seed, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _service_error(exc: LarkServiceError) -> str:
    details = exc.as_dict()
    message = str(details.pop("error", str(exc)))
    return tool_error(message, **details)


def _body(
    params: Mapping[str, Any],
    spec: ActionSpec,
    *,
    excluded: Sequence[str] = (),
) -> dict[str, Any]:
    explicit = params.get("body")
    if isinstance(explicit, Mapping):
        return dict(explicit)
    ignored = {
        *spec.path_keys,
        *spec.query_keys,
        "body",
        "max_pages",
        *excluded,
    }
    return {key: value for key, value in params.items() if key not in ignored}


def _search_message_items(
    data: Any,
    *,
    query: str,
    limit: int,
) -> dict[str, Any]:
    """Filter a bounded chat/thread history response without another API."""

    source = data if isinstance(data, Mapping) else {}
    items = source.get("items")
    if not isinstance(items, list):
        items = []
    needle = query.casefold()
    matches = [
        item
        for item in items
        if needle
        in json.dumps(item, ensure_ascii=False, default=str).casefold()
    ]
    return {
        "items": matches[: max(1, min(limit, 100))],
        "matched": len(matches),
        "scanned": len(items),
    }


def _parse_instant(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("time values must not be empty")
    if text.isdigit():
        return datetime.fromtimestamp(int(text), tz=timezone.utc)
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("time values must include a timezone")
    return parsed


def _recommended_slots(
    data: Any,
    *,
    time_min: Any,
    time_max: Any,
    duration_minutes: Any,
    max_suggestions: Any,
) -> dict[str, Any]:
    """Calculate common free slots from Lark's batch free/busy response."""

    window_start = _parse_instant(time_min)
    window_end = _parse_instant(time_max)
    if window_end <= window_start:
        raise ValueError("time_max must be later than time_min")
    duration = timedelta(minutes=max(1, int(duration_minutes)))
    suggestion_limit = max(1, min(int(max_suggestions or 5), 20))

    payload = data if isinstance(data, Mapping) else {}
    users = payload.get("freebusy_lists")
    if not isinstance(users, list):
        users = []
    busy: list[tuple[datetime, datetime]] = []
    for user in users:
        if not isinstance(user, Mapping):
            continue
        items = user.get("freebusy_items")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            start = max(window_start, _parse_instant(item.get("start_time")))
            end = min(window_end, _parse_instant(item.get("end_time")))
            if end > start:
                busy.append((start, end))

    merged: list[list[datetime]] = []
    for start, end in sorted(busy):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    suggestions: list[dict[str, str]] = []
    cursor = window_start
    for start, end in [*merged, [window_end, window_end]]:
        while cursor + duration <= start and len(suggestions) < suggestion_limit:
            slot_end = cursor + duration
            suggestions.append(
                {
                    "start_time": cursor.isoformat(),
                    "end_time": slot_end.isoformat(),
                }
            )
            cursor = slot_end
        cursor = max(cursor, end)
        if len(suggestions) >= suggestion_limit:
            break

    return {
        "suggestions": suggestions,
        "duration_minutes": int(duration.total_seconds() // 60),
        "busy_interval_count": len(merged),
    }


def _execute(
    tool_name: str,
    actions: Mapping[str, ActionSpec],
    args: dict,
    kwargs: Mapping[str, Any],
) -> str:
    action = str(args.get("action") or "").strip()
    spec = actions.get(action)
    if spec is None:
        return tool_error(
            f"Unsupported action for {tool_name}: {action or '<empty>'}",
            supported_actions=sorted(actions),
        )
    raw_params = args.get("params") or {}
    if not isinstance(raw_params, Mapping):
        return tool_error("params must be an object")
    params = dict(raw_params)
    missing = [key for key in spec.required if params.get(key) in (None, "", [])]
    if spec.custom == "resolve_users" and not (
        params.get("emails") or params.get("mobiles")
    ):
        missing.append("emails or mobiles")
    if missing:
        return tool_error(
            f"Missing required parameters for {action}: {', '.join(missing)}"
        )

    dedup_key = _idempotency_key(tool_name, action, params, args, kwargs)
    if spec.write and spec.approval_required:
        target = next(
            (
                str(params[key])
                for key in (
                    "receive_id",
                    "message_id",
                    "document_id",
                    "file_token",
                    "calendar_id",
                    "task_guid",
                    "app_token",
                    "space_id",
                )
                if params.get(key)
            ),
            "Lark",
        )
        approval = request_tool_approval(
            tool_name,
            f"Approve Lark action '{action}' for {target}.",
            rule_key=f"{tool_name}:{action}",
        )
        if not approval.get("approved"):
            return tool_error(
                approval.get("message") or "Lark write was not approved",
                blocked=True,
            )

    try:
        service = LarkService.from_environment()
        if spec.custom == "upload":
            result = service.upload_file(
                file_path=str(params["file_path"]),
                parent_type=str(params["parent_type"]),
                parent_node=str(params["parent_node"]),
                idempotency_key=dedup_key,
            )
        elif spec.custom == "upload_task_attachment":
            result = service.upload_task_attachment(
                file_path=str(params["file_path"]),
                resource_type=str(params["resource_type"]),
                resource_id=str(params["resource_id"]),
                user_id_type=str(params.get("user_id_type") or ""),
                idempotency_key=dedup_key,
            )
        elif spec.custom == "download":
            result = service.download_file(
                file_token=str(params["file_token"]),
                destination=str(params["destination"]),
                version=str(params.get("version") or ""),
            )
        elif spec.custom == "react":
            result = service.add_managed_reaction(
                message_id=str(params["message_id"]),
                emoji_type=_reaction_emoji_type(params["emoji"]),
                scopes=spec.scopes,
                idempotency_key=dedup_key,
            )
        elif spec.custom == "remove_reaction":
            result = service.remove_managed_reaction(
                message_id=str(params["message_id"]),
                scopes=spec.scopes,
            )
        else:
            paths = {key: params[key] for key in spec.path_keys}
            queries = {
                key: params[key]
                for key in spec.query_keys
                if params.get(key) not in (None, "")
            }
            excluded: tuple[str, ...] = ()
            if spec.custom == "thread_context":
                queries["container_id_type"] = "thread"
                queries["container_id"] = params["thread_id"]
                excluded = ("thread_id",)
            elif spec.custom == "search_messages":
                excluded = ("query", "limit")
            elif spec.custom == "recommend_times":
                excluded = ("duration_minutes", "max_suggestions")
            if spec.custom in {"send", "reply"}:
                body = {
                    "msg_type": str(params.get("msg_type") or "text"),
                    "content": json.dumps(
                        {"text": _message_text(params)},
                        ensure_ascii=False,
                    ),
                }
                if spec.custom == "send":
                    body["receive_id"] = str(params["receive_id"])
                    queries.setdefault(
                        "receive_id_type",
                        str(params.get("receive_id_type") or "open_id"),
                    )
            elif spec.custom == "assign_task":
                body = {
                    "task": {"members": params["members"]},
                    "update_fields": ["members"],
                }
            elif spec.custom == "unwrap_update":
                update = params["update"]
                if not isinstance(update, Mapping):
                    raise TypeError("update must be an object")
                body = dict(update)
            elif spec.custom == "update_task":
                task = params["task"]
                update_fields = params["update_fields"]
                if not isinstance(task, Mapping):
                    raise TypeError("task must be an object")
                if not isinstance(update_fields, list) or not all(
                    isinstance(field, str) and field for field in update_fields
                ):
                    raise TypeError("update_fields must be a non-empty string list")
                body = {
                    "task": dict(task),
                    "update_fields": list(update_fields),
                }
            else:
                body = _body(params, spec, excluded=excluded)

            request_kwargs = {
                "paths": paths,
                "queries": queries,
                "body": body or None,
                "scopes": spec.scopes,
                "auth": spec.auth,
            }
            if spec.paginate:
                result = service.request_pages(
                    spec.method,
                    spec.uri,
                    max_pages=int(params.get("max_pages") or 10),
                    **request_kwargs,
                )
            else:
                result = service.request(
                    spec.method,
                    spec.uri,
                    idempotency_key=dedup_key if spec.write else "",
                    retries=0 if spec.write else 2,
                    **request_kwargs,
                )
    except LarkServiceError as exc:
        return _service_error(exc)
    except (TypeError, ValueError) as exc:
        return tool_error(f"Invalid parameters for {action}: {exc}")

    payload = result.data
    if spec.custom == "search_messages":
        payload = _search_message_items(
            payload,
            query=str(params["query"]),
            limit=int(params.get("limit") or 20),
        )
    elif spec.custom == "recommend_times":
        payload = _recommended_slots(
            payload,
            time_min=params["time_min"],
            time_max=params["time_max"],
            duration_minutes=params["duration_minutes"],
            max_suggestions=params.get("max_suggestions"),
        )
    elif spec.custom == "action_items":
        source = payload if isinstance(payload, Mapping) else {}
        payload = {"action_items": source.get("minute_todos") or []}
    if tool_name == "lark_people" and isinstance(payload, dict):
        user = payload.get("user")
        if isinstance(user, dict):
            user_id = str(user.get("open_id") or user.get("user_id") or "")
            if user_id:
                user["mention_tag"] = _mention_tag(
                    user_id,
                    str(user.get("name") or ""),
                )
    return tool_result(
        {
            "success": True,
            "action": action,
            "data": payload,
            "request_id": result.request_id or None,
        }
    )


def _make_handler(
    name: str,
    actions: Mapping[str, ActionSpec],
):
    def handler(args: dict, **kwargs) -> str:
        return _execute(name, actions, args, kwargs)

    return handler


def _permission_handler(args: dict, **kwargs) -> str:
    del args, kwargs
    try:
        service = LarkService.from_environment()
        audit = service.scope_audit(_ALL_REQUIRED_SCOPES)
        authentication = service.auth_audit()
    except LarkServiceError as exc:
        return _service_error(exc)
    return tool_result(
        success=True,
        note=(
            "Available and missing states are based on requests made by this "
            "Hermes process. Unverified scopes have not been exercised yet."
        ),
        authentication=authentication,
        active_tools=sorted(_TOOLS),
        deferred_domains=_DEFERRED_DOMAINS,
        **audit,
    )


for _name, (_description, _actions) in _TOOLS.items():
    registry.register(
        name=_name,
        toolset="lark",
        schema=_schema(_name, _description, _actions),
        handler=_make_handler(_name, _actions),
        check_fn=lark_sdk_available,
        requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"],
        is_async=False,
        description=_description,
        emoji="\U0001f426",
        max_result_size_chars=50_000,
    )

registry.register(
    name="lark_permissions",
    toolset="lark",
    schema={
        "name": "lark_permissions",
        "description": (
            "Audit Lark app scopes observed by first-party tools without "
            "exposing app credentials."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    handler=_permission_handler,
    check_fn=lark_sdk_available,
    requires_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"],
    is_async=False,
    description="Audit Lark app permissions",
    emoji="\U0001f512",
)
