"""Shared authenticated Lark/Feishu OpenAPI service.

The gateway adapter and the model-callable Lark tools use the same app
credentials, but they have different lifetimes.  This module provides a lazy
process-wide SDK client for tool calls without importing ``lark_oapi`` during
normal Hermes startup (the SDK eagerly imports a large generated surface).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

_RETRYABLE_CODES = {
    90013,  # request too frequent
    99991400,  # request timeout
    99991401,  # system busy
    99991402,  # service unavailable
}
_PERMISSION_CODES = {
    99991663,
    99991668,
    99991672,
    99991679,
    99991680,
}
_MAX_RETRIES = 2
_DEDUP_TTL_SECONDS = 300.0
_DEFAULT_API_TIMEOUT_SECONDS = 15.0


def _api_timeout_seconds() -> float:
    """Return a bounded SDK request timeout from deployment configuration."""

    raw = os.getenv("LARK_API_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return _DEFAULT_API_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Ignoring invalid LARK_API_TIMEOUT_SECONDS=%r; using %.1fs",
            raw,
            _DEFAULT_API_TIMEOUT_SECONDS,
        )
        return _DEFAULT_API_TIMEOUT_SECONDS
    return max(1.0, min(value, 120.0))


@dataclass(frozen=True)
class LarkApiResult:
    """Normalized successful Lark response."""

    data: Any
    request_id: str = ""


class LarkServiceError(RuntimeError):
    """Normalized Lark API failure safe to return to the model."""

    def __init__(
        self,
        message: str,
        *,
        code: int | None = None,
        request_id: str = "",
        missing_scopes: Sequence[str] = (),
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.request_id = request_id
        self.missing_scopes = tuple(missing_scopes)
        self.retryable = retryable

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "error": str(self),
            "code": self.code,
            "retryable": self.retryable,
        }
        if self.request_id:
            result["request_id"] = self.request_id
        if self.missing_scopes:
            result["missing_scopes"] = list(self.missing_scopes)
            result["resolution"] = (
                "Add the listed scopes to the Feishu/Lark app, publish a new "
                "app version, and complete administrator approval or user "
                "OAuth consent as required by the action."
            )
        return result


def lark_sdk_available() -> bool:
    """Return whether the official SDK is importable without importing it."""

    try:
        return importlib.util.find_spec("lark_oapi") is not None
    except (ImportError, ValueError):
        return False


def _response_payload(response: Any) -> tuple[int | None, str, Any, str, int | None]:
    code = getattr(response, "code", None)
    msg = str(getattr(response, "msg", "") or "")
    request_id = str(
        getattr(response, "request_id", "")
        or getattr(response, "log_id", "")
        or ""
    )
    raw = getattr(response, "raw", None)
    status_code = getattr(raw, "status_code", None)
    payload: Any = None

    content = getattr(raw, "content", None)
    if content:
        try:
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            body = json.loads(content)
            code = body.get("code", code)
            msg = str(body.get("msg", msg) or msg)
            request_id = str(
                body.get("request_id")
                or body.get("requestId")
                or request_id
                or ""
            )
            payload = body.get("data", body)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
            payload = None

    if payload is None:
        data = getattr(response, "data", None)
        if isinstance(data, dict):
            payload = data
        elif data is not None:
            try:
                attributes = vars(data).items()
            except TypeError:
                attributes = (
                    (name, getattr(data, name))
                    for name in dir(data)
                    if not name.startswith("_")
                    and not callable(getattr(data, name, None))
                )
            payload = {
                key: value
                for key, value in attributes
                if not key.startswith("_")
            }
        else:
            payload = {}
    return code, msg, payload, request_id, status_code


def _extract_missing_scopes(message: str, expected_scopes: Sequence[str]) -> tuple[str, ...]:
    lower = message.lower()
    found = [scope for scope in expected_scopes if scope.lower() in lower]
    if found:
        return tuple(found)
    if any(token in lower for token in ("permission", "scope", "权限")):
        return tuple(expected_scopes)
    return ()


class LarkService:
    """Lazy, reusable Lark SDK client for tenant and user access tokens."""

    _instances: dict[tuple[str, str, str], "LarkService"] = {}
    _instances_lock = threading.Lock()

    def __init__(self, app_id: str, app_secret: str, domain_name: str) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._domain_name = domain_name
        self._client: Any = None
        self._client_lock = threading.Lock()
        self._dedup_lock = threading.Lock()
        self._dedup: dict[str, tuple[float, LarkApiResult]] = {}
        self._scope_lock = threading.Lock()
        self._scope_status: dict[str, str] = {}
        self._reaction_lock = threading.Lock()
        self._managed_reactions: dict[str, tuple[str, str]] = {}

    @classmethod
    def from_environment(cls) -> "LarkService":
        app_id = os.getenv("FEISHU_APP_ID", "").strip()
        app_secret = os.getenv("FEISHU_APP_SECRET", "").strip()
        if not app_id or not app_secret:
            raise LarkServiceError(
                "Feishu/Lark credentials are not configured. "
                "Set FEISHU_APP_ID and FEISHU_APP_SECRET."
            )
        domain_name = os.getenv("FEISHU_DOMAIN", "feishu").strip().lower()
        if domain_name not in {"feishu", "lark"}:
            raise LarkServiceError(
                "FEISHU_DOMAIN must be either 'feishu' or 'lark'."
            )
        key = (app_id, hashlib.sha256(app_secret.encode()).hexdigest(), domain_name)
        with cls._instances_lock:
            service = cls._instances.get(key)
            if service is None:
                service = cls(app_id, app_secret, domain_name)
                cls._instances[key] = service
            return service

    @classmethod
    def reset_for_tests(cls) -> None:
        with cls._instances_lock:
            cls._instances.clear()

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            if not lark_sdk_available():
                raise LarkServiceError(
                    "The official lark_oapi SDK is not installed."
                )
            import lark_oapi as lark
            from lark_oapi.core.const import FEISHU_DOMAIN, LARK_DOMAIN

            domain = LARK_DOMAIN if self._domain_name == "lark" else FEISHU_DOMAIN
            self._client = (
                lark.Client.builder()
                .app_id(self._app_id)
                .app_secret(self._app_secret)
                .domain(domain)
                .timeout(_api_timeout_seconds())
                .log_level(lark.LogLevel.WARNING)
                .enable_set_token(True)
                .build()
            )
            return self._client

    @staticmethod
    def _build_request(
        method: str,
        uri: str,
        *,
        paths: Mapping[str, Any] | None,
        queries: Mapping[str, Any] | Sequence[tuple[str, Any]] | None,
        body: Any,
        headers: Mapping[str, str] | None,
        auth: str,
    ) -> Any:
        from lark_oapi import AccessTokenType
        from lark_oapi.core.enum import HttpMethod
        from lark_oapi.core.model.base_request import BaseRequest

        try:
            http_method = getattr(HttpMethod, method.upper())
        except AttributeError as exc:
            raise LarkServiceError(f"Unsupported Lark HTTP method: {method}") from exc

        builder = (
            BaseRequest.builder()
            .http_method(http_method)
            .uri(uri)
            .token_types(
                {
                    AccessTokenType.USER
                    if auth == "user"
                    else AccessTokenType.TENANT
                }
            )
        )
        if paths:
            builder = builder.paths({str(k): str(v) for k, v in paths.items()})
        if queries:
            if isinstance(queries, Mapping):
                query_items: list[tuple[str, str]] = []
                for key, value in queries.items():
                    if value is None or value == "":
                        continue
                    values = value if isinstance(value, (list, tuple)) else [value]
                    query_items.extend((str(key), str(item)) for item in values)
            else:
                query_items = [
                    (str(key), str(value))
                    for key, value in queries
                    if value is not None and value != ""
                ]
            builder = builder.queries(query_items)
        if headers:
            builder = builder.headers(dict(headers))
        if body is not None:
            builder = builder.body(body)
        return builder.build()

    def _record_scope_status(self, scopes: Sequence[str], status: str) -> None:
        if not scopes:
            return
        with self._scope_lock:
            for scope in scopes:
                self._scope_status[scope] = status

    def scope_audit(self, required_scopes: Iterable[str]) -> dict[str, list[str]]:
        """Report evidence-backed scope state without exposing credentials."""

        with self._scope_lock:
            status = dict(self._scope_status)
        required = sorted(set(required_scopes))
        return {
            "available": [scope for scope in required if status.get(scope) == "available"],
            "missing": [scope for scope in required if status.get(scope) == "missing"],
            "unverified": [scope for scope in required if scope not in status],
        }

    @staticmethod
    def auth_audit() -> dict[str, str]:
        """Report configured authentication shapes without exposing tokens."""

        return {
            "tenant_token": "configured",
            "user_token": (
                "configured"
                if os.getenv("FEISHU_USER_ACCESS_TOKEN", "").strip()
                else "missing"
            ),
        }

    def _dedup_get(self, key: str) -> LarkApiResult | None:
        if not key:
            return None
        now = time.monotonic()
        with self._dedup_lock:
            expired = [
                item_key
                for item_key, (expires_at, _) in self._dedup.items()
                if expires_at <= now
            ]
            for item_key in expired:
                self._dedup.pop(item_key, None)
            cached = self._dedup.get(key)
            return cached[1] if cached else None

    def _dedup_put(self, key: str, result: LarkApiResult) -> None:
        if not key:
            return
        with self._dedup_lock:
            self._dedup[key] = (time.monotonic() + _DEDUP_TTL_SECONDS, result)

    def add_managed_reaction(
        self,
        *,
        message_id: str,
        emoji_type: str,
        scopes: Sequence[str],
        idempotency_key: str,
    ) -> LarkApiResult:
        """Add one Hermes-owned reaction to a message.

        The opaque reaction ID is retained in memory so removal can never
        target a reaction created by another user or application. Losing this
        handle on restart is intentionally fail-safe: Hermes can no longer
        remove the old reaction, but it also cannot remove the wrong one.
        """

        with self._reaction_lock:
            existing = self._managed_reactions.get(message_id)
            if existing is not None:
                reaction_id, existing_emoji = existing
                if existing_emoji == emoji_type:
                    return LarkApiResult(
                        {
                            "reaction_id": reaction_id,
                            "emoji_type": existing_emoji,
                            "already_present": True,
                        },
                        "",
                    )
                raise LarkServiceError(
                    "Hermes already has a managed reaction on this message. "
                    "Remove it before choosing another emoji."
                )

            result = self.request(
                "POST",
                "/open-apis/im/v1/messages/:message_id/reactions",
                paths={"message_id": message_id},
                body={"reaction_type": {"emoji_type": emoji_type}},
                scopes=scopes,
                idempotency_key=idempotency_key,
                retries=0,
            )
            payload = result.data if isinstance(result.data, Mapping) else {}
            reaction_id = str(payload.get("reaction_id") or "")
            if not reaction_id:
                raise LarkServiceError(
                    "Lark accepted the reaction request but returned no reaction ID."
                )
            self._managed_reactions[message_id] = (reaction_id, emoji_type)
            return result

    def remove_managed_reaction(
        self,
        *,
        message_id: str,
        scopes: Sequence[str],
    ) -> LarkApiResult:
        """Remove only the reaction previously created by this service."""

        with self._reaction_lock:
            existing = self._managed_reactions.get(message_id)
            if existing is None:
                return LarkApiResult(
                    {
                        "message_id": message_id,
                        "removed": False,
                        "reason": "no_managed_reaction",
                    },
                    "",
                )
            reaction_id, emoji_type = existing
            result = self.request(
                "DELETE",
                "/open-apis/im/v1/messages/:message_id/reactions/:reaction_id",
                paths={
                    "message_id": message_id,
                    "reaction_id": reaction_id,
                },
                scopes=scopes,
                retries=0,
            )
            self._managed_reactions.pop(message_id, None)
            payload = result.data if isinstance(result.data, Mapping) else {}
            return LarkApiResult(
                {
                    **dict(payload),
                    "message_id": message_id,
                    "reaction_id": reaction_id,
                    "emoji_type": emoji_type,
                    "removed": True,
                },
                result.request_id,
            )

    def request(
        self,
        method: str,
        uri: str,
        *,
        paths: Mapping[str, Any] | None = None,
        queries: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
        body: Any = None,
        headers: Mapping[str, str] | None = None,
        scopes: Sequence[str] = (),
        idempotency_key: str = "",
        retries: int = _MAX_RETRIES,
        auth: str = "tenant",
    ) -> LarkApiResult:
        """Execute a normalized tenant- or user-token OpenAPI request."""

        cached = self._dedup_get(idempotency_key)
        if cached is not None:
            return cached

        if auth not in {"tenant", "user"}:
            raise LarkServiceError(f"Unsupported Lark authentication mode: {auth}")

        option = None
        if auth == "user":
            user_access_token = os.getenv("FEISHU_USER_ACCESS_TOKEN", "").strip()
            if not user_access_token:
                raise LarkServiceError(
                    "This Lark action requires user OAuth. Configure "
                    "FEISHU_USER_ACCESS_TOKEN for the Hermes service account."
                )
            from lark_oapi.core.model.request_option import RequestOption

            option = (
                RequestOption.builder()
                .user_access_token(user_access_token)
                .build()
            )

        request = self._build_request(
            method,
            uri,
            paths=paths,
            queries=queries,
            body=body,
            headers=headers,
            auth=auth,
        )
        client = self._get_client()
        attempts = max(0, retries) + 1
        last_error: LarkServiceError | None = None

        for attempt in range(attempts):
            try:
                response = (
                    client.request(request, option)
                    if option is not None
                    else client.request(request)
                )
                code, msg, data, request_id, status_code = _response_payload(response)
            except Exception as exc:
                last_error = LarkServiceError(
                    f"Lark request failed: {type(exc).__name__}",
                    retryable=True,
                )
                if attempt + 1 < attempts:
                    time.sleep(0.25 * (2**attempt))
                    continue
                raise last_error from exc

            if code in (0, None) and (
                status_code is None or 200 <= int(status_code) < 300
            ):
                result = LarkApiResult(data=data, request_id=request_id)
                self._record_scope_status(scopes, "available")
                self._dedup_put(idempotency_key, result)
                return result

            missing_scopes = _extract_missing_scopes(msg, scopes)
            permission_failure = code in _PERMISSION_CODES or bool(missing_scopes)
            if permission_failure:
                self._record_scope_status(scopes, "missing")
            retryable = (
                code in _RETRYABLE_CODES
                or status_code == 429
                or (status_code is not None and int(status_code) >= 500)
            )
            last_error = LarkServiceError(
                f"Lark API request failed: {msg or 'unknown error'}",
                code=code,
                request_id=request_id,
                missing_scopes=missing_scopes,
                retryable=retryable,
            )
            if retryable and attempt + 1 < attempts:
                time.sleep(0.25 * (2**attempt))
                continue
            raise last_error

        raise last_error or LarkServiceError("Lark request failed")

    def request_pages(
        self,
        method: str,
        uri: str,
        *,
        paths: Mapping[str, Any] | None = None,
        queries: Mapping[str, Any] | None = None,
        body: Any = None,
        scopes: Sequence[str] = (),
        max_pages: int = 10,
        auth: str = "tenant",
    ) -> LarkApiResult:
        """Collect a bounded sequence of Lark page-token responses."""

        merged_items: list[Any] = []
        page_token = ""
        request_id = ""
        last_data: dict[str, Any] = {}
        base_queries = dict(queries or {})

        for _ in range(max(1, min(max_pages, 50))):
            page_queries = dict(base_queries)
            if page_token:
                page_queries["page_token"] = page_token
            result = self.request(
                method,
                uri,
                paths=paths,
                queries=page_queries,
                body=body,
                scopes=scopes,
                auth=auth,
            )
            request_id = result.request_id or request_id
            data = result.data if isinstance(result.data, dict) else {}
            last_data = data
            items = data.get("items")
            if isinstance(items, list):
                merged_items.extend(items)
            page_token = str(data.get("page_token") or "")
            if not data.get("has_more") or not page_token:
                break

        output = dict(last_data)
        if merged_items:
            output["items"] = merged_items
        output.pop("page_token", None)
        output["has_more"] = False
        return LarkApiResult(data=output, request_id=request_id)

    def upload_file(
        self,
        *,
        file_path: str,
        parent_type: str,
        parent_node: str,
        idempotency_key: str = "",
    ) -> LarkApiResult:
        """Upload a local file through the generated SDK multipart API."""

        path = Path(file_path).expanduser().resolve()
        if not path.is_file():
            raise LarkServiceError(f"Upload file does not exist: {path}")
        cached = self._dedup_get(idempotency_key)
        if cached is not None:
            return cached

        from lark_oapi.api.drive.v1 import (
            UploadAllFileRequest,
            UploadAllFileRequestBody,
        )

        scopes = ("drive:file:upload",)
        with path.open("rb") as file_obj:
            body = (
                UploadAllFileRequestBody.builder()
                .file_name(path.name)
                .parent_type(parent_type)
                .parent_node(parent_node)
                .size(path.stat().st_size)
                .file(file_obj)
                .build()
            )
            request = UploadAllFileRequest.builder().request_body(body).build()
            response = self._get_client().drive.v1.file.upload_all(request)
        code, msg, data, request_id, _ = _response_payload(response)
        if code != 0:
            missing = _extract_missing_scopes(msg, scopes)
            if missing:
                self._record_scope_status(scopes, "missing")
            raise LarkServiceError(
                f"Lark upload failed: {msg or 'unknown error'}",
                code=code,
                request_id=request_id,
                missing_scopes=missing,
            )
        self._record_scope_status(scopes, "available")
        result = LarkApiResult(data=data, request_id=request_id)
        self._dedup_put(idempotency_key, result)
        return result

    def upload_task_attachment(
        self,
        *,
        file_path: str,
        resource_type: str,
        resource_id: str,
        user_id_type: str = "",
        idempotency_key: str = "",
    ) -> LarkApiResult:
        """Upload an attachment to a task resource through the SDK."""

        path = Path(file_path).expanduser().resolve()
        if not path.is_file():
            raise LarkServiceError(f"Attachment file does not exist: {path}")
        cached = self._dedup_get(idempotency_key)
        if cached is not None:
            return cached

        from lark_oapi.api.task.v2 import (
            InputAttachment,
            UploadAttachmentRequest,
        )

        with path.open("rb") as file_obj:
            body = (
                InputAttachment.builder()
                .resource_type(resource_type)
                .resource_id(resource_id)
                .file(file_obj)
                .build()
            )
            builder = UploadAttachmentRequest.builder().request_body(body)
            if user_id_type:
                builder = builder.user_id_type(user_id_type)
            response = self._get_client().task.v2.attachment.upload(
                builder.build()
            )

        code, msg, data, request_id, _ = _response_payload(response)
        scopes = ("task:attachment:write",)
        if code != 0:
            missing = _extract_missing_scopes(msg, scopes)
            if missing:
                self._record_scope_status(scopes, "missing")
            raise LarkServiceError(
                f"Lark task attachment upload failed: "
                f"{msg or 'unknown error'}",
                code=code,
                request_id=request_id,
                missing_scopes=missing,
            )
        self._record_scope_status(scopes, "available")
        result = LarkApiResult(data=data, request_id=request_id)
        self._dedup_put(idempotency_key, result)
        return result

    def download_file(
        self,
        *,
        file_token: str,
        destination: str,
        version: str = "",
    ) -> LarkApiResult:
        """Download a Drive file to an explicit local destination."""

        from lark_oapi.api.drive.v1 import DownloadFileRequest

        destination_path = Path(destination).expanduser().resolve()
        if not destination_path.parent.is_dir():
            raise LarkServiceError(
                f"Download destination directory does not exist: "
                f"{destination_path.parent}"
            )
        builder = DownloadFileRequest.builder().file_token(file_token)
        if version:
            builder = builder.version(version)
        response = self._get_client().drive.v1.file.download(builder.build())
        code = getattr(response, "code", None)
        if code != 0:
            _, msg, _, request_id, _ = _response_payload(response)
            scopes = ("drive:file:download",)
            missing = _extract_missing_scopes(msg, scopes)
            if missing:
                self._record_scope_status(scopes, "missing")
            raise LarkServiceError(
                f"Lark download failed: {msg or 'unknown error'}",
                code=code,
                request_id=request_id,
                missing_scopes=missing,
            )
        file_obj = getattr(response, "file", None)
        if file_obj is None:
            raise LarkServiceError("Lark download returned no file content")
        destination_path.write_bytes(file_obj.read())
        self._record_scope_status(("drive:file:download",), "available")
        return LarkApiResult(
            data={
                "path": str(destination_path),
                "file_name": getattr(response, "file_name", destination_path.name),
                "size": destination_path.stat().st_size,
            }
        )
