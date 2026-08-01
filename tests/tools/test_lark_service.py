from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import pytest

from tools.lark_service import (
    _api_timeout_seconds,
    LarkApiResult,
    LarkService,
    LarkServiceError,
)


class _FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def request(self, request, option=None):
        self.requests.append((request, option))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _response(*, code=0, data=None, msg="", status_code=200, request_id="req-1"):
    body = {"code": code, "msg": msg, "data": data or {}, "request_id": request_id}
    import json

    return SimpleNamespace(
        code=code,
        msg=msg,
        request_id=request_id,
        raw=SimpleNamespace(
            status_code=status_code,
            content=json.dumps(body).encode(),
        ),
    )


@pytest.fixture
def service(monkeypatch):
    instance = LarkService("app", "secret", "feishu")
    monkeypatch.setattr(instance, "_build_request", lambda *args, **kwargs: kwargs)
    return instance


def test_from_environment_reuses_sdk_client_wrapper(monkeypatch):
    LarkService.reset_for_tests()
    monkeypatch.setenv("FEISHU_APP_ID", "app")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret")
    monkeypatch.setenv("FEISHU_DOMAIN", "feishu")

    first = LarkService.from_environment()
    second = LarkService.from_environment()

    assert first is second


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("", 15.0),
        ("0.1", 1.0),
        ("7.5", 7.5),
        ("999", 120.0),
        ("invalid", 15.0),
    ],
)
def test_api_timeout_is_bounded(monkeypatch, configured, expected):
    if configured:
        monkeypatch.setenv("LARK_API_TIMEOUT_SECONDS", configured)
    else:
        monkeypatch.delenv("LARK_API_TIMEOUT_SECONDS", raising=False)

    assert _api_timeout_seconds() == expected


def test_request_retries_rate_limit_and_records_scope(service, monkeypatch):
    client = _FakeClient(
        [
            _response(code=90013, msg="request too frequent", status_code=429),
            _response(data={"items": [1]}),
        ]
    )
    monkeypatch.setattr(service, "_get_client", lambda: client)
    monkeypatch.setattr("tools.lark_service.time.sleep", lambda _: None)

    result = service.request(
        "GET",
        "/open-apis/test",
        scopes=("scope:read",),
    )

    assert result.data == {"items": [1]}
    assert len(client.requests) == 2
    assert service.scope_audit(("scope:read",))["available"] == ["scope:read"]


def test_request_reports_exact_missing_scope(service, monkeypatch):
    client = _FakeClient(
        [
            _response(
                code=99991663,
                msg="permission denied: contact:user.base:readonly",
                status_code=403,
            )
        ]
    )
    monkeypatch.setattr(service, "_get_client", lambda: client)

    with pytest.raises(LarkServiceError) as caught:
        service.request(
            "GET",
            "/open-apis/test",
            scopes=("contact:user.base:readonly",),
        )

    assert caught.value.missing_scopes == ("contact:user.base:readonly",)
    assert service.scope_audit(("contact:user.base:readonly",))["missing"] == [
        "contact:user.base:readonly"
    ]


def test_request_idempotency_returns_cached_result(service, monkeypatch):
    client = _FakeClient([_response(data={"message_id": "om_1"})])
    monkeypatch.setattr(service, "_get_client", lambda: client)

    first = service.request(
        "POST",
        "/open-apis/test",
        idempotency_key="stable",
    )
    second = service.request(
        "POST",
        "/open-apis/test",
        idempotency_key="stable",
    )

    assert first == second
    assert len(client.requests) == 1


def test_managed_reaction_is_idempotent_and_removes_only_owned_id(
    service,
    monkeypatch,
):
    calls = []

    def request(method, uri, **kwargs):
        calls.append((method, uri, kwargs))
        if method == "POST":
            return LarkApiResult({"reaction_id": "reaction_1"}, "req-add")
        return LarkApiResult({}, "req-remove")

    monkeypatch.setattr(service, "request", request)

    first = service.add_managed_reaction(
        message_id="om_1",
        emoji_type="THUMBSUP",
        scopes=("im:message.reactions:write",),
        idempotency_key="stable",
    )
    second = service.add_managed_reaction(
        message_id="om_1",
        emoji_type="THUMBSUP",
        scopes=("im:message.reactions:write",),
        idempotency_key="stable",
    )

    assert first.data == {"reaction_id": "reaction_1"}
    assert second.data["already_present"] is True
    assert len(calls) == 1

    removed = service.remove_managed_reaction(
        message_id="om_1",
        scopes=("im:message.reactions:write",),
    )
    assert removed.data["removed"] is True
    assert calls[-1][2]["paths"]["reaction_id"] == "reaction_1"

    missing = service.remove_managed_reaction(
        message_id="om_1",
        scopes=("im:message.reactions:write",),
    )
    assert missing.data["reason"] == "no_managed_reaction"
    assert len(calls) == 2


def test_managed_reaction_refuses_replacement(service, monkeypatch):
    monkeypatch.setattr(
        service,
        "request",
        lambda *args, **kwargs: LarkApiResult(
            {"reaction_id": "reaction_1"},
            "req-add",
        ),
    )
    service.add_managed_reaction(
        message_id="om_1",
        emoji_type="THUMBSUP",
        scopes=("im:message.reactions:write",),
        idempotency_key="stable",
    )

    with pytest.raises(LarkServiceError, match="already has"):
        service.add_managed_reaction(
            message_id="om_1",
            emoji_type="HEART",
            scopes=("im:message.reactions:write",),
            idempotency_key="different",
        )


def test_user_auth_requires_explicit_user_oauth_token(service, monkeypatch):
    monkeypatch.delenv("FEISHU_USER_ACCESS_TOKEN", raising=False)

    with pytest.raises(LarkServiceError, match="requires user OAuth"):
        service.request(
            "POST",
            "/open-apis/minutes/v1/minutes/search",
            auth="user",
        )


def test_user_auth_passes_user_token_request_option(service, monkeypatch):
    client = _FakeClient([_response(data={"items": []})])
    monkeypatch.setattr(service, "_get_client", lambda: client)
    monkeypatch.setenv("FEISHU_USER_ACCESS_TOKEN", "user-token")

    result = service.request(
        "POST",
        "/open-apis/minutes/v1/minutes/search",
        auth="user",
    )

    assert result.data == {"items": []}
    request, option = client.requests[0]
    assert request["auth"] == "user"
    assert option.user_access_token == "user-token"


def test_tenant_auth_does_not_require_or_pass_user_token(service, monkeypatch):
    client = _FakeClient([_response(data={"ok": True})])
    monkeypatch.setattr(service, "_get_client", lambda: client)
    monkeypatch.delenv("FEISHU_USER_ACCESS_TOKEN", raising=False)

    result = service.request("GET", "/open-apis/test")

    assert result.data == {"ok": True}
    request, option = client.requests[0]
    assert request["auth"] == "tenant"
    assert option is None


def test_request_pages_is_bounded_and_merges_items(service, monkeypatch):
    responses = iter(
        [
            LarkApiResult(
                {"items": [{"id": "1"}], "has_more": True, "page_token": "next"},
                "req-1",
            ),
            LarkApiResult(
                {"items": [{"id": "2"}], "has_more": False},
                "req-2",
            ),
        ]
    )
    monkeypatch.setattr(service, "request", lambda *args, **kwargs: next(responses))

    result = service.request_pages("GET", "/open-apis/test", max_pages=2)

    assert result.data == {
        "items": [{"id": "1"}, {"id": "2"}],
        "has_more": False,
    }
    assert result.request_id == "req-2"


class _FakeFileResource:
    def __init__(self, *, upload_response=None, download_response=None):
        self.upload_response = upload_response
        self.download_response = download_response
        self.upload_requests = []
        self.download_requests = []

    def upload_all(self, request):
        self.upload_requests.append(request)
        return self.upload_response

    def download(self, request):
        self.download_requests.append(request)
        return self.download_response


def _client_with_file_resource(resource):
    return SimpleNamespace(
        drive=SimpleNamespace(
            v1=SimpleNamespace(
                file=resource,
            )
        )
    )


def test_upload_file_uses_generated_drive_file_resource(
    service,
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "report.txt"
    source.write_bytes(b"team-v3")
    resource = _FakeFileResource(
        upload_response=SimpleNamespace(
            code=0,
            data={"file_token": "file_1"},
            request_id="req-upload",
            raw=None,
        )
    )
    monkeypatch.setattr(
        service,
        "_get_client",
        lambda: _client_with_file_resource(resource),
    )

    result = service.upload_file(
        file_path=str(source),
        parent_type="explorer",
        parent_node="fld_1",
        idempotency_key="upload-1",
    )

    assert result.data == {"file_token": "file_1"}
    assert result.request_id == "req-upload"
    assert len(resource.upload_requests) == 1
    request = resource.upload_requests[0]
    assert request.uri == "/open-apis/drive/v1/files/upload_all"
    assert request.request_body.file_name == "report.txt"
    assert request.request_body.parent_type == "explorer"
    assert request.request_body.parent_node == "fld_1"
    assert request.request_body.size == len(b"team-v3")
    assert service.scope_audit(("drive:file:upload",))["available"] == [
        "drive:file:upload"
    ]


def test_download_file_uses_generated_drive_file_resource(
    service,
    monkeypatch,
    tmp_path,
):
    destination = tmp_path / "downloaded.txt"
    resource = _FakeFileResource(
        download_response=SimpleNamespace(
            code=0,
            file=BytesIO(b"downloaded"),
            file_name="remote.txt",
        )
    )
    monkeypatch.setattr(
        service,
        "_get_client",
        lambda: _client_with_file_resource(resource),
    )

    result = service.download_file(
        file_token="file_1",
        destination=str(destination),
        version="42",
    )

    assert destination.read_bytes() == b"downloaded"
    assert result.data == {
        "path": str(destination),
        "file_name": "remote.txt",
        "size": len(b"downloaded"),
    }
    assert len(resource.download_requests) == 1
    request = resource.download_requests[0]
    assert request.uri == "/open-apis/drive/v1/files/:file_token/download"
    assert request.file_token == "file_1"
    assert request.version == "42"
    assert service.scope_audit(("drive:file:download",))["available"] == [
        "drive:file:download"
    ]
