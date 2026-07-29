from pathlib import Path

from gateway.deployment_guard import blocked_gateway_tool_message


def _enable(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_DEPLOYMENT_GUARD", "true")


def test_gateway_blocks_active_checkout_file_edits(monkeypatch):
    _enable(monkeypatch)
    checkout = Path(__file__).resolve().parents[2]

    message = blocked_gateway_tool_message(
        platform="feishu",
        function_name="write_file",
        function_args={"path": str(checkout / "SOUL.md"), "content": "replace"},
    )

    assert message is not None
    assert "Contact the Hermes maintainer" in message


def test_gateway_blocks_lark_transfer_of_protected_files(monkeypatch):
    _enable(monkeypatch)
    checkout = Path(__file__).resolve().parents[2]

    for function_name, action, path_key in (
        ("lark_drive", "upload", "file_path"),
        ("lark_drive", "download", "destination"),
        ("lark_tasks", "upload_attachment", "file_path"),
    ):
        message = blocked_gateway_tool_message(
            platform="feishu",
            function_name=function_name,
            function_args={
                "action": action,
                "params": {path_key: str(checkout / "SOUL.md")},
            },
        )
        assert message is not None


def test_gateway_blocks_hermes_lifecycle_commands(monkeypatch):
    _enable(monkeypatch)

    for command in (
        "sudo systemctl restart hermes-gateway.service",
        "systemctl restart --no-block hermes-gateway.service",
        "service hermes-gateway restart",
        "kill -TERM $(pgrep -f hermes-gateway)",
    ):
        message = blocked_gateway_tool_message(
            platform="feishu",
            function_name="terminal",
            function_args={"command": command},
        )
        assert message is not None


def test_cli_operator_remains_unrestricted(monkeypatch):
    _enable(monkeypatch)
    checkout = Path(__file__).resolve().parents[2]

    assert (
        blocked_gateway_tool_message(
            platform="cli",
            function_name="patch",
            function_args={"path": str(checkout / "SOUL.md"), "patch": "..."},
        )
        is None
    )


def test_gateway_can_modify_unrelated_projects(monkeypatch, tmp_path):
    _enable(monkeypatch)
    project = tmp_path / "project"
    project.mkdir()

    assert (
        blocked_gateway_tool_message(
            platform="feishu",
            function_name="write_file",
            function_args={"path": str(project / "README.md"), "content": "ok"},
        )
        is None
    )
    assert (
        blocked_gateway_tool_message(
            platform="feishu",
            function_name="terminal",
            function_args={"command": "systemctl restart project-worker.service"},
        )
        is None
    )
    assert (
        blocked_gateway_tool_message(
            platform="feishu",
            function_name="terminal",
            function_args={
                "command": "touch result.txt",
                "workdir": str(project),
            },
        )
        is None
    )


def test_gateway_can_inspect_active_checkout(monkeypatch):
    _enable(monkeypatch)
    checkout = Path(__file__).resolve().parents[2]

    for command in ("git status --short", "git diff --stat", "uv run pytest --help"):
        assert (
            blocked_gateway_tool_message(
                platform="feishu",
                function_name="terminal",
                function_args={"command": command, "workdir": str(checkout)},
            )
            is None
        )


def test_gateway_blocks_active_checkout_mutation(monkeypatch):
    _enable(monkeypatch)
    checkout = Path(__file__).resolve().parents[2]

    for command in (
        "git pull --ff-only",
        "git switch main",
        "uv sync",
        "pip install example",
        "touch changed.txt",
    ):
        assert (
            blocked_gateway_tool_message(
                platform="feishu",
                function_name="terminal",
                function_args={"command": command, "workdir": str(checkout)},
            )
            is not None
        )


def test_disabled_guard_is_behavior_neutral(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_DEPLOYMENT_GUARD", "false")

    assert (
        blocked_gateway_tool_message(
            platform="feishu",
            function_name="terminal",
            function_args={"command": "hermes gateway restart"},
        )
        is None
    )
