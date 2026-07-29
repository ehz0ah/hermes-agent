"""Gateway-only protection for the running Hermes deployment.

The gateway is allowed to work on user projects, but it must not rewrite or
restart the process that is currently serving other users. Operators using the
CLI remain unrestricted.
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Any, Mapping

from hermes_constants import get_hermes_home
from utils import is_truthy_value

_BLOCK_MESSAGE = (
    "This deployment change is restricted. Contact the Hermes maintainer "
    "for feature or deployment updates."
)
_ALWAYS_MUTATING_COMMANDS = frozenset(
    {
        "chmod",
        "chown",
        "cp",
        "install",
        "ln",
        "mv",
        "patch",
        "perl",
        "rm",
        "rsync",
        "sed",
        "tee",
        "touch",
        "truncate",
    }
)
_GIT_MUTATING_SUBCOMMANDS = frozenset(
    {
        "add",
        "am",
        "apply",
        "bisect",
        "branch",
        "checkout",
        "cherry-pick",
        "clean",
        "clone",
        "commit",
        "fetch",
        "gc",
        "init",
        "merge",
        "mv",
        "pull",
        "push",
        "rebase",
        "reset",
        "restore",
        "revert",
        "rm",
        "stash",
        "submodule",
        "switch",
        "tag",
        "worktree",
    }
)
_PIP_MUTATING_SUBCOMMANDS = frozenset(
    {"download", "install", "uninstall", "wheel"}
)
_UV_MUTATING_SUBCOMMANDS = frozenset(
    {"add", "build", "init", "lock", "publish", "remove", "self", "sync", "tool"}
)
_SCRIPT_COMMANDS = frozenset(
    {"bash", "node", "perl", "python", "python3", "ruby", "sh", "zsh"}
)
_HERMES_CLI_LIFECYCLE_RE = re.compile(
    r"\bhermes\s+(?:gateway\s+)?"
    r"(?:install|restart|start|stop|uninstall|update)\b",
    re.IGNORECASE,
)
_SERVICE_LIFECYCLE_RE = re.compile(
    r"\bsystemctl\b"
    r"[^;&|\n]*\b(?:disable|enable|reload|restart|start|stop)\b"
    r"[^;&|\n]*\bhermes(?:-gateway)?(?:\.service)?\b",
    re.IGNORECASE,
)
_SYSV_LIFECYCLE_RE = re.compile(
    r"\bservice\b[^;&|\n]*\bhermes(?:-gateway)?\b"
    r"[^;&|\n]*\b(?:reload|restart|start|stop)\b",
    re.IGNORECASE,
)
_LAUNCHD_LIFECYCLE_RE = re.compile(
    r"\blaunchctl\b"
    r"[^;&|\n]*\b(?:bootout|bootstrap|kickstart|kill|remove|unload)\b"
    r"[^;&|\n]*\b[^\s;&|]*hermes[^\s;&|]*",
    re.IGNORECASE,
)
_NAMED_SIGNAL_RE = re.compile(
    r"\b(?:killall|pkill)\b[^;&|\n]*\b(?:hermes|hermes-gateway)\b"
    r"|\bkill\b[^;&|\n]*\bpgrep\b[^;&|\n]*\bhermes(?:-gateway)?\b",
    re.IGNORECASE,
)
_SIGNAL_RE = re.compile(r"(?:^|[;&|]\s*)kill\s+(?:-\S+\s+)*(\d+)\b")


def _deployment_guard_enabled() -> bool:
    raw_env = os.getenv("HERMES_GATEWAY_DEPLOYMENT_GUARD")
    if raw_env is not None:
        return is_truthy_value(raw_env, default=False)
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
        gateway = config.get("gateway", {})
        guard = gateway.get("deployment_guard", {}) if isinstance(gateway, dict) else {}
        return bool(
            isinstance(guard, dict)
            and is_truthy_value(guard.get("enabled"), default=False)
        )
    except Exception:
        return False


def _is_gateway_platform(platform: Any) -> bool:
    value = str(getattr(platform, "value", platform) or "").strip().lower()
    return bool(value) and value not in {"cli", "local"}


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolved_path(value: Any, *, base: Path) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(os.path.expandvars(raw)).expanduser()
    if not path.is_absolute():
        path = base / path
    try:
        return path.resolve(strict=False)
    except OSError:
        return path.absolute()


def _active_checkout() -> Path:
    return Path(__file__).resolve().parents[1]


def _protected_home_paths() -> tuple[Path, ...]:
    home = get_hermes_home().resolve(strict=False)
    return (
        home / ".env",
        home / "config.yaml",
        home / "SOUL.md",
        home / "plugins",
        home / "hermes-agent",
        home / "venv",
    )


def _path_is_protected(path: Path) -> bool:
    checkout = _active_checkout()
    if _is_relative_to(path, checkout):
        return True
    for protected in _protected_home_paths():
        if path == protected or _is_relative_to(path, protected):
            return True
    if "hermes" in path.name.lower() and path.name.endswith(".service") and (
        "/systemd/" in path.as_posix() or "/etc/" in path.as_posix()
    ):
        return True
    return False


def _file_tool_target(
    function_args: Mapping[str, Any],
    *,
    effective_task_id: str,
) -> Path | None:
    raw = function_args.get("path")
    if not raw:
        return None
    try:
        from tools.file_tools import _resolve_path_for_task

        return _resolve_path_for_task(
            str(raw),
            effective_task_id or "default",
        ).resolve(strict=False)
    except Exception:
        return _resolved_path(raw, base=Path.cwd())


def _lark_file_target(
    function_args: Mapping[str, Any],
    *,
    action: str,
) -> Path | None:
    params = function_args.get("params")
    if not isinstance(params, Mapping):
        return None
    key = "destination" if action == "download" else "file_path"
    return _resolved_path(params.get(key), base=Path.cwd())


def _command_segment_is_mutating(segment: str) -> bool:
    try:
        tokens = shlex.split(segment, posix=True)
    except ValueError:
        tokens = segment.split()
    while tokens and (
        tokens[0] in {"command", "env", "sudo"}
        or ("=" in tokens[0] and not tokens[0].startswith(("/", "./", "../")))
        or tokens[0].startswith("-")
    ):
        tokens.pop(0)
    if not tokens:
        return False

    executable = Path(tokens[0]).name
    if executable in _ALWAYS_MUTATING_COMMANDS or executable in _SCRIPT_COMMANDS:
        return True

    args = tokens[1:]
    if executable == "git":
        index = 0
        while index < len(args):
            token = args[index]
            if token in {"-C", "--git-dir", "--work-tree", "--namespace"}:
                index += 2
                continue
            if token.startswith("-"):
                index += 1
                continue
            return token.lower() in _GIT_MUTATING_SUBCOMMANDS
        return False

    positional = [token for token in args if not token.startswith("-")]
    subcommand = positional[0].lower() if positional else ""
    if executable in {"pip", "pip3"}:
        return subcommand in _PIP_MUTATING_SUBCOMMANDS
    if executable == "uv":
        if subcommand == "pip" and len(positional) > 1:
            return positional[1].lower() in _PIP_MUTATING_SUBCOMMANDS
        return subcommand in _UV_MUTATING_SUBCOMMANDS
    return False


def _terminal_is_blocked(command: str, *, workdir: Any) -> bool:
    if any(
        pattern.search(command)
        for pattern in (
            _HERMES_CLI_LIFECYCLE_RE,
            _SERVICE_LIFECYCLE_RE,
            _SYSV_LIFECYCLE_RE,
            _LAUNCHD_LIFECYCLE_RE,
            _NAMED_SIGNAL_RE,
        )
    ):
        return True
    for match in _SIGNAL_RE.finditer(command):
        if int(match.group(1)) in {os.getpid(), os.getppid()}:
            return True

    base = _resolved_path(workdir, base=Path.cwd()) or Path.cwd()
    mutating = any(
        _command_segment_is_mutating(segment)
        for segment in re.split(r"(?:&&|\|\||[;|])", command)
    ) or any(
        marker in command for marker in (">", "2>", ">>")
    )
    if not mutating:
        return False
    if _path_is_protected(base.resolve(strict=False)):
        return True

    protected_fragments = {
        _active_checkout().as_posix(),
        *(path.as_posix() for path in _protected_home_paths()),
    }
    expanded = os.path.expandvars(os.path.expanduser(command))
    return any(fragment in expanded for fragment in protected_fragments)


def blocked_gateway_tool_message(
    *,
    platform: Any,
    function_name: str,
    function_args: Mapping[str, Any],
    effective_task_id: str = "",
) -> str | None:
    """Return a user-facing block message for protected gateway operations."""
    if not _is_gateway_platform(platform) or not _deployment_guard_enabled():
        return None

    if function_name in {"write_file", "patch"}:
        target = _file_tool_target(
            function_args,
            effective_task_id=effective_task_id,
        )
        if target is not None and _path_is_protected(target):
            return _BLOCK_MESSAGE
    elif function_name == "lark_drive":
        action = str(function_args.get("action") or "").strip()
        if action in {"download", "upload"}:
            target = _lark_file_target(function_args, action=action)
            if target is not None and _path_is_protected(target):
                return _BLOCK_MESSAGE
    elif function_name == "lark_tasks":
        action = str(function_args.get("action") or "").strip()
        if action == "upload_attachment":
            target = _lark_file_target(function_args, action="upload")
            if target is not None and _path_is_protected(target):
                return _BLOCK_MESSAGE
    elif function_name == "terminal":
        command = str(function_args.get("command", "") or "")
        if _terminal_is_blocked(command, workdir=function_args.get("workdir")):
            return _BLOCK_MESSAGE
    return None
