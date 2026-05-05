#!/usr/bin/env python3
"""Send a Lark report for open Hermes PRs that touch the OpenViking plugin."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


DEFAULT_UPSTREAM_REPO = "NousResearch/hermes-agent"
DEFAULT_OUTPUT = "openviking-pr-report.md"
DEFAULT_MAX_SEARCH_RESULTS = 1000
DEFAULT_FILE_FETCH_CONCURRENCY = 20
DEFAULT_LLM_TIMEOUT_SECONDS = 120
DEFAULT_BODY_CHARS = 4000
DEFAULT_LARK_MARKDOWN_CHARS = 12000

OPENVIKING_SEARCH_TERMS = (
    "openviking",
    '"open viking"',
    "viking://",
    "viking_",
    "viking",
)

OPENVIKING_KEYWORD_RE = re.compile(
    r"""
    open[\s_-]*viking
    | viking://
    | \bviking\b
    | viking[_-][a-z0-9_./-]+
    | viking\s+(?:resource|memory|tool|provider|recall|search|endpoint|api|plugin)
    """,
    re.IGNORECASE | re.VERBOSE,
)

OPENVIKING_PLUGIN_DIR_PREFIXES = (
    "plugins/memory/openviking",
    "tests/plugins/memory/openviking",
)

OPENVIKING_PLUGIN_FILE_PREFIXES = (
    "tests/plugins/memory/test_openviking",
)

@dataclass
class PullRequest:
    number: int
    title: str
    body: str
    html_url: str
    updated_at: str
    head_ref: str = ""
    files: list[str] = field(default_factory=list)

    @property
    def title_body_text(self) -> str:
        return f"{self.title}\n{self.body}"

    @property
    def text_matches(self) -> bool:
        return bool(OPENVIKING_KEYWORD_RE.search(self.title_body_text))

    @property
    def path_matches(self) -> bool:
        return paths_have_openviking_signal(self.files)

    @property
    def is_relevant(self) -> bool:
        return self.path_matches

    @property
    def match_reason(self) -> str:
        matched_paths = openviking_paths(self.files)
        if matched_paths:
            return "changed OpenViking plugin path: " + ", ".join(matched_paths[:3])
        return "not a direct OpenViking plugin path match"


@dataclass
class PrSummary:
    number: int
    summary: str


@dataclass
class ReportGroup:
    title: str
    prs: list[PrSummary] = field(default_factory=list)


class GitHubApiError(RuntimeError):
    pass


class GitHubClient:
    def __init__(self, token: str, api_url: str = "https://api.github.com") -> None:
        self.api_url = api_url.rstrip("/")
        self.token = token

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        timeout: int = 30,
    ) -> Any:
        url = f"{self.api_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        body = json.dumps(data).encode("utf-8") if data is not None else None
        headers = {
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "openviking-pr-report",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise GitHubApiError(f"GitHub API {method} {path} failed: {exc.code} {detail}") from exc
        if not raw:
            return None
        return json.loads(raw.decode("utf-8"))


def split_repo(repo: str) -> tuple[str, str]:
    owner, _, name = repo.partition("/")
    if not owner or not name:
        raise ValueError(f"Repository must be owner/name, got: {repo!r}")
    return owner, name


def int_from_env(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, value)


def truncate_text(value: str, limit: int) -> str:
    compact = "\n".join(line.rstrip() for line in (value or "").splitlines()).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 20].rstrip() + "\n... [truncated]"


def make_pull_request_from_issue(raw: dict[str, Any]) -> PullRequest:
    return PullRequest(
        number=int(raw["number"]),
        title=str(raw.get("title") or ""),
        body=str(raw.get("body") or ""),
        html_url=str(raw.get("html_url") or ""),
        updated_at=str(raw.get("updated_at") or ""),
    )


def fetch_search_prs(
    client: GitHubClient,
    repo: str,
    *,
    max_results: int,
) -> list[PullRequest]:
    prs_by_number: dict[int, PullRequest] = {}
    for term in OPENVIKING_SEARCH_TERMS:
        query = f"repo:{repo} is:pr is:open {term}"
        page = 1
        while len(prs_by_number) < max_results:
            data = client.request(
                "GET",
                "/search/issues",
                params={
                    "q": query,
                    "sort": "updated",
                    "order": "desc",
                    "per_page": 100,
                    "page": page,
                },
            )
            if not isinstance(data, dict) or not isinstance(data.get("items"), list):
                raise GitHubApiError("Expected object response from /search/issues")
            items = data["items"]
            for raw in items:
                pr = make_pull_request_from_issue(raw)
                prs_by_number.setdefault(pr.number, pr)
                if len(prs_by_number) >= max_results:
                    break
            if len(items) < 100:
                break
            page += 1

    if len(prs_by_number) >= max_results:
        print(f"Stopped GitHub search scan after safety cap of {max_results} PRs.", file=sys.stderr)
    prs = sorted(prs_by_number.values(), key=lambda pr: pr.number, reverse=True)
    print(f"Fetched {len(prs)} open PR candidate(s) from GitHub search.", file=sys.stderr)
    return prs


def fetch_pull_file_paths(client: GitHubClient, repo: str, number: int) -> list[str]:
    owner, name = split_repo(repo)
    paths: list[str] = []
    page = 1
    while True:
        files = client.request(
            "GET",
            f"/repos/{owner}/{name}/pulls/{number}/files",
            params={"per_page": 100, "page": page},
        )
        if not isinstance(files, list):
            raise GitHubApiError(f"Expected list response from /repos/{owner}/{name}/pulls/{number}/files")
        paths.extend(str(item.get("filename")) for item in files if item.get("filename"))
        if len(files) < 100:
            return paths
        page += 1


def attach_file_paths(
    client: GitHubClient,
    repo: str,
    prs: list[PullRequest],
    *,
    concurrency: int,
) -> None:
    if not prs:
        return
    errors = 0
    with ThreadPoolExecutor(max_workers=min(concurrency, len(prs))) as executor:
        future_to_pr = {
            executor.submit(fetch_pull_file_paths, client, repo, pr.number): pr
            for pr in prs
        }
        for future in as_completed(future_to_pr):
            pr = future_to_pr[future]
            try:
                pr.files = future.result()
            except Exception as exc:  # noqa: BLE001 - one failed path lookup should not abort the report.
                errors += 1
                print(f"Failed to fetch changed file paths for PR #{pr.number}: {exc}", file=sys.stderr)
    if errors:
        print(f"Changed file path lookup failed for {errors} PR(s).", file=sys.stderr)


def openviking_paths(files: list[str]) -> list[str]:
    matched: list[str] = []
    for path in files:
        lowered = path.lower()
        dir_match = any(
            lowered == prefix or lowered.startswith(f"{prefix}/")
            for prefix in OPENVIKING_PLUGIN_DIR_PREFIXES
        )
        file_match = any(lowered.startswith(prefix) for prefix in OPENVIKING_PLUGIN_FILE_PREFIXES)
        if dir_match or file_match:
            matched.append(path)
    return matched


def paths_have_openviking_signal(files: list[str]) -> bool:
    return bool(openviking_paths(files))


def filter_relevant_prs(prs: list[PullRequest]) -> list[PullRequest]:
    return sorted((pr for pr in prs if pr.is_relevant), key=lambda pr: pr.number, reverse=True)


def chat_completions_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/chat/completions"):
        return trimmed
    return f"{trimmed}/chat/completions"


def llm_chat_content(
    messages: list[dict[str, str]],
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout_seconds: int,
) -> str:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        chat_completions_url(base_url),
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "openviking-pr-report",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
        raw = response.read().decode("utf-8")
    data = json.loads(raw)
    content = data["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("LLM response did not include markdown content")
    return content.strip() + "\n"


def build_llm_prompt(prs: list[PullRequest], *, body_chars: int) -> list[dict[str, str]]:
    expected_numbers = ", ".join(f"#{pr.number}" for pr in prs)
    payload = [
        {
            "number": pr.number,
            "title": pr.title,
            "body": truncate_text(pr.body, body_chars),
            "url": pr.html_url,
            "updated_at": pr.updated_at,
            "matched_by": pr.match_reason,
            "changed_file_paths": pr.files,
        }
        for pr in prs
    ]
    return [
        {
            "role": "system",
            "content": (
                "You group GitHub PRs and write concise maintainer-facing summaries. "
                "Use only the PRs and facts in the input. Return valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Summarize the currently open PRs that directly change the Hermes OpenViking plugin.\n\n"
                "OpenViking relevance here means the PR changes `plugins/memory/openviking` or "
                "OpenViking-scoped plugin tests. Generic memory framework, gateway lifecycle, cron, "
                "benchmark, or provider infrastructure PRs are intentionally excluded unless they touch "
                "those OpenViking plugin paths.\n\n"
                "The input PR JSON is the authoritative filtered list. Include every input PR exactly once. "
                f"Expected PR numbers: {expected_numbers}.\n\n"
                "Return exactly one JSON object and no Markdown, no code fence, and no prose before or after it.\n"
                "Use this schema:\n"
                "{\n"
                '  "groups": [\n'
                '    {"title": "Local resource uploads", "prs": [{"number": 123, "summary": "..."}]}\n'
                "  ]\n"
                "}\n\n"
                "Grouping requirements:\n"
                "- Reorder PRs to group related or overlapping work next to each other.\n"
                "- Use concise group titles without a `Group:` prefix; the caller adds that label.\n"
                "- Use `Other` for unrelated singles.\n"
                "- Include every expected PR number exactly once, and do not include any number not in the input.\n\n"
                "Summary requirements:\n"
                "- Write each summary in 2-3 sentences.\n"
                "- Use cause-and-effect style: start with the issue, user-visible failure, or capability gap; "
                "then explain the mechanism or affected code path when the input gives enough detail; "
                "then state the concrete fix, behavior change, and tests/validation when available.\n"
                "- For feature PRs, explain the new capability, why it matters, and how it is integrated.\n"
                "- Do not include Markdown headings, bullets, tables, confidence, changed paths, "
                "`Possible Overlaps`, or a separate why/context field.\n"
                "- Avoid vague summaries such as merely saying the PR matched the OpenViking filter.\n\n"
                f"Input PR JSON:\n{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
            ),
        },
    ]


def report_header(pr_count: int) -> str:
    suffix = "PR" if pr_count == 1 else "PRs"
    return (
        "**OpenViking Open PR Triage Report**\n\n"
        f"Overview: {pr_count} open OpenViking plugin {suffix} found.\n\n"
    )


def plain_text_excerpt(value: str, limit: int) -> str:
    lines: list[str] = []
    for line in value.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith("possible overlaps:") or lowered.startswith("**possible overlaps:**"):
            continue
        stripped = re.sub(r"^#{1,6}\s*", "", stripped)
        stripped = re.sub(r"^[-*]\s+", "", stripped)
        stripped = re.sub(r"\*\*([^*]+)\*\*", r"\1", stripped)
        lines.append(stripped)
    compact = re.sub(r"\s+", " ", " ".join(lines)).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 20].rstrip() + " ... [truncated]"


def fallback_summary(pr: PullRequest) -> str:
    body = plain_text_excerpt(pr.body, 360)
    if body:
        return f"{pr.title}. {body}"
    return f"{pr.title}. Review the linked PR for the full implementation details."


def fallback_groups(prs: list[PullRequest]) -> list[ReportGroup]:
    if not prs:
        return []
    return [
        ReportGroup(
            "OpenViking plugin PRs",
            [PrSummary(pr.number, fallback_summary(pr)) for pr in prs],
        )
    ]


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL | re.IGNORECASE)
        if fenced:
            data = json.loads(fenced.group(1))
        else:
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start < 0 or end <= start:
                raise
            data = json.loads(stripped[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("LLM response JSON must be an object")
    return data


def sanitize_group_title(value: Any) -> str:
    title = plain_text_excerpt(str(value or ""), 80).strip()
    title = re.sub(r"^group:\s*", "", title, flags=re.IGNORECASE).strip()
    return title or "Other"


def sanitize_summary(value: Any, *, fallback: str) -> str:
    summary = plain_text_excerpt(str(value or ""), 900).strip()
    summary = re.sub(r"^summary:\s*", "", summary, flags=re.IGNORECASE).strip()
    return summary or fallback


def validate_grouped_report(data: dict[str, Any], prs: list[PullRequest]) -> tuple[list[ReportGroup], list[int]]:
    pr_by_number = {pr.number: pr for pr in prs}
    seen: set[int] = set()
    groups: list[ReportGroup] = []

    raw_groups = data.get("groups")
    if not isinstance(raw_groups, list):
        raw_groups = []

    for raw_group in raw_groups:
        if not isinstance(raw_group, dict):
            continue
        items: list[PrSummary] = []
        raw_prs = raw_group.get("prs")
        if not isinstance(raw_prs, list):
            continue
        for raw_item in raw_prs:
            if not isinstance(raw_item, dict):
                continue
            number_value = raw_item.get("number", raw_item.get("pr_number"))
            try:
                number = int(number_value)
            except (TypeError, ValueError):
                continue
            if number not in pr_by_number or number in seen:
                continue
            pr = pr_by_number[number]
            summary = sanitize_summary(raw_item.get("summary"), fallback=fallback_summary(pr))
            items.append(PrSummary(number, summary))
            seen.add(number)
        if items:
            groups.append(ReportGroup(sanitize_group_title(raw_group.get("title")), items))

    missing = [pr.number for pr in prs if pr.number not in seen]
    if missing:
        groups.append(ReportGroup("Other", [PrSummary(number, fallback_summary(pr_by_number[number])) for number in missing]))

    return groups, missing


def render_markdown_report(groups: list[ReportGroup], prs: list[PullRequest], *, llm_status: str) -> str:
    lines = [report_header(len(prs)).rstrip(), f"LLM: {llm_status}", ""]
    if not prs:
        lines.append("No open OpenViking plugin PRs found.")
        return "\n".join(lines) + "\n"

    pr_by_number = {pr.number: pr for pr in prs}
    for group in groups:
        lines.append(f"**Group: {group.title}**")
        for item in group.prs:
            pr = pr_by_number.get(item.number)
            if not pr:
                continue
            lines.append(f"- [#{pr.number}]({pr.html_url}) {pr.title}")
            lines.append(f"  - **Summary:** {item.summary}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def summarize_with_llm(
    prs: list[PullRequest],
    *,
    body_chars: int,
    timeout_seconds: int,
) -> tuple[list[ReportGroup], str]:
    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "")
    model = os.getenv("LLM_MODEL", "")
    if not (api_key and base_url and model):
        return fallback_groups(prs), "not configured"

    try:
        content = llm_chat_content(
            build_llm_prompt(prs, body_chars=body_chars),
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - still send a useful report when LLM fails.
        status = f"configured but skipped after error: {exc}"
        return fallback_groups(prs), status

    try:
        groups, missing = validate_grouped_report(parse_json_object(content), prs)
    except Exception as exc:  # noqa: BLE001 - malformed LLM output should not block the report.
        status = f"configured but skipped after invalid JSON: {exc}"
        return fallback_groups(prs), status

    status = f"summarized with `{model}`"
    if missing:
        missing_refs = ", ".join(f"#{number}" for number in missing)
        status = f"{status}; filled missing PRs in Other: {missing_refs}"
    return groups, status


def build_lark_elements(groups: list[ReportGroup], prs: list[PullRequest]) -> list[dict[str, Any]]:
    pr_by_number = {pr.number: pr for pr in prs}
    elements: list[dict[str, Any]] = [
        {
            "tag": "markdown",
            "content": f"Overview: {len(prs)} open OpenViking plugin {'PR' if len(prs) == 1 else 'PRs'} found.",
        }
    ]

    if not prs:
        elements.append({"tag": "markdown", "content": "No open OpenViking plugin PRs found."})
        return elements

    for group in groups:
        elements.append({"tag": "markdown", "content": f"**Group: {group.title}**"})
        for item in group.prs:
            pr = pr_by_number.get(item.number)
            if not pr:
                continue
            elements.append(
                {
                    "tag": "collapsible_panel",
                    "expanded": False,
                    "header": {
                        "title": {
                            "tag": "markdown",
                            "content": f"[#{pr.number}]({pr.html_url}) {pr.title}",
                        },
                        "vertical_align": "center",
                        "icon": {
                            "tag": "standard_icon",
                            "token": "down-small-ccm_outlined",
                            "size": "16px 16px",
                        },
                        "icon_position": "right",
                        "icon_expanded_angle": -180,
                    },
                    "border": {
                        "color": "grey",
                        "corner_radius": "5px",
                    },
                    "vertical_spacing": "4px",
                    "padding": "6px 8px 6px 8px",
                    "elements": [
                        {
                            "tag": "markdown",
                            "content": f"**Summary:** {item.summary}",
                        }
                    ],
                }
            )
    return elements


def build_lark_card(groups: list[ReportGroup], prs: list[PullRequest], *, title: str) -> dict[str, Any]:
    return {
        "msg_type": "interactive",
        "card": {
            "schema": "2.0",
            "config": {"wide_screen_mode": True, "update_multi": True},
            "header": {
                "template": "blue",
                "title": {"tag": "plain_text", "content": title},
            },
            "body": {"elements": build_lark_elements(groups, prs)},
        },
    }


def post_lark_card(webhook_url: str, card: dict[str, Any]) -> None:
    req = urllib.request.Request(
        webhook_url,
        data=json.dumps(card).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "openviking-pr-report"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Lark webhook failed: {exc.code} {detail}") from exc


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-repo", default=os.getenv("UPSTREAM_REPOSITORY", DEFAULT_UPSTREAM_REPO))
    parser.add_argument(
        "--max-search-results",
        type=int,
        default=int_from_env("MAX_SEARCH_RESULTS", DEFAULT_MAX_SEARCH_RESULTS),
    )
    parser.add_argument(
        "--file-fetch-concurrency",
        type=int,
        default=int_from_env("FILE_FETCH_CONCURRENCY", DEFAULT_FILE_FETCH_CONCURRENCY),
    )
    parser.add_argument("--llm-timeout-seconds", type=int, default=int_from_env("LLM_TIMEOUT_SECONDS", DEFAULT_LLM_TIMEOUT_SECONDS))
    parser.add_argument("--body-chars", type=int, default=int_from_env("PR_BODY_CHARS", DEFAULT_BODY_CHARS))
    parser.add_argument("--lark-markdown-chars", type=int, default=int_from_env("LARK_MARKDOWN_CHARS", DEFAULT_LARK_MARKDOWN_CHARS))
    parser.add_argument("--output", default=os.getenv("REPORT_OUTPUT", DEFAULT_OUTPUT))
    parser.add_argument("--dry-run", action="store_true", help="Generate report without posting to Lark")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    github_token = os.getenv("UPSTREAM_GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or ""
    lark_webhook_url = os.getenv("LARK_WEBHOOK_URL", "")
    if not lark_webhook_url and not args.dry_run:
        print("LARK_WEBHOOK_URL is required unless --dry-run is used.", file=sys.stderr)
        return 2

    client = GitHubClient(github_token)
    candidate_prs = fetch_search_prs(
        client,
        args.upstream_repo,
        max_results=args.max_search_results,
    )
    attach_file_paths(
        client,
        args.upstream_repo,
        candidate_prs,
        concurrency=args.file_fetch_concurrency,
    )
    relevant_prs = filter_relevant_prs(candidate_prs)
    print(f"Found {len(relevant_prs)} OpenViking plugin PR(s).", file=sys.stderr)

    if relevant_prs:
        groups, llm_status = summarize_with_llm(
            relevant_prs,
            body_chars=args.body_chars,
            timeout_seconds=args.llm_timeout_seconds,
        )
    else:
        llm_status = "skipped because no relevant PRs were found"
        groups = []
    markdown = render_markdown_report(groups, relevant_prs, llm_status=llm_status)

    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(markdown)
    print(markdown)
    print(f"LLM status: {llm_status}", file=sys.stderr)

    suffix = "PR" if len(relevant_prs) == 1 else "PRs"
    title = f"OpenViking PR Report - {datetime.now(UTC).strftime('%Y-%m-%d')} - {len(relevant_prs)} {suffix}"
    card = build_lark_card(groups, relevant_prs, title=title)
    if args.dry_run:
        print(json.dumps(card, ensure_ascii=False, indent=2))
        return 0

    post_lark_card(lark_webhook_url, card)
    print("Posted OpenViking PR report to Lark.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
