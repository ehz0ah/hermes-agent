#!/usr/bin/env python3
"""Send a Lark report for open OpenViking-related Hermes PRs."""

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

OPENVIKING_PATH_PREFIXES = (
    "plugins/memory/openviking",
    "tests/plugins/memory/test_openviking",
)

PR_SECTION_RE = re.compile(
    r"(?:^|\n)---\s*\n+(?P<section>(?:### \[#|\*\*\[#)(?P<number>\d+)\]\([^)]*\) .+?)(?=\n---\s*\n+(?:### \[#|\*\*\[#)|\Z)",
    re.DOTALL,
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
        return self.text_matches or self.path_matches

    @property
    def match_reason(self) -> str:
        reasons: list[str] = []
        if self.text_matches:
            reasons.append("title/body keyword match")
        matched_paths = openviking_paths(self.files)
        if matched_paths:
            reasons.append("changed OpenViking path: " + ", ".join(matched_paths[:3]))
        return "; ".join(reasons) or "GitHub OpenViking search match"


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
            except Exception as exc:  # noqa: BLE001 - a failed path lookup should not hide text matches.
                errors += 1
                print(f"Failed to fetch changed file paths for PR #{pr.number}: {exc}", file=sys.stderr)
    if errors:
        print(f"Changed file path lookup failed for {errors} PR(s).", file=sys.stderr)


def openviking_paths(files: list[str]) -> list[str]:
    matched: list[str] = []
    for path in files:
        lowered = path.lower()
        if any(lowered.startswith(prefix) for prefix in OPENVIKING_PATH_PREFIXES):
            matched.append(path)
    return matched


def paths_have_openviking_signal(files: list[str]) -> bool:
    return bool(openviking_paths(files))


def filter_relevant_prs(prs: list[PullRequest]) -> list[PullRequest]:
    return sorted(prs, key=lambda pr: pr.number, reverse=True)


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
                "You write concise GitHub PR triage reports in standard Markdown. "
                "Use only the PRs and facts in the input. Do not invent PRs or code details."
            ),
        },
        {
            "role": "user",
            "content": (
                "Summarize the currently open OpenViking-related PRs.\n\n"
                "OpenViking is the Hermes memory/context plugin around "
                "`plugins/memory/openviking`, `viking_*` tools, `viking://` resources, "
                "provider setup, endpoint migration, memory recall/search, and resource handling.\n\n"
                "The input PR JSON is the authoritative filtered list. Include every input PR exactly once, "
                "even if a PR only looks indirectly related from its title/body. "
                f"Expected PR numbers: {expected_numbers}.\n\n"
                "Return standard Markdown only. Include:\n"
                "- Do not include a top-level title or overview; the caller adds those.\n"
                "- Reorder PRs to physically group related or overlapping work next to each other. "
                "Use compact group labels like `**Group: Local resource uploads**`; use `**Group: Other**` for unrelated singles.\n"
                "- Separate every PR with a visible horizontal divider line `---` before the PR title line.\n"
                "- One compact section per PR using `**[#number](url) title**`; do not use Markdown headings for PR sections.\n"
                "- Under each PR, add a bold `Summary:` label followed by 2-3 sentences in a clear "
                "cause-and-effect style: start with the issue, user-visible failure, or capability gap; "
                "then explain the mechanism or affected code path when the input gives enough detail; "
                "then state the concrete fix, behavior change, and tests/validation when available. "
                "For feature PRs, explain the new capability, why it matters, and how it is integrated. "
                "Avoid vague summaries such as merely saying the PR matched the OpenViking filter.\n"
                "- Do not include a `Possible Overlaps` field; grouping order replaces that field.\n"
                "- Do not include confidence, changed paths, or a separate why/context section in the final report.\n"
                "- Do not use Markdown tables; Lark cards render stacked sections more reliably.\n"
                "- Keep it compact and maintainer-facing.\n\n"
                f"Input PR JSON:\n{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
            ),
        },
    ]


def report_header(prs: list[PullRequest]) -> str:
    count = len(prs)
    suffix = "PR" if count == 1 else "PRs"
    return (
        "**OpenViking Open PR Triage Report**\n\n"
        f"Overview: {count} open OpenViking-related {suffix} found.\n\n"
    )


def strip_generated_preamble(markdown: str) -> str:
    lines = markdown.strip().splitlines()
    while lines and (
        lines[0].startswith("# ")
        or lines[0].startswith("**OpenViking Open PR Triage Report**")
        or lines[0].lower().startswith("overview:")
    ):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines).strip()


def plain_text_excerpt(value: str, limit: int) -> str:
    lines: list[str] = []
    for line in value.splitlines():
        stripped = line.strip()
        if not stripped:
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
        return f"{pr.title}. {truncate_text(body, 360)}"
    return f"{pr.title}. Review the linked PR for the full implementation details."


def render_pr_fallback_section(pr: PullRequest) -> str:
    return "\n".join(
        [
            "---",
            "",
            f"**[#{pr.number}]({pr.html_url}) {pr.title}**",
            f"**Summary:** {fallback_summary(pr)}",
        ]
    )


def extract_pr_sections(markdown: str) -> dict[int, str]:
    sections: dict[int, str] = {}
    for match in PR_SECTION_RE.finditer(markdown.strip()):
        number = int(match.group("number"))
        sections[number] = match.group("section").strip()
    return sections


def complete_pr_sections(markdown: str, prs: list[PullRequest]) -> tuple[str, list[int]]:
    body = strip_generated_preamble(markdown)
    sections = extract_pr_sections(body)
    missing = [pr.number for pr in prs if pr.number not in sections]
    if not missing:
        return normalize_report_markdown(body).strip() + "\n", []

    lines: list[str] = []
    for pr in prs:
        section = sections.get(pr.number)
        if section:
            lines.extend(["---", section, ""])
        else:
            lines.extend([render_pr_fallback_section(pr), ""])
    return normalize_report_markdown("\n".join(lines)).strip() + "\n", missing


def normalize_report_markdown(markdown: str) -> str:
    lines: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith("possible overlaps:") or lowered.startswith("**possible overlaps:**"):
            continue
        line = re.sub(r"^(?P<indent>\s*)### (?P<title>\[#\d+\]\([^)]*\) .+)$", r"\g<indent>**\g<title>**", line)
        line = re.sub(r"^(?P<indent>\s*)Summary:\s*", r"\g<indent>**Summary:** ", line)
        line = re.sub(r"^(?P<indent>\s*)\*\*Summary:\*\*\s*", r"\g<indent>**Summary:** ", line)
        lines.append(line)
    return "\n".join(lines)


def render_fallback_report(prs: list[PullRequest], *, llm_status: str) -> str:
    lines = [report_header(prs).rstrip(), f"LLM: {llm_status}", ""]
    if not prs:
        lines.append("No open OpenViking-related PRs found.")
        return "\n".join(lines) + "\n"

    lines.append("**Group: OpenViking-related PRs**")
    lines.extend(render_pr_fallback_section(pr) for pr in prs)
    return "\n\n".join(lines) + "\n"


def summarize_with_llm(
    prs: list[PullRequest],
    *,
    body_chars: int,
    timeout_seconds: int,
) -> tuple[str, str]:
    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "")
    model = os.getenv("LLM_MODEL", "")
    if not (api_key and base_url and model):
        return render_fallback_report(prs, llm_status="not configured"), "not configured"

    try:
        markdown = llm_chat_content(
            build_llm_prompt(prs, body_chars=body_chars),
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - still send a useful report when LLM fails.
        status = f"configured but skipped after error: {exc}"
        return render_fallback_report(prs, llm_status=status), status
    body, missing = complete_pr_sections(markdown, prs)
    status = f"summarized with `{model}`"
    if missing:
        missing_refs = ", ".join(f"#{number}" for number in missing)
        status = f"{status}; filled missing sections for {missing_refs}"
    return report_header(prs) + body, status


def split_lark_markdown(markdown: str, *, markdown_limit: int) -> list[str]:
    blocks = re.split(r"(?=^---\s*$)", markdown.strip(), flags=re.MULTILINE)
    chunks: list[str] = []
    current = ""
    for block in (part.strip() for part in blocks if part.strip()):
        candidate = f"{current}\n\n{block}".strip() if current else block
        if len(candidate) <= markdown_limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(block) <= markdown_limit:
            current = block
        else:
            chunks.append(block[: markdown_limit - 80].rstrip() + "\n\n_Section truncated; open the workflow summary for full text._")
            current = ""
    if current:
        chunks.append(current)
    return chunks or [""]


def build_lark_card(markdown: str, *, title: str, markdown_limit: int) -> dict[str, Any]:
    elements = [
        {
            "tag": "markdown",
            "content": content,
        }
        for content in split_lark_markdown(markdown, markdown_limit=markdown_limit)
    ]
    return {
        "msg_type": "interactive",
        "card": {
            "schema": "2.0",
            "config": {"wide_screen_mode": True, "update_multi": True},
            "header": {
                "template": "blue",
                "title": {"tag": "plain_text", "content": title},
            },
            "body": {"elements": elements},
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
    print(f"Found {len(relevant_prs)} OpenViking-related PR(s).", file=sys.stderr)

    if relevant_prs:
        markdown, llm_status = summarize_with_llm(
            relevant_prs,
            body_chars=args.body_chars,
            timeout_seconds=args.llm_timeout_seconds,
        )
    else:
        llm_status = "skipped because no relevant PRs were found"
        markdown = report_header([]) + "No open OpenViking-related PRs found.\n"

    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(markdown)
    print(markdown)
    print(f"LLM status: {llm_status}", file=sys.stderr)

    suffix = "PR" if len(relevant_prs) == 1 else "PRs"
    title = f"OpenViking PR Report - {datetime.now(UTC).strftime('%Y-%m-%d')} - {len(relevant_prs)} {suffix}"
    card = build_lark_card(markdown, title=title, markdown_limit=args.lark_markdown_chars)
    if args.dry_run:
        print(json.dumps(card, ensure_ascii=False, indent=2))
        return 0

    post_lark_card(lark_webhook_url, card)
    print("Posted OpenViking PR report to Lark.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
