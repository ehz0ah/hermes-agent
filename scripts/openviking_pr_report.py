#!/usr/bin/env python3
"""Generate a GitHub issue report for OpenViking-related Hermes PRs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any


DEFAULT_UPSTREAM_REPO = "NousResearch/hermes-agent"
DEFAULT_REPORT_TITLE = "OpenViking PR Report"
OPENVIKING_TERMS = (
    "openviking",
    "open viking",
    "viking_",
    "viking://",
    "temp_upload",
    "temp upload",
)
SEARCH_TERMS = (
    "openviking",
    "viking_add_resource",
    "viking_delete_resource",
    "viking://",
    "temp_upload",
)
OPENVIKING_PATH_MARKERS = (
    "plugins/memory/openviking",
    "tests/plugins/memory/test_openviking",
    "openviking",
)
STOPWORDS = {
    "a",
    "add",
    "adds",
    "and",
    "are",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "pr",
    "support",
    "the",
    "this",
    "to",
    "with",
}


@dataclass
class PullRequest:
    number: int
    title: str
    body: str
    state: str
    html_url: str
    author: str
    created_at: str
    updated_at: str
    closed_at: str | None = None
    merged_at: str | None = None
    draft: bool = False
    head_ref: str = ""
    labels: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)
    linked_issues: set[int] = field(default_factory=set)

    @property
    def lifecycle_state(self) -> str:
        if self.merged_at:
            return "merged"
        return self.state


@dataclass
class DuplicateEdge:
    left: int
    right: int
    score: int
    reasons: list[str]


@dataclass
class DuplicateCluster:
    prs: list[PullRequest]
    reasons: list[str]
    topic: str


class GitHubApiError(RuntimeError):
    pass


class GitHubClient:
    def __init__(
        self,
        token: str,
        api_url: str = "https://api.github.com",
        *,
        fallback_to_unauth: bool = False,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.token = token
        self.fallback_to_unauth = fallback_to_unauth

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        _allow_fallback: bool = True,
    ) -> Any:
        url = f"{self.api_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        body = None
        if data is not None:
            body = json.dumps(data).encode("utf-8")
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
            with urllib.request.urlopen(req, timeout=30) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if (
                self.fallback_to_unauth
                and self.token
                and method == "GET"
                and exc.code in {403, 404}
                and _allow_fallback
            ):
                fallback = GitHubClient("", self.api_url)
                return fallback.request(method, path, params=params, data=data, _allow_fallback=False)
            raise GitHubApiError(f"GitHub API {method} {path} failed: {exc.code} {detail}") from exc
        if not raw:
            return None
        return json.loads(raw.decode("utf-8"))

    def paginate_list(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[Any]:
        collected: list[Any] = []
        page = 1
        while True:
            page_params = dict(params or {})
            page_params.update({"per_page": 100, "page": page})
            items = self.request("GET", path, params=page_params)
            if not isinstance(items, list):
                raise GitHubApiError(f"Expected list response from {path}")
            collected.extend(items)
            if len(items) < 100 or (limit is not None and len(collected) >= limit):
                return collected[:limit] if limit is not None else collected
            page += 1

    def search_issues(self, query: str, *, limit: int = 100) -> list[dict[str, Any]]:
        response = self.request(
            "GET",
            "/search/issues",
            params={"q": query, "per_page": min(limit, 100), "page": 1},
        )
        return list(response.get("items", []))[:limit]


def split_repo(repo: str) -> tuple[str, str]:
    parts = repo.split("/", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Repository must be owner/name, got: {repo!r}")
    return parts[0], parts[1]


def parse_github_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def label_names(raw: dict[str, Any]) -> list[str]:
    labels = raw.get("labels") or []
    names: list[str] = []
    for label in labels:
        if isinstance(label, dict):
            name = label.get("name")
            if name:
                names.append(str(name))
        elif label:
            names.append(str(label))
    return names


def text_has_openviking_signal(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in OPENVIKING_TERMS)


def paths_have_openviking_signal(files: list[str]) -> bool:
    lowered = [path.lower() for path in files]
    return any(any(marker in path for marker in OPENVIKING_PATH_MARKERS) for path in lowered)


def raw_pr_metadata_has_signal(raw: dict[str, Any]) -> bool:
    head = raw.get("head") or {}
    labels = " ".join(label_names(raw))
    text = "\n".join(
        [
            str(raw.get("title") or ""),
            str(raw.get("body") or ""),
            str(head.get("ref") or ""),
            labels,
        ]
    )
    return text_has_openviking_signal(text)


def pr_has_openviking_signal(pr: PullRequest) -> bool:
    text = "\n".join([pr.title, pr.body, pr.head_ref, " ".join(pr.labels), *pr.comments])
    return text_has_openviking_signal(text) or paths_have_openviking_signal(pr.files)


def extract_issue_refs(*texts: str) -> set[int]:
    refs: set[int] = set()
    for text in texts:
        for match in re.findall(r"#(\d+)", text or ""):
            refs.add(int(match))
    return refs


def token_set(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9_]+", text.lower()))
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}


def signature_terms(pr: PullRequest) -> set[str]:
    text = "\n".join([pr.title, pr.body, *pr.comments]).lower()
    signatures: set[str] = set(re.findall(r"viking_[a-z0-9_]+", text))
    endpoint_matches = re.findall(r"/api/v1/[a-z0-9_/{}/-]+", text)
    signatures.update(endpoint_matches)
    if re.search(r"temp[_ -]?upload", text):
        signatures.add("temp_upload")
    if re.search(r"local (?:file|files|path|paths|resource|resources|director(?:y|ies))", text) and "upload" in text:
        signatures.add("local_resource_upload")
    if "fallback" in text and ("recall" in text or "prefetch" in text or "search" in text):
        signatures.add("fallback_recall")
    if "auto" in text and "commit" in text:
        signatures.add("auto_commit")
    return signatures


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def duplicate_reasons(left: PullRequest, right: PullRequest) -> DuplicateEdge | None:
    reasons: list[str] = []
    score = 0

    left_refs = left.linked_issues - {right.number}
    right_refs = right.linked_issues - {left.number}
    shared_refs = left_refs & right_refs
    if shared_refs:
        score += 5
        refs = ", ".join(f"#{ref}" for ref in sorted(shared_refs)[:5])
        reasons.append(f"both reference {refs}")

    left_text = "\n".join([left.title, left.body, *left.comments])
    right_text = "\n".join([right.title, right.body, *right.comments])
    if f"#{right.number}" in left_text or f"#{left.number}" in right_text:
        score += 4
        reasons.append("one PR directly references the other")

    shared_files = set(left.files) & set(right.files)
    if shared_files:
        score += min(2, len(shared_files))
        examples = ", ".join(sorted(shared_files)[:3])
        reasons.append(f"overlapping changed files: {examples}")

    shared_signatures = signature_terms(left) & signature_terms(right)
    if shared_signatures:
        score += min(4, 1 + len(shared_signatures))
        examples = ", ".join(sorted(shared_signatures)[:4])
        reasons.append(f"shared OpenViking surface: {examples}")

    title_similarity = jaccard(token_set(left.title), token_set(right.title))
    if title_similarity >= 0.42:
        score += 3
        reasons.append(f"similar titles ({title_similarity:.0%} token overlap)")

    combined_similarity = jaccard(
        token_set(f"{left.title}\n{left.body[:1200]}"),
        token_set(f"{right.title}\n{right.body[:1200]}"),
    )
    if combined_similarity >= 0.32:
        score += 2
        reasons.append(f"similar descriptions ({combined_similarity:.0%} token overlap)")

    if score >= 4:
        return DuplicateEdge(left.number, right.number, score, reasons)
    return None


def cluster_topic(prs: list[PullRequest], reasons: list[str]) -> str:
    text = "\n".join([pr.title + "\n" + pr.body for pr in prs] + reasons).lower()
    if "temp_upload" in text or "local_resource_upload" in text or "local resource upload" in text:
        return "Local resource upload / temp_upload"
    if "fallback" in text or "prefetch" in text:
        return "Fallback recall"
    if "auto" in text and "commit" in text:
        return "Auto-commit / searchability"
    if "toolset" in text or "reconnect" in text:
        return "Provider lifecycle and tool injection"
    return "OpenViking overlap"


def build_duplicate_clusters(prs: list[PullRequest]) -> list[DuplicateCluster]:
    by_number = {pr.number: pr for pr in prs}
    edges: list[DuplicateEdge] = []
    numbers = sorted(by_number)
    for index, left_number in enumerate(numbers):
        for right_number in numbers[index + 1 :]:
            edge = duplicate_reasons(by_number[left_number], by_number[right_number])
            if edge:
                edges.append(edge)

    adjacency: dict[int, set[int]] = {number: set() for number in numbers}
    edge_reasons: dict[frozenset[int], list[str]] = {}
    for edge in edges:
        adjacency[edge.left].add(edge.right)
        adjacency[edge.right].add(edge.left)
        edge_reasons[frozenset({edge.left, edge.right})] = edge.reasons

    clusters: list[DuplicateCluster] = []
    seen: set[int] = set()
    for number in numbers:
        if number in seen or not adjacency[number]:
            continue
        stack = [number]
        component: set[int] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(sorted(adjacency[current] - component))
        seen.update(component)
        prs_in_cluster = sorted((by_number[item] for item in component), key=sort_prs)
        reasons: list[str] = []
        for left in component:
            for right in adjacency[left] & component:
                if left < right:
                    reasons.extend(edge_reasons[frozenset({left, right})])
        unique_reasons = []
        for reason in reasons:
            if reason not in unique_reasons:
                unique_reasons.append(reason)
        clusters.append(
            DuplicateCluster(
                prs=prs_in_cluster,
                reasons=unique_reasons[:6],
                topic=cluster_topic(prs_in_cluster, unique_reasons),
            )
        )
    return sorted(clusters, key=lambda cluster: (cluster.prs[0].lifecycle_state != "open", -cluster.prs[0].number))


def sort_prs(pr: PullRequest) -> tuple[int, float, int]:
    state_rank = 0 if pr.lifecycle_state == "open" else 1
    updated_at = parse_github_time(pr.updated_at)
    updated_rank = -(updated_at.timestamp() if updated_at else 0.0)
    return (state_rank, updated_rank, -pr.number)


def pr_line(pr: PullRequest) -> str:
    state = pr.lifecycle_state
    draft = " draft" if pr.draft else ""
    updated = pr.updated_at[:10] if pr.updated_at else "unknown"
    labels = f" labels: {', '.join(pr.labels[:4])}" if pr.labels else ""
    return f"- [#{pr.number}]({pr.html_url}) `{state}{draft}` @{pr.author} updated {updated} - {pr.title}{labels}"


def render_deterministic_report(
    prs: list[PullRequest],
    clusters: list[DuplicateCluster],
    *,
    upstream_repo: str,
    recent_days: int,
    generated_at: datetime,
    llm_status: str,
) -> str:
    active = [pr for pr in prs if pr.lifecycle_state == "open"]
    recent_context = [pr for pr in prs if pr.lifecycle_state != "open"]
    clustered = {pr.number for cluster in clusters for pr in cluster.prs}
    unclustered_active = [pr for pr in active if pr.number not in clustered]
    stale = [
        pr
        for pr in active
        if (generated_at - (parse_github_time(pr.updated_at) or generated_at)) >= timedelta(days=21)
    ]

    lines = [
        "# OpenViking PR Report",
        "",
        f"Generated: {generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
        f"Upstream: `{upstream_repo}`",
        f"Scope: open PRs plus closed/merged PRs updated in the last {recent_days} days.",
        f"LLM: {llm_status}",
        "",
    ]

    if not prs:
        lines.extend(["No OpenViking-related PRs found.", ""])
        return "\n".join(lines)

    lines.extend(["## Likely Duplicate Groups", ""])
    if clusters:
        for cluster in clusters:
            lines.extend([f"### {cluster.topic}", ""])
            lines.extend(pr_line(pr) for pr in cluster.prs)
            if cluster.reasons:
                lines.append("")
                lines.append("Why this looks related:")
                lines.extend(f"- {reason}" for reason in cluster.reasons)
            lines.append("")
    else:
        lines.extend(["No likely duplicate groups found.", ""])

    lines.extend(["## Active PRs", ""])
    if active:
        if unclustered_active:
            lines.extend(pr_line(pr) for pr in sorted(unclustered_active, key=sort_prs))
        else:
            lines.append("All active matches are already represented in duplicate groups.")
    else:
        lines.append("No active OpenViking PRs found.")
    lines.append("")

    lines.extend(["## Stale Or Needs Attention", ""])
    if stale:
        lines.extend(pr_line(pr) for pr in sorted(stale, key=sort_prs))
    else:
        lines.append("No matched open PRs have gone 21+ days without updates.")
    lines.append("")

    lines.extend(["## Recent Merged Or Closed Context", ""])
    if recent_context:
        lines.extend(pr_line(pr) for pr in sorted(recent_context, key=sort_prs)[:20])
    else:
        lines.append("No recent merged or closed OpenViking PRs found.")
    lines.append("")

    lines.extend(
        [
            "---",
            "Duplicate detection is deterministic: shared linked issues, direct PR references, changed-file overlap, OpenViking tool/API terms, and title/body similarity.",
            "",
        ]
    )
    return "\n".join(lines)


def build_llm_payload(prs: list[PullRequest], clusters: list[DuplicateCluster]) -> dict[str, Any]:
    return {
        "pull_requests": [
            {
                "number": pr.number,
                "title": pr.title,
                "state": pr.lifecycle_state,
                "url": pr.html_url,
                "author": pr.author,
                "updated_at": pr.updated_at,
                "labels": pr.labels[:8],
                "head_ref": pr.head_ref,
                "linked_issues": sorted(pr.linked_issues)[:10],
                "files": pr.files[:20],
                "body_excerpt": textwrap.shorten(" ".join(pr.body.split()), width=900, placeholder="..."),
            }
            for pr in prs
        ],
        "duplicate_groups": [
            {
                "topic": cluster.topic,
                "prs": [pr.number for pr in cluster.prs],
                "reasons": cluster.reasons,
            }
            for cluster in clusters
        ],
    }


def chat_completions_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/chat/completions"):
        return trimmed
    return f"{trimmed}/chat/completions"


def enhance_with_llm(
    deterministic_report: str,
    payload: dict[str, Any],
    *,
    api_key: str,
    base_url: str,
    model: str,
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You write concise GitHub PR triage reports. Preserve PR numbers, links, "
                "states, and duplicate reasoning. Do not invent facts. Return markdown only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Rewrite this deterministic OpenViking PR report for a maintainer. Keep the "
                "same sections, mention likely duplicates clearly, and keep it concise.\n\n"
                f"Structured data:\n{json.dumps(payload, sort_keys=True)}\n\n"
                f"Deterministic report:\n{deterministic_report}"
            ),
        },
    ]
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
    with urllib.request.urlopen(req, timeout=60) as response:
        raw = response.read().decode("utf-8")
    data = json.loads(raw)
    content = data["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("LLM response did not include markdown content")
    return content.strip() + "\n"


def fetch_pull_files(client: GitHubClient, repo: str, number: int) -> list[str]:
    owner, name = split_repo(repo)
    files = client.paginate_list(f"/repos/{owner}/{name}/pulls/{number}/files")
    return [str(item.get("filename")) for item in files if item.get("filename")]


def fetch_issue_comments(client: GitHubClient, repo: str, number: int) -> list[str]:
    owner, name = split_repo(repo)
    comments = client.paginate_list(f"/repos/{owner}/{name}/issues/{number}/comments", limit=50)
    bodies = [str(comment.get("body") or "") for comment in comments]
    return [body for body in bodies if text_has_openviking_signal(body) or re.search(r"#\d+", body)]


def make_pull_request(
    pull: dict[str, Any],
    issue: dict[str, Any],
    files: list[str],
    comments: list[str],
) -> PullRequest:
    user = pull.get("user") or issue.get("user") or {}
    head = pull.get("head") or {}
    body = str(pull.get("body") or issue.get("body") or "")
    labels = label_names(issue) or label_names(pull)
    pr = PullRequest(
        number=int(pull["number"]),
        title=str(pull.get("title") or issue.get("title") or ""),
        body=body,
        state=str(pull.get("state") or issue.get("state") or "unknown"),
        html_url=str(pull.get("html_url") or issue.get("html_url") or ""),
        author=str(user.get("login") or "unknown"),
        created_at=str(pull.get("created_at") or issue.get("created_at") or ""),
        updated_at=str(pull.get("updated_at") or issue.get("updated_at") or ""),
        closed_at=pull.get("closed_at") or issue.get("closed_at"),
        merged_at=pull.get("merged_at"),
        draft=bool(pull.get("draft")),
        head_ref=str(head.get("ref") or ""),
        labels=labels,
        files=files,
        comments=comments,
    )
    pr.linked_issues = extract_issue_refs(pr.body, *pr.comments) - {pr.number}
    return pr


def fetch_pull_request(
    client: GitHubClient,
    repo: str,
    number: int,
    *,
    preloaded_files: list[str] | None = None,
) -> PullRequest:
    owner, name = split_repo(repo)
    pull = client.request("GET", f"/repos/{owner}/{name}/pulls/{number}")
    issue = client.request("GET", f"/repos/{owner}/{name}/issues/{number}")
    files = preloaded_files if preloaded_files is not None else fetch_pull_files(client, repo, number)
    comments = fetch_issue_comments(client, repo, number)
    return make_pull_request(pull, issue, files, comments)


def collect_candidate_numbers(
    client: GitHubClient,
    repo: str,
    *,
    recent_days: int,
    max_open_prs: int,
    max_closed_prs: int,
    file_probe_limit: int,
) -> tuple[set[int], dict[int, list[str]]]:
    owner, name = split_repo(repo)
    cutoff = datetime.now(UTC) - timedelta(days=recent_days)
    numbers: set[int] = set()
    preloaded_files: dict[int, list[str]] = {}

    for term in SEARCH_TERMS:
        quoted = f'"{term}"' if " " in term or "://" in term else term
        for state_query in ("is:open", f"is:closed updated:>={cutoff.date()}"):
            query = f"repo:{repo} is:pr {state_query} {quoted}"
            for item in client.search_issues(query):
                if item.get("number"):
                    numbers.add(int(item["number"]))

    scanned = 0
    file_probes = 0
    for state, limit in (("open", max_open_prs), ("closed", max_closed_prs)):
        pulls = client.paginate_list(
            f"/repos/{owner}/{name}/pulls",
            params={"state": state, "sort": "updated", "direction": "desc"},
            limit=limit,
        )
        for raw in pulls:
            updated_at = parse_github_time(raw.get("updated_at"))
            if state == "closed" and updated_at and updated_at < cutoff:
                continue
            number = int(raw["number"])
            scanned += 1
            if raw_pr_metadata_has_signal(raw):
                numbers.add(number)
                continue
            if file_probes >= file_probe_limit:
                continue
            files = fetch_pull_files(client, repo, number)
            file_probes += 1
            if paths_have_openviking_signal(files):
                numbers.add(number)
                preloaded_files[number] = files

    print(
        f"Discovered {len(numbers)} candidate PRs after scanning {scanned} PR records "
        f"and probing files for {file_probes} PRs.",
        file=sys.stderr,
    )
    return numbers, preloaded_files


def collect_pull_requests(
    client: GitHubClient,
    repo: str,
    *,
    recent_days: int,
    max_open_prs: int,
    max_closed_prs: int,
    file_probe_limit: int,
) -> list[PullRequest]:
    numbers, preloaded_files = collect_candidate_numbers(
        client,
        repo,
        recent_days=recent_days,
        max_open_prs=max_open_prs,
        max_closed_prs=max_closed_prs,
        file_probe_limit=file_probe_limit,
    )
    cutoff = datetime.now(UTC) - timedelta(days=recent_days)
    prs: list[PullRequest] = []
    for number in sorted(numbers, reverse=True):
        pr = fetch_pull_request(client, repo, number, preloaded_files=preloaded_files.get(number))
        updated_at = parse_github_time(pr.updated_at)
        if pr.lifecycle_state != "open" and updated_at and updated_at < cutoff:
            continue
        if pr_has_openviking_signal(pr):
            prs.append(pr)
    return sorted(prs, key=sort_prs)


def find_report_issue(client: GitHubClient, report_repo: str, title: str) -> dict[str, Any] | None:
    owner, name = split_repo(report_repo)
    issues = client.paginate_list(
        f"/repos/{owner}/{name}/issues",
        params={"state": "open", "sort": "updated", "direction": "desc"},
        limit=100,
    )
    for issue in issues:
        if issue.get("title") == title and "pull_request" not in issue:
            return issue
    return None


def publish_report(
    client: GitHubClient,
    *,
    report_repo: str,
    title: str,
    body: str,
    post_comment: bool,
) -> str:
    owner, name = split_repo(report_repo)
    issue = find_report_issue(client, report_repo, title)
    if issue is None:
        issue = client.request(
            "POST",
            f"/repos/{owner}/{name}/issues",
            data={"title": title, "body": body},
        )
    else:
        client.request(
            "PATCH",
            f"/repos/{owner}/{name}/issues/{issue['number']}",
            data={"body": body},
        )
    if post_comment:
        run_url = github_run_url()
        footer = f"\n\n---\nGenerated by GitHub Actions"
        if run_url:
            footer += f": {run_url}"
        client.request(
            "POST",
            f"/repos/{owner}/{name}/issues/{issue['number']}/comments",
            data={"body": body + footer},
        )
    return str(issue.get("html_url") or f"https://github.com/{report_repo}/issues/{issue['number']}")


def github_run_url() -> str | None:
    server = os.getenv("GITHUB_SERVER_URL")
    repo = os.getenv("GITHUB_REPOSITORY")
    run_id = os.getenv("GITHUB_RUN_ID")
    if server and repo and run_id:
        return f"{server}/{repo}/actions/runs/{run_id}"
    return None


def build_report(
    prs: list[PullRequest],
    *,
    upstream_repo: str,
    recent_days: int,
    generated_at: datetime,
) -> tuple[str, str]:
    clusters = build_duplicate_clusters(prs)
    llm_status = "not configured"
    deterministic = render_deterministic_report(
        prs,
        clusters,
        upstream_repo=upstream_repo,
        recent_days=recent_days,
        generated_at=generated_at,
        llm_status=llm_status,
    )

    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "")
    model = os.getenv("LLM_MODEL", "")
    if not (api_key and base_url and model):
        return deterministic, llm_status

    try:
        enhanced = enhance_with_llm(
            deterministic,
            build_llm_payload(prs, clusters),
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
    except Exception as exc:  # noqa: BLE001 - report generation should survive LLM outages.
        llm_status = f"configured but skipped after error: {exc}"
        return render_deterministic_report(
            prs,
            clusters,
            upstream_repo=upstream_repo,
            recent_days=recent_days,
            generated_at=generated_at,
            llm_status=llm_status,
        ), llm_status

    llm_status = f"enhanced with `{model}`"
    enhanced = re.sub(r"^LLM: .*$", f"LLM: {llm_status}", enhanced, flags=re.MULTILINE)
    if "LLM:" not in enhanced:
        enhanced = enhanced.rstrip() + f"\n\nLLM: {llm_status}\n"
    return enhanced, llm_status


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-repo", default=os.getenv("UPSTREAM_REPOSITORY", DEFAULT_UPSTREAM_REPO))
    parser.add_argument("--report-repo", default=os.getenv("REPORT_REPOSITORY") or os.getenv("GITHUB_REPOSITORY"))
    parser.add_argument("--report-title", default=os.getenv("REPORT_ISSUE_TITLE", DEFAULT_REPORT_TITLE))
    parser.add_argument("--recent-days", type=int, default=int(os.getenv("RECENT_DAYS", "14")))
    parser.add_argument("--max-open-prs", type=int, default=int(os.getenv("MAX_OPEN_PRS", "300")))
    parser.add_argument("--max-closed-prs", type=int, default=int(os.getenv("MAX_CLOSED_PRS", "300")))
    parser.add_argument("--file-probe-limit", type=int, default=int(os.getenv("FILE_PROBE_LIMIT", "150")))
    parser.add_argument("--output", default=os.getenv("REPORT_OUTPUT", "openviking-pr-report.md"))
    parser.add_argument("--dry-run", action="store_true", help="Print and write the report without updating GitHub issues")
    parser.add_argument("--no-comment", action="store_true", help="Update the tracking issue body without posting a comment")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    write_token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    read_token = os.getenv("UPSTREAM_GITHUB_TOKEN") or write_token or ""
    if not write_token and not args.dry_run:
        print("GITHUB_TOKEN or GH_TOKEN is required when not using --dry-run.", file=sys.stderr)
        return 2
    if not args.report_repo and not args.dry_run:
        print("REPORT_REPOSITORY or GITHUB_REPOSITORY is required when not using --dry-run.", file=sys.stderr)
        return 2

    read_client = GitHubClient(read_token, fallback_to_unauth=True)
    if not read_token:
        args.file_probe_limit = min(args.file_probe_limit, 30)
        print(
            "UPSTREAM_GITHUB_TOKEN is not set; reading public upstream data without authentication "
            f"and limiting file probes to {args.file_probe_limit}.",
            file=sys.stderr,
        )
    elif not os.getenv("UPSTREAM_GITHUB_TOKEN"):
        print("UPSTREAM_GITHUB_TOKEN is not set; using GITHUB_TOKEN/GH_TOKEN for upstream reads.", file=sys.stderr)

    generated_at = datetime.now(UTC)
    prs = collect_pull_requests(
        read_client,
        args.upstream_repo,
        recent_days=args.recent_days,
        max_open_prs=args.max_open_prs,
        max_closed_prs=args.max_closed_prs,
        file_probe_limit=args.file_probe_limit,
    )
    report, llm_status = build_report(
        prs,
        upstream_repo=args.upstream_repo,
        recent_days=args.recent_days,
        generated_at=generated_at,
    )
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report)
    print(f"LLM status: {llm_status}", file=sys.stderr)

    if args.dry_run:
        return 0
    write_client = GitHubClient(write_token)
    issue_url = publish_report(
        write_client,
        report_repo=args.report_repo,
        title=args.report_title,
        body=report,
        post_comment=not args.no_comment,
    )
    print(f"Updated report issue: {issue_url}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
