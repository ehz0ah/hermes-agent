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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from typing import Any


DEFAULT_UPSTREAM_REPO = "NousResearch/hermes-agent"
DEFAULT_REPORT_TITLE = "OpenViking PR Report"
SINGAPORE_TZ = timezone(timedelta(hours=8))
OPENVIKING_TERMS = (
    "openviking",
    "open viking",
    "viking_",
    "viking://",
)
SEARCH_TERMS = (
    "openviking",
    "open viking",
    "viking_",
    "viking://",
)
OPENVIKING_PATH_PREFIXES = (
    "plugins/memory/openviking",
    "tests/plugins/memory/test_openviking",
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
TOPIC_ORDER = (
    "Local resource upload",
    "Read/file URI routing",
    "Explicit fallback / remember recall",
    "Tool surface expansion",
    "Provider lifecycle and configuration",
    "API endpoint migration",
    "Auto-commit / searchability",
    "Other OpenViking overlap",
)


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


@dataclass
class LlmPrAssessment:
    pr: PullRequest
    is_openviking_related: bool
    confidence: str
    summary: str
    why: str
    duplicate_group: str = ""
    topic: str = ""


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
    return any(any(path.startswith(prefix) for prefix in OPENVIKING_PATH_PREFIXES) for path in lowered)


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
    text = "\n".join([pr.title, pr.body, pr.head_ref, " ".join(pr.labels)])
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
    supporting_reasons: list[str] = []
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
        examples = ", ".join(sorted(shared_files)[:3])
        supporting_reasons.append(f"overlapping changed files: {examples}")

    shared_signatures = signature_terms(left) & signature_terms(right)
    if shared_signatures:
        score += min(3, len(shared_signatures))
        examples = ", ".join(sorted(shared_signatures)[:4])
        reasons.append(f"shared OpenViking surface: {examples}")

    title_similarity = jaccard(token_set(left.title), token_set(right.title))
    if title_similarity >= 0.50:
        score += 3
        reasons.append(f"similar titles ({title_similarity:.0%} token overlap)")

    combined_similarity = jaccard(
        token_set(f"{left.title}\n{left.body[:1200]}"),
        token_set(f"{right.title}\n{right.body[:1200]}"),
    )
    if combined_similarity >= 0.40:
        score += 2
        reasons.append(f"similar descriptions ({combined_similarity:.0%} token overlap)")

    if score >= 4:
        reasons.extend(reason for reason in supporting_reasons if reason not in reasons)
        return DuplicateEdge(left.number, right.number, score, reasons)
    return None


def cluster_topic(prs: list[PullRequest], reasons: list[str]) -> str:
    text = "\n".join([pr.title + "\n" + pr.body for pr in prs] + reasons).lower()
    if "local_resource_upload" in text or "local resource upload" in text:
        return "Local resource upload"
    if "fallback" in text or "prefetch" in text:
        return "Fallback recall"
    if "auto" in text and "commit" in text:
        return "Auto-commit / searchability"
    if "toolset" in text or "reconnect" in text:
        return "Provider lifecycle and tool injection"
    return "OpenViking overlap"


def topic_tags(pr: PullRequest) -> set[str]:
    text = "\n".join([pr.title, pr.body, *pr.comments]).lower()
    signatures = signature_terms(pr)
    tags: set[str] = set()
    if "local_resource_upload" in signatures:
        tags.add("Local resource upload")
    if "viking_read" in text or "file uri" in text or "content/read" in text or "overview.md" in text:
        tags.add("Read/file URI routing")
    if "fallback" in text or "explicit remember" in text or "prefetch" in text:
        tags.add("Explicit fallback / remember recall")
    if any(term in text for term in ("viking_delete", "viking_write", "viking_link", "viking_grep", "viking_glob", "viking_extract")):
        tags.add("Tool surface expansion")
    if any(term in text for term in ("toolset", "reconnect", "reachability", "identity headers", "user api key", "startup warning")):
        tags.add("Provider lifecycle and configuration")
    if "endpoint" in text or "v0.2" in text or "v0.3" in text:
        tags.add("API endpoint migration")
    if "auto-commit" in text or "auto commit" in text:
        tags.add("Auto-commit / searchability")
    return tags or {"Other OpenViking overlap"}


def primary_topic(pr: PullRequest) -> str:
    title = pr.title.lower()
    if "fallback" in title or "explicit remember" in title:
        return "Explicit fallback / remember recall"
    if "local" in title and ("upload" in title or "resource" in title):
        return "Local resource upload"
    if "viking_read" in title or "file uri" in title or "content/read" in title:
        return "Read/file URI routing"
    if "endpoint" in title or "v0.2" in title or "v0.3" in title:
        return "API endpoint migration"
    if "delete" in title or "expand openviking" in title or "memory tools" in title:
        return "Tool surface expansion"
    if "toolset" in title or "reconnect" in title or "warning" in title or "headers" in title:
        return "Provider lifecycle and configuration"
    if "auto-commit" in title or "auto commit" in title:
        return "Auto-commit / searchability"

    tags = topic_tags(pr)
    for topic in TOPIC_ORDER:
        if topic in tags:
            return topic
    return "Other OpenViking overlap"


def cluster_reasons(topic: str, prs: list[PullRequest]) -> list[str]:
    reasons = [f"shared topic: {topic}"]
    ref_counts: dict[int, int] = {}
    signature_counts: dict[str, int] = {}
    for pr in prs:
        for ref in pr.linked_issues:
            ref_counts[ref] = ref_counts.get(ref, 0) + 1
        for signature in signature_terms(pr):
            signature_counts[signature] = signature_counts.get(signature, 0) + 1

    common_refs = sorted(ref for ref, count in ref_counts.items() if count >= 2)
    if common_refs:
        reasons.append("shared linked issues/references: " + ", ".join(f"#{ref}" for ref in common_refs[:5]))

    common_signatures = sorted(signature for signature, count in signature_counts.items() if count >= 2)
    if common_signatures:
        reasons.append("common OpenViking terms: " + ", ".join(common_signatures[:5]))
    return reasons


def build_duplicate_clusters(prs: list[PullRequest]) -> list[DuplicateCluster]:
    grouped: dict[str, list[PullRequest]] = {}
    for pr in prs:
        grouped.setdefault(primary_topic(pr), []).append(pr)

    clusters = [
        DuplicateCluster(
            prs=sorted(topic_prs, key=sort_prs),
            reasons=cluster_reasons(topic, topic_prs),
            topic=topic,
        )
        for topic, topic_prs in grouped.items()
        if len(topic_prs) > 1 and topic != "Other OpenViking overlap"
    ]
    return sorted(clusters, key=lambda cluster: TOPIC_ORDER.index(cluster.topic))


def build_duplicate_clusters_by_edges(prs: list[PullRequest]) -> list[DuplicateCluster]:
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


def table_cell(text: str, *, width: int = 220) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) > width:
        compact = textwrap.shorten(compact, width=width, placeholder="...")
    return compact.replace("|", "\\|")


def confidence_rank(confidence: str) -> int:
    normalized = confidence.lower()
    if normalized == "high":
        return 0
    if normalized == "medium":
        return 1
    return 2


def sort_assessments(assessment: LlmPrAssessment) -> tuple[int, tuple[int, float, int]]:
    return (confidence_rank(assessment.confidence), sort_prs(assessment.pr))


def deterministic_openviking_reason(pr: PullRequest) -> str:
    text = "\n".join([pr.title, pr.body, pr.head_ref, " ".join(pr.labels)]).lower()
    matched_terms = [term for term in OPENVIKING_TERMS if term in text]
    matched_paths = [
        path
        for path in pr.files
        if any(path.lower().startswith(prefix) for prefix in OPENVIKING_PATH_PREFIXES)
    ]
    reasons: list[str] = []
    if matched_terms:
        reasons.append("matched OpenViking terms: " + ", ".join(matched_terms[:4]))
    if matched_paths:
        reasons.append("touched OpenViking plugin/test paths: " + ", ".join(matched_paths[:3]))
    return "; ".join(reasons) or "matched deterministic OpenViking signal"


def deterministic_assessment(pr: PullRequest) -> LlmPrAssessment:
    topic = primary_topic(pr)
    duplicate_group = "" if topic == "Other OpenViking overlap" else topic
    return LlmPrAssessment(
        pr=pr,
        is_openviking_related=True,
        confidence="high",
        summary=pr.title,
        why="Deterministic guardrail: " + deterministic_openviking_reason(pr),
        duplicate_group=duplicate_group,
        topic=topic,
    )


def merge_with_deterministic_matches(
    assessments: list[LlmPrAssessment],
    prs: list[PullRequest],
) -> tuple[list[LlmPrAssessment], int]:
    by_number = {assessment.pr.number: assessment for assessment in assessments}
    added = 0
    for pr in prs:
        if pr.number in by_number or not pr_has_openviking_signal(pr):
            continue
        by_number[pr.number] = deterministic_assessment(pr)
        added += 1
    return sorted(by_number.values(), key=sort_assessments), added


def render_deterministic_report(
    prs: list[PullRequest],
    clusters: list[DuplicateCluster],
    *,
    upstream_repo: str,
    recent_hours: int,
    generated_at: datetime,
    llm_status: str,
) -> str:
    active = prs
    clustered = {pr.number for cluster in clusters for pr in cluster.prs}
    unclustered_active = [pr for pr in active if pr.number not in clustered]

    lines = [
        "# OpenViking PR Report",
        "",
        f"Generated: {generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
        f"Upstream: `{upstream_repo}`",
        f"Scope: open OpenViking-related PRs updated in the last {recent_hours} hours.",
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

    lines.extend(
        [
            "---",
            "Duplicate grouping is deterministic and based on shared OpenViking topics and references.",
            "",
        ]
    )
    return "\n".join(lines)


def build_llm_payload(prs: list[PullRequest]) -> dict[str, Any]:
    return {
        "candidate_pull_requests": [
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
                "files": pr.files[:15],
                "body_excerpt": textwrap.shorten(" ".join(pr.body.split()), width=320, placeholder="..."),
                "openviking_signal": pr_has_openviking_signal(pr),
                "comments": [
                    textwrap.shorten(" ".join(comment.split()), width=160, placeholder="...")
                    for comment in pr.comments[:2]
                ],
            }
            for pr in prs
        ],
    }


def chat_completions_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/chat/completions"):
        return trimmed
    return f"{trimmed}/chat/completions"


def extract_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise RuntimeError("LLM response must be a JSON object")
    return parsed


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1"}
    return bool(value)


def normalize_confidence(value: Any) -> str:
    lowered = str(value or "").strip().lower()
    if lowered in {"high", "medium", "low"}:
        return lowered
    return "medium"


def parse_llm_assessments(data: dict[str, Any], prs: list[PullRequest]) -> list[LlmPrAssessment]:
    by_number = {pr.number: pr for pr in prs}
    raw_items = data.get("pull_requests")
    force_related = False
    if not isinstance(raw_items, list):
        raw_items = data.get("related_prs")
        force_related = isinstance(raw_items, list)
    if not isinstance(raw_items, list):
        raise RuntimeError("LLM response missing pull_requests list")

    assessments: list[LlmPrAssessment] = []
    seen: set[int] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        try:
            number = int(item.get("number"))
        except (TypeError, ValueError):
            continue
        if number not in by_number or number in seen:
            continue
        seen.add(number)
        is_related = True if force_related else normalize_bool(item.get("is_openviking_related"))
        if not is_related:
            continue
        summary = str(item.get("summary") or by_number[number].title).strip()
        why = str(item.get("why") or item.get("reason") or "Classified as OpenViking-related by the LLM.").strip()
        assessments.append(
            LlmPrAssessment(
                pr=by_number[number],
                is_openviking_related=True,
                confidence=normalize_confidence(item.get("confidence")),
                summary=summary,
                why=why,
                duplicate_group=str(item.get("duplicate_group") or "").strip(),
                topic=str(item.get("topic") or "").strip(),
            )
        )
    return sorted(assessments, key=sort_assessments)


def render_llm_report(
    assessments: list[LlmPrAssessment],
    *,
    reviewed_count: int,
    upstream_repo: str,
    recent_hours: int,
    generated_at: datetime,
    llm_status: str,
) -> str:
    lines = [
        "# OpenViking PR Report",
        "",
        f"Generated: {generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
        f"Upstream: `{upstream_repo}`",
        f"Scope: open PRs updated in the last {recent_hours} hours, classified for OpenViking relevance.",
        f"Reviewed: {reviewed_count} open PRs",
        f"LLM: {llm_status}",
        "",
    ]

    if not assessments:
        lines.extend(["No OpenViking-related PRs found.", ""])
        return "\n".join(lines)

    lines.extend(
        [
            "## OpenViking Matches",
            "",
            "| PR | Confidence | Topic / duplicate group | Summary | Why included |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for assessment in assessments:
        pr = assessment.pr
        topic = assessment.duplicate_group or assessment.topic or primary_topic(pr)
        lines.append(
            "| "
            f"[#{pr.number}]({pr.html_url}) "
            f"| {table_cell(assessment.confidence.title(), width=20)} "
            f"| {table_cell(topic, width=90)} "
            f"| {table_cell(assessment.summary)} "
            f"| {table_cell(assessment.why)} |"
        )
    lines.append("")

    groups: dict[str, list[LlmPrAssessment]] = {}
    for assessment in assessments:
        group_name = assessment.duplicate_group.strip()
        if group_name:
            groups.setdefault(group_name, []).append(assessment)

    duplicate_groups = [(name, items) for name, items in groups.items() if len(items) > 1]
    lines.extend(["## Likely Duplicate Groups", ""])
    if duplicate_groups:
        for name, items in sorted(duplicate_groups, key=lambda entry: (-len(entry[1]), entry[0].lower())):
            links = ", ".join(f"[#{item.pr.number}]({item.pr.html_url})" for item in sorted(items, key=sort_assessments))
            reasons = "; ".join(table_cell(item.why, width=140) for item in items[:3])
            lines.extend([f"### {name}", "", f"- PRs: {links}", f"- Why: {reasons}", ""])
    else:
        lines.extend(["No likely duplicate groups found.", ""])

    lines.extend(
        [
            "---",
            "GitHub API supplied the candidate PRs; the LLM classified OpenViking relevance and wrote per-PR summaries.",
            "",
        ]
    )
    return "\n".join(lines)


def classify_llm_batch(
    prs: list[PullRequest],
    *,
    api_key: str,
    base_url: str,
    model: str,
    upstream_repo: str,
    timeout_seconds: int,
) -> list[LlmPrAssessment]:
    payload = build_llm_payload(prs)
    messages = [
        {
            "role": "system",
            "content": (
                "You classify GitHub pull requests for a maintainer. Return JSON only. "
                "Only use PR numbers present in the provided candidate_pull_requests list. "
                "Do not invent PRs, file paths, labels, or facts."
            ),
        },
        {
            "role": "user",
            "content": (
                "OpenViking is the Hermes memory/context database plugin, centered on "
                "`plugins/memory/openviking`. It may expose or change `viking_*` tools, "
                "`viking://` resources, OpenViking provider setup, OpenViking API endpoints, "
                "memory recall/search behavior, local resource handling for OpenViking, or tests "
                "under `tests/plugins/memory/test_openviking`. Generic memory, provider, upload, "
                "or CLI changes are not OpenViking-related unless they explicitly touch this plugin, "
                "its paths, APIs, tools, resources, or integration behavior.\n\n"
                f"Classify open PRs from `{upstream_repo}`. Review every candidate, but return only "
                "PRs that are OpenViking-related. Return a JSON object with this exact shape:\n"
                "{\n"
                '  "pull_requests": [\n'
                "    {\n"
                '      "number": 123,\n'
                '      "is_openviking_related": true,\n'
                '      "confidence": "high|medium|low",\n'
                '      "topic": "short topic",\n'
                '      "summary": "one sentence maintainer-facing summary",\n'
                '      "why": "brief evidence from title/body/files/labels",\n'
                '      "duplicate_group": "shared group name or empty string"\n'
                "    }\n"
                "  ]\n"
                "}\n\n"
                "Use duplicate_group only when two or more included PRs appear to solve or modify "
                "the same OpenViking behavior. Omit rejected PRs entirely. Keep summaries concise.\n\n"
                f"Candidate data:\n{json.dumps(payload, sort_keys=True)}"
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
    with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
        raw = response.read().decode("utf-8")
    data = json.loads(raw)
    content = data["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("LLM response did not include JSON content")
    return parse_llm_assessments(extract_json_object(content), prs)


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
        raise RuntimeError("LLM response did not include content")
    return content.strip() + "\n"


def format_report_with_llm(
    report: str,
    assessments: list[LlmPrAssessment],
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout_seconds: int,
) -> str:
    payload = {
        "openviking_matches": [
            {
                "number": assessment.pr.number,
                "url": assessment.pr.html_url,
                "title": assessment.pr.title,
                "confidence": assessment.confidence,
                "topic": assessment.topic,
                "duplicate_group": assessment.duplicate_group,
                "summary": assessment.summary,
                "why": assessment.why,
            }
            for assessment in assessments
        ],
        "draft_report": report,
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You format concise GitHub PR triage reports in Markdown. Preserve every PR number, "
                "link, confidence, and factual summary from the input. Do not invent PRs or omit PRs."
            ),
        },
        {
            "role": "user",
            "content": (
                "Format this OpenViking PR report for a maintainer. Keep the table. Improve wording "
                "and duplicate-group explanations if useful, but include every PR in openviking_matches "
                "exactly once in the main table.\n\n"
                f"Input JSON:\n{json.dumps(payload, sort_keys=True)}"
            ),
        },
    ]
    formatted = llm_chat_content(
        messages,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
    )
    missing = [assessment.pr.number for assessment in assessments if f"#{assessment.pr.number}" not in formatted]
    if missing:
        raise RuntimeError("LLM formatted report omitted PRs: " + ", ".join(f"#{number}" for number in missing))
    return formatted


def int_from_env(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, value)


def classify_with_llm(
    prs: list[PullRequest],
    *,
    api_key: str,
    base_url: str,
    model: str,
    upstream_repo: str,
) -> list[LlmPrAssessment]:
    batch_size = int_from_env("LLM_BATCH_SIZE", 20)
    concurrency = int_from_env("LLM_CONCURRENCY", 20)
    timeout_seconds = int_from_env("LLM_TIMEOUT_SECONDS", 120, minimum=30)
    batches = [prs[start : start + batch_size] for start in range(0, len(prs), batch_size)]
    assessments: list[LlmPrAssessment] = []
    errors: list[Exception] = []
    if not batches:
        return []

    with ThreadPoolExecutor(max_workers=min(concurrency, len(batches))) as executor:
        futures = [
            executor.submit(
                classify_llm_batch,
                batch,
                api_key=api_key,
                base_url=base_url,
                model=model,
                upstream_repo=upstream_repo,
                timeout_seconds=timeout_seconds,
            )
            for batch in batches
        ]
        for future in as_completed(futures):
            try:
                assessments.extend(future.result())
            except Exception as exc:  # noqa: BLE001 - one LLM batch should not discard successful batches.
                errors.append(exc)

    if errors:
        print(f"{len(errors)} LLM classification batch(es) failed.", file=sys.stderr)
        if not assessments:
            raise errors[0]

    by_number: dict[int, LlmPrAssessment] = {}
    for assessment in assessments:
        current = by_number.get(assessment.pr.number)
        if current is None or confidence_rank(assessment.confidence) < confidence_rank(current.confidence):
            by_number[assessment.pr.number] = assessment
    return sorted(by_number.values(), key=sort_assessments)


def fetch_pull_files(client: GitHubClient, repo: str, number: int) -> list[str]:
    owner, name = split_repo(repo)
    files = client.paginate_list(f"/repos/{owner}/{name}/pulls/{number}/files")
    return [str(item.get("filename")) for item in files if item.get("filename")]


def fetch_issue_comments(client: GitHubClient, repo: str, number: int) -> list[str]:
    owner, name = split_repo(repo)
    comments = client.paginate_list(f"/repos/{owner}/{name}/issues/{number}/comments", limit=50)
    bodies = [str(comment.get("body") or "") for comment in comments]
    return [body for body in bodies if text_has_openviking_signal(body)]


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
    recent_hours: int,
    max_open_prs: int,
    file_probe_limit: int,
) -> tuple[set[int], dict[int, list[str]]]:
    owner, name = split_repo(repo)
    cutoff = datetime.now(UTC) - timedelta(hours=recent_hours)
    numbers: set[int] = set()
    preloaded_files: dict[int, list[str]] = {}

    for term in SEARCH_TERMS:
        quoted = f'"{term}"' if " " in term or "://" in term else term
        query = f"repo:{repo} is:pr is:open updated:>={cutoff.isoformat(timespec='seconds')} {quoted}"
        for item in client.search_issues(query):
            if item.get("number"):
                numbers.add(int(item["number"]))

    scanned = 0
    file_probes = 0
    pulls = client.paginate_list(
        f"/repos/{owner}/{name}/pulls",
        params={"state": "open", "sort": "updated", "direction": "desc"},
        limit=max_open_prs,
    )
    for raw in pulls:
        updated_at = parse_github_time(raw.get("updated_at"))
        if updated_at and updated_at < cutoff:
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


def collect_search_candidate_numbers(client: GitHubClient, repo: str, *, recent_hours: int) -> set[int]:
    cutoff = datetime.now(UTC) - timedelta(hours=recent_hours)
    numbers: set[int] = set()
    for term in SEARCH_TERMS:
        quoted = f'"{term}"' if " " in term or "://" in term else term
        query = f"repo:{repo} is:pr is:open updated:>={cutoff.isoformat(timespec='seconds')} {quoted}"
        for item in client.search_issues(query):
            if item.get("number"):
                numbers.add(int(item["number"]))
    return numbers


def fetch_recent_open_pull_records(
    client: GitHubClient,
    repo: str,
    *,
    recent_hours: int,
    max_open_prs: int,
) -> list[dict[str, Any]]:
    owner, name = split_repo(repo)
    cutoff = datetime.now(UTC) - timedelta(hours=recent_hours)
    pulls: list[dict[str, Any]] = []
    page = 1
    reached_cutoff = False

    while len(pulls) < max_open_prs and not reached_cutoff:
        items = client.request(
            "GET",
            f"/repos/{owner}/{name}/pulls",
            params={"state": "open", "sort": "updated", "direction": "desc", "per_page": 100, "page": page},
        )
        if not isinstance(items, list):
            raise GitHubApiError(f"Expected list response from /repos/{owner}/{name}/pulls")
        if not items:
            break

        for raw in items:
            updated_at = parse_github_time(raw.get("updated_at"))
            if updated_at and updated_at < cutoff:
                reached_cutoff = True
                break
            pulls.append(raw)
            if len(pulls) >= max_open_prs:
                break

        if len(items) < 100:
            break
        page += 1

    if len(pulls) >= max_open_prs:
        print(f"Stopped recent PR scan after safety cap of {max_open_prs} open PRs.", file=sys.stderr)
    return pulls


def collect_pull_requests(
    client: GitHubClient,
    repo: str,
    *,
    recent_hours: int,
    max_open_prs: int,
    file_probe_limit: int,
) -> list[PullRequest]:
    cutoff = datetime.now(UTC) - timedelta(hours=recent_hours)
    pulls = fetch_recent_open_pull_records(
        client,
        repo,
        recent_hours=recent_hours,
        max_open_prs=max_open_prs,
    )

    prs_by_number: dict[int, PullRequest] = {}
    for raw in pulls:
        number = int(raw["number"])
        pr = make_pull_request(raw, raw, [], [])
        if pr.lifecycle_state == "open":
            prs_by_number[pr.number] = pr

    search_numbers = collect_search_candidate_numbers(client, repo, recent_hours=recent_hours)
    enriched = 0
    for number in sorted(search_numbers, reverse=True):
        if enriched >= file_probe_limit:
            break
        pr = fetch_pull_request(client, repo, number)
        enriched += 1
        updated_at = parse_github_time(pr.updated_at)
        if pr.lifecycle_state == "open" and (not updated_at or updated_at >= cutoff):
            prs_by_number[pr.number] = pr

    print(
        f"Collected {len(prs_by_number)} recent open PRs after scanning {len(pulls)} lightweight PR records "
        f"and enriching {enriched} OpenViking search hit(s).",
        file=sys.stderr,
    )
    return sorted(prs_by_number.values(), key=sort_prs)


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


def dated_report_title(base_title: str, generated_at: datetime) -> str:
    local_date = generated_at.astimezone(SINGAPORE_TZ).strftime("%Y-%m-%d")
    if "%Y" in base_title:
        return generated_at.astimezone(SINGAPORE_TZ).strftime(base_title)
    if re.search(r"\d{4}-\d{2}-\d{2}$", base_title):
        return base_title
    return f"{base_title} - {local_date}"


def build_report(
    prs: list[PullRequest],
    *,
    upstream_repo: str,
    recent_hours: int,
    generated_at: datetime,
) -> tuple[str, str]:
    llm_status = "not configured"
    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "")
    model = os.getenv("LLM_MODEL", "")
    if api_key and base_url and model:
        try:
            assessments = classify_with_llm(
                prs,
                api_key=api_key,
                base_url=base_url,
                model=model,
                upstream_repo=upstream_repo,
            )
        except Exception as exc:  # noqa: BLE001 - report generation should survive LLM outages.
            llm_status = f"configured but skipped after error: {exc}"
        else:
            assessments, deterministic_added = merge_with_deterministic_matches(assessments, prs)
            llm_status = f"classified with `{model}`"
            if deterministic_added:
                llm_status += f"; deterministic guardrail added {deterministic_added} match(es)"
            report = render_llm_report(
                assessments,
                reviewed_count=len(prs),
                upstream_repo=upstream_repo,
                recent_hours=recent_hours,
                generated_at=generated_at,
                llm_status=llm_status,
            )
            try:
                report = format_report_with_llm(
                    report,
                    assessments,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    timeout_seconds=int_from_env("LLM_TIMEOUT_SECONDS", 120, minimum=30),
                )
            except Exception as exc:  # noqa: BLE001 - the rendered table is a valid fallback.
                print(f"Final LLM formatting skipped: {exc}", file=sys.stderr)
            else:
                llm_status += "; final report formatted with LLM"
                report = re.sub(r"^LLM: .*$", f"LLM: {llm_status}", report, flags=re.MULTILINE)
                if "LLM:" not in report:
                    report = report.rstrip() + f"\n\nLLM: {llm_status}\n"
            return (report, llm_status)

    deterministic_prs = [pr for pr in prs if pr_has_openviking_signal(pr)]
    clusters = build_duplicate_clusters(deterministic_prs)
    return (
        render_deterministic_report(
            deterministic_prs,
            clusters,
            upstream_repo=upstream_repo,
            recent_hours=recent_hours,
            generated_at=generated_at,
            llm_status=llm_status,
        ),
        llm_status,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-repo", default=os.getenv("UPSTREAM_REPOSITORY", DEFAULT_UPSTREAM_REPO))
    parser.add_argument("--report-repo", default=os.getenv("REPORT_REPOSITORY") or os.getenv("GITHUB_REPOSITORY"))
    parser.add_argument("--report-title", default=os.getenv("REPORT_ISSUE_TITLE", DEFAULT_REPORT_TITLE))
    parser.add_argument("--recent-hours", type=int, default=int(os.getenv("RECENT_HOURS", "24")))
    parser.add_argument("--max-open-prs", type=int, default=int(os.getenv("MAX_OPEN_PRS", "100")))
    parser.add_argument("--file-probe-limit", type=int, default=int(os.getenv("FILE_PROBE_LIMIT", "25")))
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
        recent_hours=args.recent_hours,
        max_open_prs=args.max_open_prs,
        file_probe_limit=args.file_probe_limit,
    )
    report, llm_status = build_report(
        prs,
        upstream_repo=args.upstream_repo,
        recent_hours=args.recent_hours,
        generated_at=generated_at,
    )
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report)
    print(f"LLM status: {llm_status}", file=sys.stderr)

    if args.dry_run:
        return 0
    write_client = GitHubClient(write_token)
    issue_title = dated_report_title(args.report_title, generated_at)
    issue_url = publish_report(
        write_client,
        report_repo=args.report_repo,
        title=issue_title,
        body=report,
        post_comment=not args.no_comment,
    )
    print(f"Updated report issue: {issue_url}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
