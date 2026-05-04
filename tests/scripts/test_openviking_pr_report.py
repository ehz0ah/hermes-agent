from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "openviking_pr_report.py"
SPEC = importlib.util.spec_from_file_location("openviking_pr_report", MODULE_PATH)
assert SPEC and SPEC.loader
report = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = report
SPEC.loader.exec_module(report)


def make_pr(
    number: int,
    title: str,
    body: str = "",
    files: list[str] | None = None,
    state: str = "open",
    merged_at: str | None = None,
) -> report.PullRequest:
    pr = report.PullRequest(
        number=number,
        title=title,
        body=body,
        state=state,
        html_url=f"https://github.com/NousResearch/hermes-agent/pull/{number}",
        author="contributor",
        created_at="2026-04-01T00:00:00Z",
        updated_at="2026-04-20T00:00:00Z",
        merged_at=merged_at,
        files=files or [],
    )
    pr.linked_issues = report.extract_issue_refs(body)
    return pr


def test_openviking_path_marks_pr_as_related() -> None:
    pr = make_pr(
        101,
        "fix: memory provider cleanup",
        files=["plugins/memory/openviking/__init__.py"],
    )

    assert report.pr_has_openviking_signal(pr)


def test_duplicate_clusters_group_local_upload_prs() -> None:
    prs = [
        make_pr(
            10360,
            "fix(memory): support OpenViking local resource uploads",
            "Fixes #10350. Uploads local files before add_resource.",
            ["plugins/memory/openviking/__init__.py", "tests/plugins/memory/test_openviking_provider.py"],
        ),
        make_pr(
            19569,
            "fix(memory): harden OpenViking local resource uploads",
            "Builds on #10360 and follows local upload handling for files/directories.",
            ["plugins/memory/openviking/__init__.py", "tests/plugins/memory/test_openviking_provider.py"],
        ),
        make_pr(
            11710,
            "fix(openviking): improve explicit fallback recall",
            "Ensure fallback memories show up in prefetch and search.",
            ["plugins/memory/openviking/__init__.py"],
        ),
    ]

    clusters = report.build_duplicate_clusters(prs)

    local_upload_clusters = [
        cluster for cluster in clusters if {10360, 19569} <= {pr.number for pr in cluster.prs}
    ]
    assert local_upload_clusters
    assert local_upload_clusters[0].topic == "Local resource upload"
    assert {pr.number for pr in local_upload_clusters[0].prs} == {10360, 19569}
    assert any("shared topic" in reason for reason in local_upload_clusters[0].reasons)


def test_file_overlap_alone_does_not_create_duplicate_cluster() -> None:
    prs = [
        make_pr(
            1,
            "fix(openviking): improve fallback recall",
            "Ensure fallback memories show up in prefetch and search.",
            ["plugins/memory/openviking/__init__.py"],
        ),
        make_pr(
            2,
            "fix(openviking): reconnect after health check failure",
            "Lazily reconnect OpenViking after startup health checks fail.",
            ["plugins/memory/openviking/__init__.py"],
        ),
    ]

    assert report.build_duplicate_clusters(prs) == []


def test_non_openviking_reference_comment_is_not_relevance_signal() -> None:
    pr = make_pr(3, "fix: unrelated gateway command", "Fixes #1234", ["gateway/run.py"])
    pr.comments = ["Related to #10360"]

    assert not report.pr_has_openviking_signal(pr)


def test_render_report_includes_duplicate_and_recent_sections() -> None:
    open_pr = make_pr(
        10360,
        "fix(memory): support OpenViking local resource uploads",
        "Upload local files.",
    )
    merged_pr = make_pr(
        19569,
        "fix(memory): harden OpenViking local resource uploads",
        "Builds on #10360 and preserves local upload behavior.",
        state="closed",
        merged_at="2026-04-21T00:00:00Z",
    )
    prs = [open_pr, merged_pr]
    clusters = report.build_duplicate_clusters(prs)

    markdown = report.render_deterministic_report(
        prs,
        clusters,
        upstream_repo="NousResearch/hermes-agent",
        recent_hours=24,
        generated_at=datetime(2026, 5, 4, tzinfo=UTC),
        llm_status="not configured",
    )

    assert "## Likely Duplicate Groups" in markdown
    assert "Local resource upload" in markdown
    assert "Scope: open OpenViking-related PRs updated in the last 24 hours." in markdown
    assert "[#19569]" in markdown


def test_llm_assessments_accept_real_related_prs_only() -> None:
    prs = [
        make_pr(10, "fix(openviking): update resource reads"),
        make_pr(11, "fix: unrelated gateway cleanup"),
    ]
    data = {
        "pull_requests": [
            {
                "number": 10,
                "is_openviking_related": True,
                "confidence": "high",
                "topic": "Read routing",
                "summary": "Updates OpenViking resource read behavior.",
                "why": "Title explicitly names OpenViking.",
                "duplicate_group": "",
            },
            {
                "number": 11,
                "is_openviking_related": False,
                "confidence": "low",
                "summary": "Rejected.",
                "why": "No OpenViking signal.",
            },
            {
                "number": 999,
                "is_openviking_related": True,
                "confidence": "high",
                "summary": "Invented.",
                "why": "Should be ignored.",
            },
        ]
    }

    assessments = report.parse_llm_assessments(data, prs)

    assert [assessment.pr.number for assessment in assessments] == [10]
    assert assessments[0].summary == "Updates OpenViking resource read behavior."


def test_extract_json_object_accepts_fenced_json() -> None:
    parsed = report.extract_json_object(
        """```json
{"pull_requests": []}
```"""
    )

    assert parsed == {"pull_requests": []}


def test_render_llm_report_includes_table_summaries_and_duplicate_groups() -> None:
    first = make_pr(10, "fix(openviking): local upload")
    second = make_pr(11, "feat(openviking): upload local resources")
    assessments = [
        report.LlmPrAssessment(
            pr=first,
            is_openviking_related=True,
            confidence="high",
            summary="Hardens local uploads before resource creation.",
            why="Touches OpenViking local resource upload behavior.",
            duplicate_group="Local resource upload",
            topic="Local resource upload",
        ),
        report.LlmPrAssessment(
            pr=second,
            is_openviking_related=True,
            confidence="medium",
            summary="Adds another local resource upload path.",
            why="Changes the same OpenViking upload flow.",
            duplicate_group="Local resource upload",
            topic="Local resource upload",
        ),
    ]

    markdown = report.render_llm_report(
        assessments,
        reviewed_count=20,
        upstream_repo="NousResearch/hermes-agent",
        recent_hours=24,
        generated_at=datetime(2026, 5, 4, tzinfo=UTC),
        llm_status="classified with `model`",
    )

    assert "| PR | Confidence | Topic / duplicate group | Summary | Why included |" in markdown
    assert "Hardens local uploads before resource creation." in markdown
    assert "Reviewed: 20 open PRs" in markdown
    assert "### Local resource upload" in markdown


def test_chat_completions_url_accepts_base_or_full_endpoint() -> None:
    assert report.chat_completions_url("https://api.example.com/v1") == "https://api.example.com/v1/chat/completions"
    assert (
        report.chat_completions_url("https://api.example.com/v1/chat/completions")
        == "https://api.example.com/v1/chat/completions"
    )


def test_dated_report_title_uses_singapore_date() -> None:
    generated_at = datetime(2026, 5, 3, 18, 30, tzinfo=UTC)

    assert report.dated_report_title("OpenViking PR Report", generated_at) == "OpenViking PR Report - 2026-05-04"
    assert report.dated_report_title("OpenViking %Y-%m-%d", generated_at) == "OpenViking 2026-05-04"
    assert report.dated_report_title("OpenViking PR Report - 2026-05-04", generated_at) == "OpenViking PR Report - 2026-05-04"
