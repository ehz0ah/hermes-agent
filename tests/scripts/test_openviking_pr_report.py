from __future__ import annotations

import importlib.util
import sys
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
) -> report.PullRequest:
    return report.PullRequest(
        number=number,
        title=title,
        body=body,
        html_url=f"https://github.com/NousResearch/hermes-agent/pull/{number}",
        updated_at="2026-05-04T12:00:00Z",
        files=files or [],
    )


def test_keyword_filter_matches_any_keyword_in_title_or_body() -> None:
    title_match = make_pr(1, "fix(openviking): reconnect provider")
    body_match = make_pr(2, "fix: memory provider", "This updates viking:// resource handling.")
    unrelated = make_pr(3, "fix: gateway cleanup", "No memory changes.")

    assert title_match.text_matches
    assert body_match.text_matches
    assert not unrelated.text_matches


def test_path_filter_matches_openviking_plugin_paths() -> None:
    pr = make_pr(
        10,
        "fix: memory provider cleanup",
        files=["plugins/memory/openviking/__init__.py", "gateway/run.py"],
    )

    assert pr.path_matches
    assert pr.is_relevant
    assert "plugins/memory/openviking/__init__.py" in pr.match_reason


def test_path_filter_matches_openviking_test_paths() -> None:
    pr = make_pr(
        11,
        "test: memory provider cleanup",
        files=["tests/plugins/memory/test_openviking_provider.py"],
    )

    assert pr.path_matches
    assert report.openviking_paths(pr.files) == ["tests/plugins/memory/test_openviking_provider.py"]


def test_fetch_search_prs_deduplicates_keyword_results() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.seen_queries: list[str] = []

        def request(self, method, path, *, params=None, data=None, timeout=30):
            assert method == "GET"
            assert path == "/search/issues"
            assert params["page"] == 1
            self.seen_queries.append(params["q"])
            if "openviking" in params["q"]:
                return {
                    "items": [
                        {
                            "number": 100,
                            "title": "fix(openviking): first",
                            "body": "OpenViking body",
                            "html_url": "https://example.test/100",
                            "updated_at": "2026-05-04T12:00:00Z",
                        },
                        {
                            "number": 99,
                            "title": "fix(openviking): duplicate",
                            "body": "",
                            "html_url": "https://example.test/99",
                            "updated_at": "2026-05-04T12:00:00Z",
                        },
                    ]
                }
            if "viking://" in params["q"]:
                return {
                    "items": [
                        {
                            "number": 99,
                            "title": "fix(openviking): duplicate",
                            "body": "",
                            "html_url": "https://example.test/99",
                            "updated_at": "2026-05-04T12:00:00Z",
                        },
                    ]
                }
            return {"items": []}

    fake_client = FakeClient()
    prs = report.fetch_search_prs(
        fake_client,
        "NousResearch/hermes-agent",
        max_results=1000,
    )

    assert [pr.number for pr in prs] == [100, 99]
    assert any("repo:NousResearch/hermes-agent is:pr is:open openviking" == query for query in fake_client.seen_queries)


def test_fetch_search_prs_paginates_keyword_results() -> None:
    class FakeClient:
        def request(self, method, path, *, params=None, data=None, timeout=30):
            assert method == "GET"
            assert path == "/search/issues"
            if "openviking" not in params["q"]:
                return {"items": []}
            if params["page"] == 1:
                return {
                    "items": [
                        {
                            "number": number,
                            "title": f"PR {number}",
                            "body": "",
                            "html_url": f"https://example.test/{number}",
                            "updated_at": "2026-05-04T12:00:00Z",
                        }
                        for number in range(1, 101)
                    ]
                }
            if params["page"] == 2:
                return {
                    "items": [
                        {
                            "number": 101,
                            "title": "next page",
                            "body": "",
                            "html_url": "https://example.test/101",
                            "updated_at": "2026-05-03T10:00:00Z",
                        }
                    ]
                }
            return {"items": []}

    prs = report.fetch_search_prs(
        FakeClient(),
        "NousResearch/hermes-agent",
        max_results=1000,
    )

    assert [pr.number for pr in prs][:2] == [101, 100]
    assert len(prs) == 101


def test_filter_relevant_prs_keeps_github_search_matches() -> None:
    github_search_match = make_pr(10, "fix: cron memory")
    text_match = make_pr(11, "fix(openviking): reconnect")

    assert [pr.number for pr in report.filter_relevant_prs([github_search_match, text_match])] == [11, 10]


def test_report_header_includes_match_count() -> None:
    markdown = report.report_header(2)

    assert "Overview: 2 open OpenViking-related PRs found." in markdown


def test_parse_json_object_accepts_fenced_json() -> None:
    data = report.parse_json_object('```json\n{"groups": []}\n```')

    assert data == {"groups": []}


def test_attach_file_paths_parallel_adds_filenames() -> None:
    class FakeClient:
        def request(self, method, path, *, params=None, data=None, timeout=30):
            number = int(path.rsplit("/", 2)[1])
            return [{"filename": f"plugins/memory/openviking/{number}.py"}]

    prs = [make_pr(1, "fix: one"), make_pr(2, "fix: two")]

    report.attach_file_paths(FakeClient(), "NousResearch/hermes-agent", prs, concurrency=2)

    assert prs[0].files == ["plugins/memory/openviking/1.py"]
    assert prs[1].files == ["plugins/memory/openviking/2.py"]


def test_build_llm_prompt_includes_only_matched_pr_facts() -> None:
    pr = make_pr(
        42,
        "fix(openviking): resource routing",
        "Longer body explaining viking_read behavior.",
        ["plugins/memory/openviking/__init__.py"],
    )

    messages = report.build_llm_prompt([pr], body_chars=200)
    user_content = messages[1]["content"]

    assert "fix(openviking): resource routing" in user_content
    assert "plugins/memory/openviking/__init__.py" in user_content
    assert "viking_read behavior" in user_content
    assert '"groups"' in user_content
    assert "Return exactly one JSON object" in user_content
    assert "group related or overlapping work" in user_content
    assert "cause-and-effect style" in user_content
    assert "Expected PR numbers: #42" in user_content
    assert "Do not include Markdown headings" in user_content


def test_validate_grouped_report_drops_hallucinations_and_adds_missing_prs() -> None:
    first = make_pr(2, "fix(openviking): first")
    missing = make_pr(1, "fix(openviking): missing")
    data = {
        "groups": [
            {
                "title": "Group: Endpoint fixes",
                "prs": [
                    {"number": 2, "summary": "Summary: Detailed summary.\n**Possible Overlaps:** #1."},
                    {"number": 999, "summary": "Hallucinated PR."},
                    {"number": 2, "summary": "Duplicate PR."},
                ],
            }
        ]
    }

    groups, missing_numbers = report.validate_grouped_report(data, [first, missing])

    assert [group.title for group in groups] == ["Endpoint fixes", "Other"]
    assert [item.number for group in groups for item in group.prs] == [2, 1]
    assert missing_numbers == [1]
    assert groups[0].prs[0].summary == "Detailed summary."
    assert "Possible Overlaps" not in groups[0].prs[0].summary


def test_lark_card_uses_collapsible_pr_panels() -> None:
    pr = make_pr(42, "fix(openviking): resource routing")
    groups = [report.ReportGroup("Resource routing", [report.PrSummary(42, "Issue, mechanism, and fix.")])]
    card = report.build_lark_card(groups, [pr], title="OpenViking PR Report")

    assert card["msg_type"] == "interactive"
    assert card["card"]["schema"] == "2.0"
    assert card["card"]["header"]["title"]["content"] == "OpenViking PR Report"
    elements = card["card"]["body"]["elements"]
    assert elements[0]["tag"] == "markdown"
    assert "Overview: 1 open OpenViking-related PR found." in elements[0]["content"]
    assert elements[1] == {"tag": "markdown", "content": "**Group: Resource routing**"}
    assert elements[2]["tag"] == "collapsible_panel"
    assert elements[2]["expanded"] is False
    assert elements[2]["header"]["title"]["content"] == (
        "[#42](https://github.com/NousResearch/hermes-agent/pull/42) fix(openviking): resource routing"
    )
    assert elements[2]["elements"][0]["content"] == "**Summary:** Issue, mechanism, and fix."


def test_lark_card_no_matches_text() -> None:
    card = report.build_lark_card([], [], title="OpenViking PR Report")
    elements = card["card"]["body"]["elements"]

    assert elements[0]["content"] == "Overview: 0 open OpenViking-related PRs found."
    assert elements[1]["content"] == "No open OpenViking-related PRs found."


def test_markdown_report_renders_grouped_pr_list() -> None:
    pr = make_pr(
        42,
        "fix(openviking): resource routing",
        "## Problem\nOpenViking routes the resource incorrectly.\n### Fix\nUse the provider path.",
        files=["plugins/memory/openviking/__init__.py"],
    )
    groups = report.fallback_groups([pr])

    markdown = report.render_markdown_report(groups, [pr], llm_status="not configured")

    assert "**Group: OpenViking-related PRs**" in markdown
    assert "- [#42]" in markdown
    assert "**Summary:**" in markdown
    assert "plugins/memory/openviking" not in markdown
    assert "matched the OpenViking report filter" not in markdown
    assert "Possible Overlaps" not in markdown
    assert "##" not in markdown
    assert "Problem OpenViking routes the resource incorrectly." in markdown


def test_no_matches_markdown_report_text() -> None:
    markdown = report.render_markdown_report([], [], llm_status="skipped")

    assert "Overview: 0 open OpenViking-related PRs found." in markdown
    assert "No open OpenViking-related PRs found." in markdown
