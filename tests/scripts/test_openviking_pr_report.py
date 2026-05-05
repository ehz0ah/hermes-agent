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
    assert [pr.number for pr in report.filter_relevant_prs([title_match, body_match, unrelated])] == [2, 1]


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


def test_fetch_recent_open_prs_paginates_until_cutoff(monkeypatch) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.pages = {
                1: [
                    {
                        "number": number,
                        "title": f"PR {number}",
                        "body": "",
                        "html_url": f"https://example.test/{number}",
                        "updated_at": "2026-05-04T12:00:00Z",
                        "user": {"login": "octo"},
                        "head": {"ref": "branch"},
                    }
                    for number in range(1, 101)
                ],
                2: [
                    {
                        "number": 101,
                        "title": "inside cutoff",
                        "body": "",
                        "html_url": "https://example.test/101",
                        "updated_at": "2026-05-03T13:00:00Z",
                        "user": {"login": "octo"},
                        "head": {"ref": "branch"},
                    },
                    {
                        "number": 102,
                        "title": "outside cutoff",
                        "body": "",
                        "html_url": "https://example.test/102",
                        "updated_at": "2026-05-03T10:00:00Z",
                        "user": {"login": "octo"},
                        "head": {"ref": "branch"},
                    },
                ],
            }

        def request(self, method, path, *, params=None, data=None, timeout=30):
            assert method == "GET"
            assert path == "/repos/NousResearch/hermes-agent/pulls"
            return self.pages.get(params["page"], [])

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 5, 4, 12, 0, tzinfo=tz)

    monkeypatch.setattr(report, "datetime", FixedDatetime)

    prs = report.fetch_recent_open_prs(
        FakeClient(),
        "NousResearch/hermes-agent",
        recent_hours=24,
        max_open_prs=1000,
    )

    assert [pr.number for pr in prs][-2:] == [100, 101]


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

    messages = report.build_llm_prompt([pr], recent_hours=24, body_chars=200)
    user_content = messages[1]["content"]

    assert "fix(openviking): resource routing" in user_content
    assert "plugins/memory/openviking/__init__.py" in user_content
    assert "viking_read behavior" in user_content


def test_lark_card_envelope_uses_interactive_markdown_card() -> None:
    card = report.build_lark_card("# Report\n\nBody", title="OpenViking PR Report", markdown_limit=1000)

    assert card["msg_type"] == "interactive"
    assert card["card"]["header"]["title"]["content"] == "OpenViking PR Report"
    assert card["card"]["elements"][0]["tag"] == "markdown"
    assert "# Report" in card["card"]["elements"][0]["content"]


def test_no_matches_fallback_report_text() -> None:
    markdown = report.render_fallback_report([], recent_hours=24, llm_status="skipped")

    assert "No relevant PRs in the last 24 hours." in markdown
