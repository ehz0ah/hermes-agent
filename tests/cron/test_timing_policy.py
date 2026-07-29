"""Tests for exact and low-contention background cron scheduling."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from cron.jobs import create_job, get_due_jobs, get_job, load_jobs, save_jobs, update_job
from cron.timing_policy import (
    BACKGROUND,
    EXACT,
    background_defer_until,
    normalize_timing_policy,
    resolve_background_window,
)


SGT = ZoneInfo("Asia/Singapore")


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


def _config(**overrides):
    window = {
        "timezone": "Asia/Singapore",
        "start": "20:00",
        "end": "08:00",
        "active_grace_seconds": 300,
        "max_deferral_seconds": 43200,
        "jitter_seconds": 0,
    }
    window.update(overrides)
    return {"cron": {"background_window": window}}


def test_normalize_timing_policy_defaults_to_exact():
    assert normalize_timing_policy(None) == EXACT
    assert normalize_timing_policy("") == EXACT
    assert normalize_timing_policy(" Background ") == BACKGROUND
    with pytest.raises(ValueError, match="timing_policy"):
        normalize_timing_policy("whenever")


def test_background_window_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="HH:MM"):
        resolve_background_window(_config(start="8pm"))
    with pytest.raises(ValueError, match="timezone"):
        resolve_background_window(_config(timezone="Mars/Olympus"))


def test_background_job_outside_window_defers_to_evening():
    now = datetime(2026, 7, 29, 10, 0, tzinfo=SGT)
    deferred = background_defer_until(
        {
            "id": "daily-digest",
            "timing_policy": BACKGROUND,
            "background_deferred_since": now.isoformat(),
        },
        now,
        config=_config(),
        activity_age_seconds=None,
    )
    assert deferred == datetime(2026, 7, 29, 20, 0, tzinfo=SGT)


def test_background_job_inside_window_defers_for_recent_activity():
    now = datetime(2026, 7, 29, 21, 0, tzinfo=SGT)
    deferred = background_defer_until(
        {
            "id": "nightly-report",
            "timing_policy": BACKGROUND,
            "background_deferred_since": now.isoformat(),
        },
        now,
        config=_config(),
        activity_age_seconds=120,
    )
    assert deferred == now + timedelta(seconds=180)


def test_background_job_runs_after_bounded_deferral():
    now = datetime(2026, 7, 29, 10, 0, tzinfo=SGT)
    deferred = background_defer_until(
        {
            "id": "bounded-report",
            "timing_policy": BACKGROUND,
            "background_deferred_since": (
                now - timedelta(hours=12)
            ).isoformat(),
        },
        now,
        config=_config(max_deferral_seconds=43200),
        activity_age_seconds=0,
    )
    assert deferred is None


def test_create_update_and_legacy_normalization(tmp_cron_dir):
    job = create_job(
        prompt="Summarize activity",
        schedule="every 1h",
        timing_policy=BACKGROUND,
    )
    assert job["timing_policy"] == BACKGROUND
    assert get_job(job["id"])["timing_policy"] == BACKGROUND

    jobs = load_jobs()
    jobs[0]["background_deferred_since"] = datetime.now(SGT).isoformat()
    save_jobs(jobs)
    updated = update_job(job["id"], {"timing_policy": EXACT})
    assert updated["timing_policy"] == EXACT
    assert updated.get("background_deferred_since") is None

    jobs = load_jobs()
    jobs[0].pop("timing_policy", None)
    save_jobs(jobs)
    assert get_job(job["id"])["timing_policy"] == EXACT


def test_due_scan_persists_background_deferral(tmp_cron_dir, monkeypatch):
    now = datetime(2026, 7, 29, 10, 0, tzinfo=SGT)
    job = create_job(
        prompt="Background summary",
        schedule="every 1h",
        timing_policy=BACKGROUND,
    )
    jobs = load_jobs()
    jobs[0]["next_run_at"] = (now - timedelta(minutes=1)).isoformat()
    save_jobs(jobs)

    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _config())
    monkeypatch.setattr(
        "cron.timing_policy.seconds_since_interactive_activity",
        lambda: None,
    )

    assert get_due_jobs() == []
    deferred = get_job(job["id"])
    assert datetime.fromisoformat(deferred["next_run_at"]) == datetime(
        2026, 7, 29, 20, 0, tzinfo=SGT
    )
    assert deferred["background_deferred_since"] == (
        now - timedelta(minutes=1)
    ).isoformat()


def test_due_scan_runs_background_job_after_deadline(tmp_cron_dir, monkeypatch):
    now = datetime(2026, 7, 29, 10, 0, tzinfo=SGT)
    job = create_job(
        prompt="Eventually run",
        schedule="every 1h",
        timing_policy=BACKGROUND,
    )
    jobs = load_jobs()
    jobs[0]["next_run_at"] = (now - timedelta(minutes=1)).isoformat()
    jobs[0]["background_deferred_since"] = (
        now - timedelta(hours=12, minutes=1)
    ).isoformat()
    save_jobs(jobs)

    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _config())

    due = get_due_jobs()
    assert [item["id"] for item in due] == [job["id"]]
