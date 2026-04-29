"""
test_git_metrics_summary.py — get_repo_commit_summary + is_refactor_message.

F7: repo-level commit ozeti Flask UI Project Health kartlari icin gerekli.
git log cagrisini subprocess.run'i mock'layarak dogrulariz; gercek git
kurulumuna ihtiyac duymaz.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pipeline import git_metrics


# ── is_refactor_message ───────────────────────────────────────────

def test_is_refactor_message_positive():
    assert git_metrics.is_refactor_message("refactor user model")
    assert git_metrics.is_refactor_message("Refactored auth flow")
    assert git_metrics.is_refactor_message("cleanup old tests")
    assert git_metrics.is_refactor_message("rename helper to util")
    assert git_metrics.is_refactor_message("simplify query logic")
    assert git_metrics.is_refactor_message("extract helper function")


def test_is_refactor_message_negative():
    assert not git_metrics.is_refactor_message("fix bug in auth")
    assert not git_metrics.is_refactor_message("add new feature")
    assert not git_metrics.is_refactor_message("")
    assert not git_metrics.is_refactor_message(None)


# ── get_repo_commit_summary (subprocess mocked) ───────────────────

def _fake_run(stdout: str, returncode: int = 0):
    """subprocess.run donus objesini tekler."""
    return SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)


def test_summary_counts_bugs_refactors_and_recent():
    """3 commit: 1 bug-fix (recent), 1 refactor (recent), 1 nötr (eski)."""
    import time
    now_ts = int(time.time())
    old_ts = now_ts - 365 * 24 * 3600  # 1 yil once

    stdout = "\n".join([
        f"{now_ts}|fix crash in parser",
        f"{now_ts - 3600}|refactor token handler",
        f"{old_ts}|add new feature",
    ])

    with patch("pipeline.git_metrics.subprocess.run",
               return_value=_fake_run(stdout, returncode=0)):
        summary = git_metrics.get_repo_commit_summary(Path("/fake"))

    assert summary["total_commits"] == 3
    assert summary["bug_fix_commits"] == 1
    assert summary["refactor_commits"] == 1
    assert summary["recent_commits_90d"] == 2  # last 90d window


def test_summary_empty_stdout():
    with patch("pipeline.git_metrics.subprocess.run",
               return_value=_fake_run("", returncode=0)):
        summary = git_metrics.get_repo_commit_summary(Path("/fake"))
    assert summary == {
        "total_commits":      0,
        "bug_fix_commits":    0,
        "refactor_commits":   0,
        "recent_commits_90d": 0,
    }


def test_summary_nonzero_returncode():
    with patch("pipeline.git_metrics.subprocess.run",
               return_value=_fake_run("irrelevant", returncode=128)):
        summary = git_metrics.get_repo_commit_summary(Path("/fake"))
    assert summary["total_commits"] == 0


def test_summary_timeout_fails_soft():
    def _boom(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=1)

    with patch("pipeline.git_metrics.subprocess.run", side_effect=_boom):
        summary = git_metrics.get_repo_commit_summary(Path("/fake"))
    assert summary["total_commits"] == 0
    assert summary["bug_fix_commits"] == 0


def test_summary_malformed_timestamp_ignored():
    """Bozuk ts satirlari sadece recent_commits_90d sayacini etkilemez."""
    stdout = "\n".join([
        "not-a-number|fix crash",
        "also-bad|refactor ui",
    ])
    with patch("pipeline.git_metrics.subprocess.run",
               return_value=_fake_run(stdout, returncode=0)):
        summary = git_metrics.get_repo_commit_summary(Path("/fake"))
    assert summary["total_commits"] == 2
    assert summary["bug_fix_commits"] == 1
    assert summary["refactor_commits"] == 1
    assert summary["recent_commits_90d"] == 0


# ── F3.2 — classify_bug_message ──────────────────────────────────

def test_classify_bug_message_fix_group():
    r = git_metrics.classify_bug_message("fix: null pointer in auth")
    assert r["bug_kw_fix"] == 1
    assert r["bug_kw_bug"] == 0


def test_classify_bug_message_bug_group():
    r = git_metrics.classify_bug_message("Found a bug in the parser")
    assert r["bug_kw_bug"] == 1


def test_classify_bug_message_error_group():
    r = git_metrics.classify_bug_message("resolve error in encoding")
    assert r["bug_kw_error"] == 1


def test_classify_bug_message_no_match():
    r = git_metrics.classify_bug_message("add new feature for export")
    assert all(v == 0 for v in r.values())


def test_classify_bug_message_multiple_groups():
    r = git_metrics.classify_bug_message("fix bug and error in loop")
    assert r["bug_kw_fix"] == 1
    assert r["bug_kw_bug"] == 1
    assert r["bug_kw_error"] == 1


def test_bug_kw_counts_in_bulk_git_stats():
    """get_bulk_git_stats: fix commit -> bug_kw_fix_count > 0."""
    import time
    now_ts = int(time.time())
    stdout = (
        f"COMMIT|abc|dev@x.com|{now_ts}|fix: memory leak\n"
        f"3\t0\tmodule.py\n"
    )
    with patch("pipeline.git_metrics.subprocess.run",
               return_value=_fake_run(stdout, returncode=0)):
        stats = git_metrics.get_bulk_git_stats(Path("/fake"), ["module.py"])
    assert "module.py" in stats
    assert stats["module.py"]["bug_kw_fix_count"] == 1
    assert stats["module.py"]["bug_kw_bug_count"] == 0


def test_bug_kw_all_keys_present_in_bulk():
    """Tum 6 keyword key her zaman mevcut olmali."""
    import time
    now_ts = int(time.time())
    stdout = f"COMMIT|abc|dev@x.com|{now_ts}|chore: update deps\n3\t0\tf.py\n"
    with patch("pipeline.git_metrics.subprocess.run",
               return_value=_fake_run(stdout, returncode=0)):
        stats = git_metrics.get_bulk_git_stats(Path("/fake"), ["f.py"])
    row = stats.get("f.py", {})
    for g in ("fix", "bug", "error", "defect", "issue", "anomaly"):
        assert f"bug_kw_{g}_count" in row
