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
