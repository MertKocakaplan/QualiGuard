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

import pytest

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
        f"{now_ts}|dev@x.com|fix crash in parser",
        f"{now_ts - 3600}|dev@x.com|refactor token handler",
        f"{old_ts}|other@x.com|add new feature",
    ])

    with patch("pipeline.git_metrics.subprocess.run",
               return_value=_fake_run(stdout, returncode=0)):
        summary = git_metrics.get_repo_commit_summary(Path("/fake"))

    assert summary["total_commits"] == 3
    assert summary["bug_fix_commits"] == 1
    assert summary["refactor_commits"] == 1
    assert summary["recent_commits_90d"] == 2  # last 90d window
    # F3.3 — refactor_ratio
    assert summary["refactor_ratio"] == pytest.approx(1 / 3, rel=1e-3)
    # F3.4 — contribution_gini (2 authors: 2 commits, 1 commit)
    assert 0.0 <= summary["contribution_gini"] <= 1.0
    # F3.5 — cadence / entropy fields present
    assert "revert_count" in summary
    assert "inter_commit_time_cv" in summary
    assert "author_entropy" in summary


def test_summary_empty_stdout():
    with patch("pipeline.git_metrics.subprocess.run",
               return_value=_fake_run("", returncode=0)):
        summary = git_metrics.get_repo_commit_summary(Path("/fake"))
    assert summary["total_commits"] == 0
    assert summary["bug_fix_commits"] == 0
    assert summary["refactor_commits"] == 0
    assert summary["recent_commits_90d"] == 0
    assert summary["refactor_ratio"] == 0.0
    assert summary["contribution_gini"] == 0.0
    assert summary["revert_count"] == 0
    assert summary["inter_commit_time_cv"] == 0.0
    assert summary["author_entropy"] == 0.0


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
        "not-a-number|dev@x.com|fix crash",
        "also-bad|dev@x.com|refactor ui",
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


# ── F3.3 — gini_coefficient ──────────────────────────────────────

def test_gini_equal_distribution_is_zero():
    assert git_metrics.gini_coefficient([10, 10, 10, 10]) == pytest.approx(0.0, abs=1e-9)


def test_gini_single_contributor_near_max():
    # Gini max = (n-1)/n; for n=4 contributors where only 1 contributes: 3/4 = 0.75
    assert git_metrics.gini_coefficient([0, 0, 0, 100]) == pytest.approx(0.75, rel=1e-3)


def test_gini_approaches_one_with_large_n():
    # 99 contributors with 0 commits, 1 with 1000 → close to 1
    values = [0] * 99 + [1000]
    g = git_metrics.gini_coefficient(values)
    assert g > 0.98


def test_gini_empty_or_all_zero():
    assert git_metrics.gini_coefficient([]) == 0.0
    assert git_metrics.gini_coefficient([0, 0]) == 0.0


def test_gini_two_contributors():
    # 1 commit vs 3 commits: moderate inequality
    g = git_metrics.gini_coefficient([1, 3])
    assert 0.0 < g < 1.0


# ── F3.4 — inter_commit_time_cv ──────────────────────────────────

def test_cv_uniform_spacing_is_zero():
    # Equal gaps -> std=0 -> cv=0
    ts = [1000, 2000, 3000, 4000]
    assert git_metrics.inter_commit_time_cv(ts) == pytest.approx(0.0, abs=1e-9)


def test_cv_single_timestamp_is_zero():
    assert git_metrics.inter_commit_time_cv([999]) == 0.0


def test_cv_empty_is_zero():
    assert git_metrics.inter_commit_time_cv([]) == 0.0


def test_cv_irregular_spacing_positive():
    # 1, 1, 1, 100 — highly irregular
    ts = [0, 1, 2, 3, 103]
    cv = git_metrics.inter_commit_time_cv(ts)
    assert cv > 0.5


# ── F3.4 — author_entropy ────────────────────────────────────────

def test_author_entropy_single_author_is_zero():
    assert git_metrics.author_entropy({"a@x.com": 10}) == pytest.approx(0.0, abs=1e-9)


def test_author_entropy_equal_distribution_is_max():
    # 2 authors equal -> entropy = 1.0 bit
    e = git_metrics.author_entropy({"a@x.com": 5, "b@x.com": 5})
    assert e == pytest.approx(1.0, rel=1e-3)


def test_author_entropy_empty_is_zero():
    assert git_metrics.author_entropy({}) == 0.0


def test_author_entropy_increases_with_more_equal_authors():
    e2 = git_metrics.author_entropy({"a": 1, "b": 1})
    e4 = git_metrics.author_entropy({"a": 1, "b": 1, "c": 1, "d": 1})
    assert e4 > e2


# ── F3.5 — bug_fix_density ───────────────────────────────────────

def test_bug_fix_density_basic():
    # 10 bugs / 2 kloc / 5 years = 1.0
    assert git_metrics.bug_fix_density(10, 2.0, 5.0) == pytest.approx(1.0)


def test_bug_fix_density_zero_kloc_clamped():
    # kloc=0 clamped to 0.1 — no division by zero
    d = git_metrics.bug_fix_density(5, 0.0, 1.0)
    assert d > 0


def test_bug_fix_density_zero_age_clamped():
    d = git_metrics.bug_fix_density(5, 1.0, 0.0)
    assert d > 0


# ── F3.5 — revert_count in summary ──────────────────────────────

def test_summary_counts_reverts():
    import time
    now_ts = int(time.time())
    stdout = "\n".join([
        f"{now_ts}|dev@x.com|Revert \"add feature X\"",
        f"{now_ts - 100}|dev@x.com|fix actual bug",
        f"{now_ts - 200}|dev@x.com|add feature Y",
    ])
    with patch("pipeline.git_metrics.subprocess.run",
               return_value=_fake_run(stdout, returncode=0)):
        summary = git_metrics.get_repo_commit_summary(Path("/fake"))
    assert summary["revert_count"] == 1
    assert summary["total_commits"] == 3
