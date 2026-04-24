"""
test_health.py — app.health helper'lari.

compute_project_health ve compute_smell_summary icin birim testler.
Hicbir Flask veya pipeline modulune bagli degil — saf fonksiyon testi.
"""
from __future__ import annotations

from app.health import compute_project_health, compute_smell_summary


# ── compute_project_health ───────────────────────────────────────

def _rows():
    return [
        {"file_path": "a.py", "loc": 100, "bug_count": 1, "commit_count": 10},
        {"file_path": "b.py", "loc": 200, "bug_count": 0, "commit_count": 5},
        {"file_path": "c.py", "loc": 300, "bug_count": 2, "commit_count": 15},
    ]


def test_project_health_defect_density():
    cs = {"total_commits": 100, "bug_fix_commits": 15,
          "refactor_commits": 5, "recent_commits_90d": 20}
    h = compute_project_health(cs, _rows())
    # 3 bugs / (600 loc / 1000) = 3 / 0.6 = 5.0
    assert h["defect_density_per_kloc"] == 5.0
    assert h["refactor_ratio"] == 0.05
    assert h["bug_fix_ratio"] == 0.15
    assert h["recent_commits_90d"] == 20
    assert h["total_commits"] == 100
    assert h["bug_fix_commits"] == 15
    assert h["refactor_commits"] == 5
    assert h["avg_commits_per_file"] == 10.0  # (10+5+15)/3
    assert h["total_loc"] == 600


def test_project_health_zero_commits_safe():
    cs = {"total_commits": 0, "bug_fix_commits": 0,
          "refactor_commits": 0, "recent_commits_90d": 0}
    h = compute_project_health(cs, _rows())
    assert h["refactor_ratio"] == 0.0
    assert h["bug_fix_ratio"] == 0.0
    assert h["total_commits"] == 0


def test_project_health_empty_rows_zero_loc():
    cs = {"total_commits": 10, "bug_fix_commits": 2,
          "refactor_commits": 1, "recent_commits_90d": 3}
    h = compute_project_health(cs, [])
    assert h["defect_density_per_kloc"] == 0.0
    assert h["total_loc"] == 0
    assert h["avg_commits_per_file"] == 0.0


def test_project_health_missing_keys_fallback():
    # commit_summary bos — tum degerler 0 olmali
    h = compute_project_health({}, _rows())
    assert h["total_commits"] == 0
    assert h["bug_fix_commits"] == 0
    assert h["refactor_commits"] == 0
    assert h["recent_commits_90d"] == 0


# ── compute_smell_summary ────────────────────────────────────────

def _file_results():
    return [
        {"file_path": "a.py", "bug_pred": 1, "smell_pred": 1},
        {"file_path": "b.py", "bug_pred": 0, "smell_pred": 1},
        {"file_path": "c.py", "bug_pred": 1, "smell_pred": 0},
        {"file_path": "d.py", "bug_pred": 0, "smell_pred": 0},
    ]


def test_smell_summary_no_prospector_only_ml():
    fr = _file_results()
    rows = [{"file_path": f["file_path"], "loc": 100} for f in fr]
    s = compute_smell_summary(fr, rows, prospector_results=None)
    assert s["prospector_enabled"] is False
    assert s["total_smells"] == 0
    assert s["smell_density_per_kloc"] == 0.0
    assert s["ml_smell_risk_count"] == 2   # a, b
    assert s["refactor_priority_count"] == 1  # a
    assert s["top_smelly_files"] == []


def test_smell_summary_with_prospector():
    fr = _file_results()
    rows = [{"file_path": f["file_path"], "loc": 100} for f in fr]
    pres = {
        "a.py": {"smell_count": 5,  "categories": {"pylint": 5}},
        "b.py": {"smell_count": 10, "categories": {"pylint": 10}},
        "c.py": {"smell_count": 0,  "categories": {}},
        "d.py": {"smell_count": None},
    }
    s = compute_smell_summary(fr, rows, prospector_results=pres)
    assert s["prospector_enabled"] is True
    assert s["total_smells"] == 15
    # 15 / (400/1000) = 37.5
    assert s["smell_density_per_kloc"] == 37.5
    assert s["refactor_priority_count"] == 1
    assert len(s["top_smelly_files"]) == 2  # zero-count is omitted
    assert s["top_smelly_files"][0]["file_path"] == "b.py"
    assert s["top_smelly_files"][0]["smell_count"] == 10


def test_smell_summary_all_none_prospector():
    fr = _file_results()
    rows = [{"file_path": f["file_path"], "loc": 100} for f in fr]
    pres = {f: {"smell_count": None} for f in ("a.py", "b.py")}
    s = compute_smell_summary(fr, rows, prospector_results=pres)
    assert s["prospector_enabled"] is True  # dict non-empty
    assert s["total_smells"] == 0
    assert s["top_smelly_files"] == []


def test_smell_summary_malformed_values_ignored():
    fr = _file_results()
    rows = [{"file_path": "a.py", "loc": 100}]
    pres = {"a.py": {"smell_count": "not-a-number"}}
    s = compute_smell_summary(fr, rows, prospector_results=pres)
    assert s["total_smells"] == 0


def test_smell_summary_zero_loc_density_safe():
    fr = [{"file_path": "a.py", "bug_pred": 0, "smell_pred": 1}]
    rows = [{"file_path": "a.py", "loc": 0}]
    pres = {"a.py": {"smell_count": 5}}
    s = compute_smell_summary(fr, rows, prospector_results=pres)
    assert s["total_smells"] == 5
    assert s["smell_density_per_kloc"] == 0.0
