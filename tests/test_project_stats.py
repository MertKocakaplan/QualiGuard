"""
test_project_stats.py — pipeline.project_stats compute/write helper'lari.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pipeline import project_stats as ps


def _base_df() -> pd.DataFrame:
    return pd.DataFrame({
        "project_name":         ["u/a", "u/a", "u/b", "u/b", "u/c"],
        "category_primary":     ["Web", "Web", "AI/ML", "AI/ML", "Web"],
        "loc":                  [100,   200,   300,    400,    500],
        "smell_count":          [1,     2,     3,      4,      5],
        "bug_keyword":          [0,     1,     0,      1,      1],
        "bug_szz":              [0,     1,     0,      0,      1],
        "commits_to_first_bug": [5,     8,     -1,     12,     3],
    })


def test_compute_project_stats_global_basics():
    df = _base_df()
    stats = ps.compute_project_stats(df)
    g = stats["global"]
    assert g["n_projects"] == 3
    assert g["n_files"] == 5
    assert g["total_loc"] == 1500
    # Bug rate = pozitif oran
    assert g["bug_rate_keyword"] == 0.6  # 3/5
    assert g["bug_rate_szz"]     == 0.4  # 2/5
    # commits_to_first_bug -1 haric: ortalama (5+8+12+3)/4 = 7.0
    assert g["avg_commits_to_first_bug"] == 7.0


def test_compute_project_stats_by_category_present():
    df = _base_df()
    stats = ps.compute_project_stats(df)
    assert "Web" in stats["by_category"]
    assert "AI/ML" in stats["by_category"]
    web = stats["by_category"]["Web"]
    assert web["n_files"] == 3
    assert web["n_projects"] == 2  # u/a, u/c


def test_compute_project_stats_empty_df_safe():
    stats = ps.compute_project_stats(pd.DataFrame())
    assert stats["global"]["n_projects"] == 0
    assert stats["global"]["n_files"] == 0
    assert stats["by_category"] == {}


def test_compute_project_stats_missing_optional_columns():
    df = pd.DataFrame({
        "project_name": ["u/a", "u/b"],
        "loc":          [10, 20],
    })
    stats = ps.compute_project_stats(df)
    # Eksik bug/smell/ctfb → 0.0 fallback
    assert stats["global"]["bug_rate_keyword"] == 0.0
    assert stats["global"]["bug_rate_szz"] == 0.0
    assert stats["global"]["avg_commits_to_first_bug"] == 0.0


def test_write_project_stats_writes_json(tmp_path: Path):
    df = _base_df()
    out = ps.write_project_stats(df, tmp_path / "stats.json")
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "global" in data and "by_category" in data
    assert data["global"]["n_files"] == 5
