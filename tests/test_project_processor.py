"""
test_project_processor.py — process_project'in baslica karar yollari.

Gercek git/prospector/szz cagrilari mock'lanir. Hedef: istege bagli
adimlar (skip_szz, skip_prospector), bos sonuclar, yazilan parquet'in
sema uyumu.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from pipeline import project_processor


# ── Yardimci mock factory'leri ────────────────────────────────────

_PROJ = {
    "full_name":         "user/demo",
    "clone_url":         "https://github.com/user/demo.git",
    "stars":             42,
    "contributor_count": 3,
    "project_age_days":  200,
    "default_branch":    "main",
}


def _static_ok(sloc: int = 20) -> dict:
    """Gercek `calculate_metrics` ciktisina yakin, tam sayili sozluk."""
    return {
        "loc": 30, "lloc": 20, "sloc": sloc, "comments": 3,
        "multi": 2, "blank": 5, "single_comments": 3,
        "cc_mean": 1.5, "cc_max": 3, "cc_total": 6, "num_functions": 4,
        "h_vocabulary": 20, "h_length": 40, "h_volume": 150.0,
        "h_difficulty": 2.5, "h_effort": 400.0, "h_bugs": 0.05,
        "h_time": 25.0, "h_calculated_length": 35.0,
        "maintainability_index": 65.0,
        "comment_ratio": 0.1, "doc_ratio": 0.05,
        "complexity_density": 0.2, "comment_per_function": 0.8,
        "avg_function_length": 5.0, "effort_per_line": 13.3,
    }


def _bulk_for(files):
    return {
        f: {
            "commit_count": 5, "bug_count": 1, "n_authors": 2,
            "file_age_days": 100.0, "churn_total": 80,
            "avg_churn_per_commit": 16.0, "max_single_churn": 40,
            "recent_commits_90d": 3,
        }
        for f in files
    }


@pytest.fixture
def tmp_dirs(tmp_path, monkeypatch):
    repos = tmp_path / "repos"
    projects = tmp_path / "projects"
    repos.mkdir()
    projects.mkdir()
    return repos, projects


# ── Testler ───────────────────────────────────────────────────────

def test_process_project_happy_path_writes_parquet(tmp_dirs):
    repos, projects = tmp_dirs
    repo = repos / "demo"
    (repo / ".git").mkdir(parents=True)  # mevcut klon

    files = ["app/module.py", "app/service.py"]

    with patch("pipeline.project_processor.cloning.clone_repo",
               return_value=(repo, "basarili")), \
         patch("pipeline.project_processor.git_metrics.get_head_python_files",
               return_value=files), \
         patch("pipeline.project_processor.git_metrics.get_bulk_git_stats",
               return_value=_bulk_for(files)), \
         patch("pipeline.project_processor.static_metrics.calculate_metrics",
               side_effect=lambda *a, **kw: _static_ok()), \
         patch("pipeline.project_processor.static_metrics.calculate_derived",
               side_effect=lambda d: d), \
         patch("pipeline.project_processor._bug_fix_hashes", return_value=["h1"]), \
         patch("pipeline.project_processor.szz.compute_szz_labels",
               return_value={"app/module.py": 1, "app/service.py": 0}), \
         patch("pipeline.project_processor.prospector_runner.run_prospector_batch",
               return_value={
                   repo / "app/module.py":  {"smell_count": 7,  "categories": {"pylint": 7}, "messages": []},
                   repo / "app/service.py": {"smell_count": 2,  "categories": {"pylint": 2}, "messages": []},
               }), \
         patch("pipeline.project_processor.Path.read_text",
               return_value="def f():\n  return 1\n"):
        result = project_processor.process_project(
            _PROJ, repos_dir=repos, projects_dir=projects, workers=2,
        )

    assert result["status"] == "ok"
    assert result["files"] == 2
    assert result["bugs_keyword"] == 2  # her iki dosya bug_count=1
    assert result["bugs_szz"]     == 1
    assert result["smells_total"] == 9
    parquet = Path(result["parquet"])
    assert parquet.exists()

    df = pd.read_parquet(parquet)
    assert len(df) == 2
    # Sema alanlari
    expected_cols = {
        "file_path", "project_name", "stars", "contributor_count",
        "project_age_days", "commit_count", "bug_count", "bug_keyword",
        "bug_szz", "commits_to_first_bug", "n_authors", "file_age_days",
        "churn_total", "avg_churn_per_commit", "max_single_churn",
        "recent_commits_90d", "loc", "lloc", "sloc",
        "cc_mean", "cc_max", "cc_total", "num_functions",
        "h_vocabulary", "h_volume", "maintainability_index",
        "comment_ratio", "doc_ratio", "complexity_density",
        "smell_count", "smell_categories",
    }
    assert expected_cols.issubset(df.columns), (
        f"eksik sutunlar: {expected_cols - set(df.columns)}"
    )
    # Tip uyumu
    assert str(df["bug_szz"].dtype).startswith("Int")  # nullable int
    assert str(df["smell_count"].dtype).startswith("Int")
    assert df["project_name"].iloc[0] == "user/demo"


def test_process_project_skip_szz_and_prospector(tmp_dirs):
    repos, projects = tmp_dirs
    repo = repos / "demo"
    (repo / ".git").mkdir(parents=True)

    files = ["a.py"]

    with patch("pipeline.project_processor.cloning.clone_repo",
               return_value=(repo, "basarili")), \
         patch("pipeline.project_processor.git_metrics.get_head_python_files",
               return_value=files), \
         patch("pipeline.project_processor.git_metrics.get_bulk_git_stats",
               return_value=_bulk_for(files)), \
         patch("pipeline.project_processor.static_metrics.calculate_metrics",
               side_effect=lambda *a, **kw: _static_ok()), \
         patch("pipeline.project_processor.static_metrics.calculate_derived",
               side_effect=lambda d: d), \
         patch("pipeline.project_processor.Path.read_text", return_value="x=1\n"), \
         patch("pipeline.project_processor.szz.compute_szz_labels") as m_szz, \
         patch("pipeline.project_processor.prospector_runner.run_prospector_batch") as m_pros:
        result = project_processor.process_project(
            _PROJ, skip_szz=True, skip_prospector=True,
            repos_dir=repos, projects_dir=projects,
        )

    assert result["status"] == "ok"
    # Hicbirisi cagrilmamali
    m_szz.assert_not_called()
    m_pros.assert_not_called()
    # bug_szz ve smell_count null olmali
    df = pd.read_parquet(result["parquet"])
    assert df["bug_szz"].isna().all()
    assert df["smell_count"].isna().all()


def test_process_project_clone_failure_returns_failed(tmp_dirs):
    repos, projects = tmp_dirs
    with patch("pipeline.project_processor.cloning.clone_repo",
               return_value=(None, "timeout 600s")):
        result = project_processor.process_project(
            _PROJ, repos_dir=repos, projects_dir=projects,
        )
    assert result["status"] == "failed"
    assert "clone" in result["error"]
    assert not any(projects.iterdir())  # parquet yazilmamis


def test_process_project_empty_when_no_head_files(tmp_dirs):
    repos, projects = tmp_dirs
    repo = repos / "demo"
    (repo / ".git").mkdir(parents=True)

    with patch("pipeline.project_processor.cloning.clone_repo",
               return_value=(repo, "basarili")), \
         patch("pipeline.project_processor.git_metrics.get_head_python_files",
               return_value=[]):
        result = project_processor.process_project(
            _PROJ, repos_dir=repos, projects_dir=projects,
        )
    assert result["status"] == "empty"
    assert result["error"] == "no_python_files"


def test_process_project_parquet_schema_matches_plan_14_1(tmp_dirs):
    """
    PLAN §14.1 per-project parquet sutun/tip sozlesmesi.

    Tum zorunlu sutunlar mevcut olmali ve dtype beklentilere uymali.
    Eksik bir sutun ya da yanlis dtype veri setinin merge + label
    asamasini kiracagi icin burada kesin dogrulanir.
    """
    repos, projects = tmp_dirs
    repo = repos / "demo"
    (repo / ".git").mkdir(parents=True)

    files = ["pkg/m.py"]

    with patch("pipeline.project_processor.cloning.clone_repo",
               return_value=(repo, "basarili")), \
         patch("pipeline.project_processor.git_metrics.get_head_python_files",
               return_value=files), \
         patch("pipeline.project_processor.git_metrics.get_bulk_git_stats",
               return_value=_bulk_for(files)), \
         patch("pipeline.project_processor.static_metrics.calculate_metrics",
               side_effect=lambda *a, **kw: _static_ok()), \
         patch("pipeline.project_processor.static_metrics.calculate_derived",
               side_effect=lambda d: d), \
         patch("pipeline.project_processor.Path.read_text", return_value="x=1\n"), \
         patch("pipeline.project_processor._bug_fix_hashes", return_value=["h1"]), \
         patch("pipeline.project_processor.szz.compute_szz_labels",
               return_value={"pkg/m.py": 1}), \
         patch("pipeline.project_processor.prospector_runner.run_prospector_batch",
               return_value={
                   repo / "pkg/m.py": {"smell_count": 4, "categories": {"pylint": 4}, "messages": []},
               }):
        result = project_processor.process_project(
            _PROJ, repos_dir=repos, projects_dir=projects,
        )

    assert result["status"] == "ok"
    df = pd.read_parquet(result["parquet"])

    # PLAN §14.1: zorunlu sutunlar
    required = {
        # Proje seviyesi
        "file_path", "project_name", "stars", "contributor_count",
        "project_age_days",
        # Git metrikleri
        "commit_count", "bug_count", "bug_keyword", "bug_szz",
        "commits_to_first_bug", "n_authors", "file_age_days",
        "churn_total", "avg_churn_per_commit", "max_single_churn",
        "recent_commits_90d",
        # Radon raw
        "loc", "lloc", "sloc", "comments", "multi", "blank", "single_comments",
        "cc_mean", "cc_max", "cc_total", "num_functions",
        # Halstead 8
        "h_vocabulary", "h_length", "h_volume", "h_difficulty",
        "h_effort", "h_bugs", "h_time", "h_calculated_length",
        # Maintainability
        "maintainability_index", "comment_ratio", "doc_ratio",
        # Derived 4
        "complexity_density", "comment_per_function",
        "avg_function_length", "effort_per_line",
        # Smell
        "smell_count", "smell_categories",
    }
    missing = required - set(df.columns)
    assert not missing, f"PLAN §14.1 eksik sutunlar: {sorted(missing)}"

    # Dtype beklentileri (PLAN §14.1)
    int32_cols = (
        "stars", "contributor_count", "project_age_days",
        "commit_count", "bug_count", "n_authors",
        "churn_total", "max_single_churn", "recent_commits_90d",
        "loc", "lloc", "sloc", "comments", "multi", "blank",
        "single_comments", "num_functions", "commits_to_first_bug",
    )
    float32_cols = (
        "file_age_days", "avg_churn_per_commit",
        "cc_mean", "cc_max", "cc_total",
        "h_vocabulary", "h_length", "h_volume", "h_difficulty",
        "h_effort", "h_bugs", "h_time", "h_calculated_length",
        "maintainability_index", "comment_ratio", "doc_ratio",
        "complexity_density", "comment_per_function",
        "avg_function_length", "effort_per_line",
    )
    for col in int32_cols:
        assert df[col].dtype == "int32", f"{col}: int32 bekleniyor, {df[col].dtype}"
    for col in float32_cols:
        assert df[col].dtype == "float32", f"{col}: float32 bekleniyor, {df[col].dtype}"

    # bug_keyword: int8
    assert df["bug_keyword"].dtype == "int8"
    # Nullable tipler: bug_szz=Int8, smell_count=Int32
    assert str(df["bug_szz"].dtype) == "Int8"
    assert str(df["smell_count"].dtype) == "Int32"
    # String sutunlar
    assert df["file_path"].dtype == object
    assert df["project_name"].dtype == object
    assert df["smell_categories"].dtype == object

    # Degerler mantikli
    assert df["project_name"].iloc[0] == "user/demo"
    assert df["bug_szz"].iloc[0] == 1
    assert df["smell_count"].iloc[0] == 4


def test_process_project_szz_fallback_when_empty_dict(tmp_dirs):
    repos, projects = tmp_dirs
    repo = repos / "demo"
    (repo / ".git").mkdir(parents=True)

    files = ["x.py"]

    with patch("pipeline.project_processor.cloning.clone_repo",
               return_value=(repo, "basarili")), \
         patch("pipeline.project_processor.git_metrics.get_head_python_files",
               return_value=files), \
         patch("pipeline.project_processor.git_metrics.get_bulk_git_stats",
               return_value=_bulk_for(files)), \
         patch("pipeline.project_processor.static_metrics.calculate_metrics",
               side_effect=lambda *a, **kw: _static_ok()), \
         patch("pipeline.project_processor.static_metrics.calculate_derived",
               side_effect=lambda d: d), \
         patch("pipeline.project_processor.Path.read_text", return_value="x=1\n"), \
         patch("pipeline.project_processor._bug_fix_hashes", return_value=["h1"]), \
         patch("pipeline.project_processor.szz.compute_szz_labels",
               return_value={}), \
         patch("pipeline.project_processor.prospector_runner.run_prospector_batch",
               return_value={repo / "x.py": {"smell_count": 0, "categories": {}, "messages": []}}):
        result = project_processor.process_project(
            _PROJ, repos_dir=repos, projects_dir=projects,
        )
    assert result["status"] == "ok"
    assert result["szz_fallback"] is True
    # fallback kullanildi: bulk_for'da bug_count=1 → bug_szz=1 beklenir
    df = pd.read_parquet(result["parquet"])
    assert int(df["bug_szz"].iloc[0]) == 1
