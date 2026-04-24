"""
test_analyzer_prospector.py — app.analyzer.analyze_repo F7 akisi.

Heavy cagrilar (clone/git/predictor/prospector) mock'lanir; analyze_repo
sonuc sozlugu icinde project_health + smell_summary + prospector
entegrasyonunu dogrulariz.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from app import analyzer


# ── Yardimcilar ──────────────────────────────────────────────────

_PROJ = {
    "full_name":         "user/demo",
    "stars":             42,
    "contributor_count": 3,
    "project_age_days":  200,
    "default_branch":    "main",
}


def _static_ok() -> dict:
    return {
        "loc": 30, "lloc": 20, "sloc": 25, "comments": 3,
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


_COMMIT_SUMMARY = {
    "total_commits":      80,
    "bug_fix_commits":    10,
    "refactor_commits":   4,
    "recent_commits_90d": 30,
}


@pytest.fixture
def _common_patches(tmp_path):
    """clone/ls-tree/git + predictor mocks — her testte ortak."""
    repo = tmp_path / "demo"
    repo.mkdir()
    files = ["app/a.py", "app/b.py"]

    feature_names = {
        "commit": ["loc", "cc_mean"],
        "bug":    ["loc", "cc_mean"],
        "smell":  ["loc", "cc_mean"],
    }

    patches = [
        patch("app.analyzer.get_project_info", return_value=_PROJ),
        patch("app.analyzer.clone_repo", return_value=(repo, "ok")),
        patch("app.analyzer.get_head_python_files", return_value=files),
        patch("app.analyzer.get_bulk_git_stats", return_value=_bulk_for(files)),
        patch("app.analyzer.get_repo_commit_summary",
              return_value=dict(_COMMIT_SUMMARY)),
        patch("app.analyzer.calculate_metrics", side_effect=lambda *a, **k: _static_ok()),
        patch("app.analyzer.calculate_derived", side_effect=lambda d: d),
        patch("app.analyzer._read_file", return_value="def f():\n  return 1\n"),
        patch("app.analyzer.predictor.get_feature_names", return_value=feature_names),
        patch("app.analyzer.predictor.predict_commit",
              return_value=(np.array([1, 0]), np.array([0.9, 0.2]))),
        patch("app.analyzer.predictor.predict_bug",
              return_value=(np.array([0, 1]), np.array([0.3, 0.8]))),
        patch("app.analyzer.predictor.predict_smell",
              return_value=(np.array([1, 1]), np.array([0.7, 0.85]))),
        patch("app.analyzer.predictor.smell_available", return_value=True),
    ]
    for p in patches:
        p.start()
    yield repo, files
    for p in patches:
        p.stop()


def _run_analyzer(prospector_enabled: bool):
    events = []
    def cb(pct, msg):
        events.append((pct, msg))
    result = analyzer.analyze_repo(
        "https://github.com/user/demo",
        progress_callback=cb,
        prospector_enabled=prospector_enabled,
    )
    return result, events


# ── Testler ──────────────────────────────────────────────────────

def test_analyzer_with_prospector_full_payload(_common_patches):
    _repo, files = _common_patches

    pres = {
        _repo / "app/a.py": {"smell_count": 8, "categories": {"pylint": 8}, "messages": []},
        _repo / "app/b.py": {"smell_count": 3, "categories": {"pylint": 3}, "messages": []},
    }
    with patch("pipeline.prospector_runner.run_prospector_batch", return_value=pres):
        result, events = _run_analyzer(prospector_enabled=True)

    assert result["error"] is None
    assert result["prospector_enabled"] is True
    assert len(result["files"]) == 2

    # Her dosyada 3 tahmin de var
    for f in result["files"]:
        assert f["commit_pred"] in (0, 1)
        assert f["bug_pred"] in (0, 1)
        assert f["smell_pred"] in (0, 1)
        assert f["prospector_count"] in (3, 8)

    # Summary sayaclari
    assert result["summary"]["has_smell_risk"] == 2

    # project_health sozlugu
    h = result["project_health"]
    assert h["total_commits"] == 80
    assert h["bug_fix_commits"] == 10
    # defect density: toplam bug_count = 2 (her dosya 1), toplam loc = 60
    #   2 / (60/1000) = 33.33
    assert h["defect_density_per_kloc"] == pytest.approx(33.33, abs=0.1)
    assert h["refactor_ratio"] == pytest.approx(0.05, abs=0.001)

    # smell_summary
    s = result["smell_summary"]
    assert s["prospector_enabled"] is True
    assert s["total_smells"] == 11
    assert s["ml_smell_risk_count"] == 2
    # refactor priority: bug_pred==1 AND smell_pred==1 → yalniz 2. dosya
    assert s["refactor_priority_count"] == 1
    assert len(s["top_smelly_files"]) == 2

    # Progress callback sonu 100
    assert events[-1][0] == 100


def test_analyzer_without_prospector_skips_call(_common_patches):
    _repo, _files = _common_patches

    with patch("pipeline.prospector_runner.run_prospector_batch") as m_pros:
        result, _ = _run_analyzer(prospector_enabled=False)
        m_pros.assert_not_called()

    assert result["prospector_enabled"] is False
    s = result["smell_summary"]
    assert s["prospector_enabled"] is False
    assert s["total_smells"] == 0
    # ML smell tahmini hala calisti
    assert s["ml_smell_risk_count"] == 2
    # Dosya satirlarinda prospector_count olmamali
    for f in result["files"]:
        assert "prospector_count" not in f


def test_analyzer_prospector_import_error_soft_fails(_common_patches):
    """Prospector kurulu degilse analiz devam etmeli, smell_summary
    prospector_enabled=False donmeli."""
    _repo, _files = _common_patches

    with patch.object(analyzer, "_run_prospector_safe", return_value={}):
        result, _ = _run_analyzer(prospector_enabled=True)

    # prospector_enabled bayragi True (kullanici istedi) ama
    # smell_summary.prospector_enabled False (cikti yok)
    assert result["prospector_enabled"] is True
    assert result["smell_summary"]["prospector_enabled"] is False
    assert result["smell_summary"]["total_smells"] == 0
    # Flask yine de basarili bir sonuc sozlugu donmeli
    assert result["error"] is None
    assert len(result["files"]) == 2
