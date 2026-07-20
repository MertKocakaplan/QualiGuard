"""
test_config.py — pipeline.config sabitleri ve ensure_runtime_dirs.
"""
from __future__ import annotations

from pathlib import Path

from pipeline import config


def test_feature_counts_match_plan():
    """
    FEATURES_COMMIT sabit 35 (feature selection yalniz BUG/SMELL'i etkiler).

    BUG/SMELL, feature selection (analysis/05) sonrasi LEAN — kesin sayi 05
    ciktisina baglidir (ornek calismada 42->31, 48->28). Kirilgan sabit sayi
    yerine invariant test edilir: orijinal ust siniri (42/48) asmaz, bos olmaz.
    Leak invariant ayrica test_features_bug_excludes_keyword_counts'ta.
    """
    assert len(config.FEATURES_COMMIT) == 35, \
        f"T1 commit 35 olmali, {len(config.FEATURES_COMMIT)} bulundu"
    assert 0 < len(config.FEATURES_BUG) <= 42, \
        f"FEATURES_BUG 1..42 (lean) olmali, {len(config.FEATURES_BUG)} bulundu"
    assert 0 < len(config.FEATURES_SMELL) <= 48, \
        f"FEATURES_SMELL 1..48 (lean) olmali, {len(config.FEATURES_SMELL)} bulundu"


def test_features_bug_excludes_keyword_counts():
    """Regresyon (KRITIK leak korumasi): bug_kw_*_count FEATURES_BUG'a girmemeli.

    bug_keyword label'i sum(bug_kw_*_count)>0'dan turedigi icin bu sutunlar
    FEATURES_BUG'a girerse model trivial ~%99 F1'e sicrar (label leakage).
    Bu invariant feature selection'dan BAGIMSIZ HER ZAMAN gecerli olmali.

    NOT: Eski tasarimda bu sutunlar FEATURES_SMELL'de tutuluyordu (smell label'i
    keyword'den turemez — leak degil). Feature selection (05) bunlari importance≈0
    bularak smell'den de cikardi; artik smell'de olmalari ZORUNLU degil.
    """
    forbidden = {
        "bug_kw_fix_count", "bug_kw_bug_count", "bug_kw_error_count",
        "bug_kw_defect_count", "bug_kw_issue_count", "bug_kw_anomaly_count",
    }
    leaked = forbidden & set(config.FEATURES_BUG)
    assert not leaked, f"FEATURES_BUG'da leak sutun(lar): {leaked}"


def test_feature_names_unique():
    """Her feature seti icinde tekrar eden sutun ismi olmamali."""
    assert len(set(config.FEATURES_COMMIT)) == len(config.FEATURES_COMMIT)
    assert len(set(config.FEATURES_BUG))    == len(config.FEATURES_BUG)
    assert len(set(config.FEATURES_SMELL))  == len(config.FEATURES_SMELL)


def test_project_root_has_pipeline(tmp_path):
    """PROJECT_ROOT icinde pipeline/ dizini olmali."""
    assert (config.PROJECT_ROOT / "pipeline").is_dir()


def test_ensure_runtime_dirs_creates_directories(tmp_path, monkeypatch):
    """ensure_runtime_dirs tum runtime dizinlerini olustursun."""
    new_output = tmp_path / "out"
    monkeypatch.setattr(config, "OUTPUT_DIR",     new_output,                 raising=False)
    monkeypatch.setattr(config, "CHECKPOINT_DIR", new_output / "checkpoints", raising=False)
    monkeypatch.setattr(config, "PROJECTS_DIR",   new_output / "projects",    raising=False)
    monkeypatch.setattr(config, "LOGS_DIR",       new_output / "logs",        raising=False)
    monkeypatch.setattr(config, "FIGURES_DIR",    new_output / "figures",     raising=False)
    monkeypatch.setattr(config, "REPOS_DIR",      tmp_path / "repos",         raising=False)

    config.ensure_runtime_dirs()
    assert (new_output / "checkpoints").is_dir()
    assert (new_output / "projects").is_dir()
    assert (new_output / "logs").is_dir()
    assert (new_output / "figures").is_dir()
    assert (tmp_path / "repos").is_dir()
