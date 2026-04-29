"""
test_config.py — pipeline.config sabitleri ve ensure_runtime_dirs.
"""
from __future__ import annotations

from pathlib import Path

from pipeline import config


def test_feature_counts_match_plan():
    """F3.1 sonrasi: T1=31 (+2 cognitive), T2=T3=38."""
    assert len(config.FEATURES_COMMIT) == 31, \
        f"T1 commit 31 olmali, {len(config.FEATURES_COMMIT)} bulundu"
    assert len(config.FEATURES_BUG) == 38, \
        f"T2 bug 38 olmali, {len(config.FEATURES_BUG)} bulundu"
    assert len(config.FEATURES_SMELL) == 38, \
        f"T3 smell 38 olmali, {len(config.FEATURES_SMELL)} bulundu"


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
