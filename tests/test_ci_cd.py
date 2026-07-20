"""
test_ci_cd.py — CI/CD detection sinyalleri.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.ci_cd import _CI_CD_KEYS, detect_ci_cd_signals, empty_ci_cd_signals


def test_empty_signals_all_false():
    """empty_ci_cd_signals() tum anahtarlari False vermeli."""
    sig = empty_ci_cd_signals()
    assert set(sig.keys()) == set(_CI_CD_KEYS)
    assert all(v is False for v in sig.values())


def test_detect_no_signals_returns_all_false(tmp_path: Path):
    """Bos klasorde hicbir sinyal bulunmamali."""
    sig = detect_ci_cd_signals(tmp_path)
    assert sig == empty_ci_cd_signals()
    assert sig["is_devops_project"] is False


def test_detect_github_actions(tmp_path: Path):
    """`.github/workflows/` dizini varsa has_github_actions=True."""
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    sig = detect_ci_cd_signals(tmp_path)
    assert sig["has_github_actions"] is True
    assert sig["is_devops_project"] is True


def test_detect_dockerfile(tmp_path: Path):
    (tmp_path / "Dockerfile").write_text("FROM python:3.10\n")
    sig = detect_ci_cd_signals(tmp_path)
    assert sig["has_dockerfile"] is True
    assert sig["is_devops_project"] is True


def test_detect_compose_yml(tmp_path: Path):
    (tmp_path / "docker-compose.yml").write_text("version: '3'\n")
    sig = detect_ci_cd_signals(tmp_path)
    assert sig["has_compose"] is True


def test_detect_compose_yaml_alt(tmp_path: Path):
    """compose.yaml (yeni format) da yakalanmali."""
    (tmp_path / "compose.yaml").write_text("services:\n")
    sig = detect_ci_cd_signals(tmp_path)
    assert sig["has_compose"] is True


def test_detect_compose_dev_variant(tmp_path: Path):
    """docker-compose.dev.yml gibi varyantlar da yakalanmali (glob)."""
    (tmp_path / "docker-compose.dev.yml").write_text("services:\n")
    sig = detect_ci_cd_signals(tmp_path)
    assert sig["has_compose"] is True


def test_detect_travis(tmp_path: Path):
    (tmp_path / ".travis.yml").write_text("language: python\n")
    sig = detect_ci_cd_signals(tmp_path)
    assert sig["has_travis"] is True


def test_detect_jenkins(tmp_path: Path):
    (tmp_path / "Jenkinsfile").write_text("pipeline {}\n")
    sig = detect_ci_cd_signals(tmp_path)
    assert sig["has_jenkins"] is True


def test_detect_gitlab_ci(tmp_path: Path):
    (tmp_path / ".gitlab-ci.yml").write_text("stages:\n")
    sig = detect_ci_cd_signals(tmp_path)
    assert sig["has_gitlab_ci"] is True


def test_detect_pre_commit(tmp_path: Path):
    (tmp_path / ".pre-commit-config.yaml").write_text("repos:\n")
    sig = detect_ci_cd_signals(tmp_path)
    assert sig["has_pre_commit"] is True


def test_multiple_signals_combined(tmp_path: Path):
    """Birden fazla sinyal varsa tum True'lari + is_devops_project True."""
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "Dockerfile").write_text("FROM alpine\n")
    (tmp_path / ".pre-commit-config.yaml").write_text("repos:\n")

    sig = detect_ci_cd_signals(tmp_path)

    assert sig["has_github_actions"] is True
    assert sig["has_dockerfile"] is True
    assert sig["has_pre_commit"] is True
    assert sig["has_travis"] is False
    assert sig["has_jenkins"] is False
    assert sig["has_gitlab_ci"] is False
    assert sig["has_compose"] is False
    assert sig["is_devops_project"] is True


def test_nonexistent_path_returns_empty():
    """Olmayan path patlamamali, bos sinyaller donmeli."""
    sig = detect_ci_cd_signals(Path("/nonexistent/path/repo"))
    assert sig == empty_ci_cd_signals()


def test_file_instead_of_dir_returns_empty(tmp_path: Path):
    """Path bir dosyaysa (dizin degil) bos doner."""
    f = tmp_path / "not_a_repo.txt"
    f.write_text("hi")
    sig = detect_ci_cd_signals(f)
    assert sig == empty_ci_cd_signals()


def test_empty_workflows_dir_still_counts(tmp_path: Path):
    """Bos `.github/workflows/` bile DevOps niyeti gosterir."""
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    # Icinde dosya yok
    sig = detect_ci_cd_signals(tmp_path)
    assert sig["has_github_actions"] is True
