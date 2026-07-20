"""
ci_cd.py — CI/CD pipeline detection (DevOps practice signals).

PLAN §F6+ ek: docx'in "Yazilim Gelistirme Yaklasimi: Agile (DevOps)"
kisitinin veri-seviyesinde dogrulanmasi icin. Repo kokunde standart
CI/CD ve container artifact'lari aranir; her dosya satirina 8 boolean
olarak yazilir (project-level alanlar tum dosyalara replikasyon).

Paper iddiasi: "X% of N projects use CI/CD, validating DevOps focus."

Kontrol edilen sinyaller (Planning P6):
    has_github_actions  — .github/workflows/ dizini
    has_travis          — .travis.yml
    has_jenkins         — Jenkinsfile
    has_gitlab_ci       — .gitlab-ci.yml
    has_dockerfile      — Dockerfile
    has_compose         — docker-compose*.yml veya .yaml
    has_pre_commit      — .pre-commit-config.yaml

Turetilmis:
    is_devops_project   — yukaridakilerden en az biri True
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)


# Tek-dosya kontrolleri: dosya var/yok
_FILE_SIGNALS: Final[dict[str, str]] = {
    "has_travis":     ".travis.yml",
    "has_jenkins":    "Jenkinsfile",
    "has_gitlab_ci":  ".gitlab-ci.yml",
    "has_dockerfile": "Dockerfile",
    "has_pre_commit": ".pre-commit-config.yaml",
}

_CI_CD_KEYS: Final[tuple[str, ...]] = (
    "has_github_actions",
    "has_travis",
    "has_jenkins",
    "has_gitlab_ci",
    "has_dockerfile",
    "has_compose",
    "has_pre_commit",
    "is_devops_project",
)


def detect_ci_cd_signals(repo_path: Path) -> dict[str, bool]:
    """
    Repo kokunde DevOps practice sinyallerini tespit et.

    Args:
        repo_path: Klonlanmis git repo yolu.

    Returns:
        8 anahtar (7 ham + 1 turetilmis) boolean sozluk. Hicbir sinyal
        yoksa hepsi False. Hata (OSError) durumunda da bos sozluk.
    """
    try:
        repo = Path(repo_path)
        if not repo.is_dir():
            return empty_ci_cd_signals()
    except OSError as exc:
        logger.debug("ci_cd: repo path erisilemedi (%s): %s", repo_path, exc)
        return empty_ci_cd_signals()

    out: dict[str, bool] = {}

    # Tek-dosya sinyalleri
    for key, fname in _FILE_SIGNALS.items():
        out[key] = (repo / fname).is_file()

    # GitHub Actions: .github/workflows/ dizini (icinde dosya olmasi yeterli degil,
    # bos dizin de DevOps niyeti gosterir — is_dir kontrolu yeterli)
    out["has_github_actions"] = (repo / ".github" / "workflows").is_dir()

    # Docker Compose: birden fazla isim varyasyonu (hem .yml hem .yaml,
    # hem docker-compose hem compose; iki extension da kontrol edilir)
    out["has_compose"] = (
        any(repo.glob("docker-compose*.yml"))
        or any(repo.glob("docker-compose*.yaml"))
        or any(repo.glob("compose.yml"))
        or any(repo.glob("compose.yaml"))
    )

    # Turetilmis: en az bir DevOps sinyali var mi?
    out["is_devops_project"] = any(
        v for k, v in out.items() if k != "is_devops_project"
    )

    return out


def empty_ci_cd_signals() -> dict[str, bool]:
    """Bos/hatali repo durumu icin tum sinyaller False."""
    return {key: False for key in _CI_CD_KEYS}


__all__ = ["detect_ci_cd_signals", "empty_ci_cd_signals", "_CI_CD_KEYS"]
