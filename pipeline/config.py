"""
config.py — MetricHunter V2 sabitleri, esikler, yol ve tool ayarlari.

Tum magic number/string degerleri buraya toplanir. Diger modullerden
absolute import ile kullanilir:

    from pipeline.config import DEFAULT_MIN_AGE_DAYS
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# ── Proje kok yolu ─────────────────────────────────────────────────
# Bu dosya Final/v2/pipeline/config.py konumundadir.
# parents[1] = Final/v2 (proje koku).
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

OUTPUT_DIR: Final[Path]     = PROJECT_ROOT / "output"
CHECKPOINT_DIR: Final[Path] = OUTPUT_DIR / "checkpoints"
PROJECTS_DIR: Final[Path]   = OUTPUT_DIR / "projects"
LOGS_DIR: Final[Path]       = OUTPUT_DIR / "logs"
FIGURES_DIR: Final[Path]    = OUTPUT_DIR / "figures"
REPOS_DIR: Final[Path]      = PROJECT_ROOT / "repos"
MODELS_DIR: Final[Path]     = PROJECT_ROOT / "models"
ARCHIVE_DIR: Final[Path]    = PROJECT_ROOT / "archive"


def ensure_runtime_dirs() -> None:
    """Runtime'da yazim yapilacak dizinleri olustur (idempotent)."""
    for d in (OUTPUT_DIR, CHECKPOINT_DIR, PROJECTS_DIR, LOGS_DIR, FIGURES_DIR, REPOS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ── GitHub API ─────────────────────────────────────────────────────
GITHUB_API_BASE: Final[str]       = "https://api.github.com"
GITHUB_SEARCH_URL: Final[str]     = f"{GITHUB_API_BASE}/search/repositories"
GITHUB_REPO_URL: Final[str]       = f"{GITHUB_API_BASE}/repos"
GITHUB_RATELIMIT_URL: Final[str]  = f"{GITHUB_API_BASE}/rate_limit"
GITHUB_TOKEN_ENV: Final[str]      = "GITHUB_TOKEN"

GITHUB_ACCEPT: Final[str]         = "application/vnd.github+json"
GITHUB_API_VERSION: Final[str]    = "2022-11-28"

# GitHub search API tek sorguda 1000 sonuc verir. Bu ustunde kaymali sorgu.
GITHUB_SEARCH_MAX_PER_QUERY: Final[int] = 1000
GITHUB_SEARCH_PER_PAGE: Final[int]      = 100

# Rate limit davranisi
RATE_LIMIT_MIN_REMAINING: Final[int] = 10   # Bu esik altinda reset'e kadar sleep
RATE_LIMIT_BACKOFF: Final[tuple]     = (5, 15, 45, 135)  # 403 icin uslu bekleme
RATE_LIMIT_THROTTLE_SLEEP: Final[int] = 60  # 429 icin sabit bekleme
RATE_LIMIT_MAX_RETRIES: Final[int]    = 3
SERVER_ERROR_BACKOFF: Final[tuple]   = (2, 5, 10)  # 5xx icin linear

HTTP_TIMEOUT_SECONDS: Final[int] = 15


# ── Discovery kriterleri (varsayilan) ─────────────────────────────
DEFAULT_TARGET_COUNT: Final[int]      = 1000
DEFAULT_MIN_AGE_DAYS: Final[int]      = 180  # ~6 ay
DEFAULT_MAX_AGE_DAYS: Final[int]      = 365  # ~12 ay
DEFAULT_MAX_CONTRIBUTORS: Final[int]  = 10
DEFAULT_MIN_STARS: Final[int]         = 50
DEFAULT_LANGUAGE: Final[str]          = "python"

DISCOVERY_FLUSH_EVERY: Final[int]     = 50   # her N projede checkpoint


# ── Klonlama ───────────────────────────────────────────────────────
CLONE_TIMEOUT_SECONDS: Final[int]     = 600  # 10 dakika
CLONE_SINGLE_BRANCH: Final[bool]      = True


# ── Statik metrikler ──────────────────────────────────────────────
MAX_FILE_SIZE_BYTES: Final[int] = 500_000
MIN_SLOC_THRESHOLD: Final[int]  = 10


# ── Git metrikleri ────────────────────────────────────────────────
GIT_LOG_TIMEOUT_SECONDS: Final[int] = 300
RECENT_COMMIT_WINDOW_DAYS: Final[int] = 90


# ── SZZ (F2) ───────────────────────────────────────────────────────
SZZ_TIMEOUT_SECONDS: Final[int]       = 600


# ── Prospector (F2) ───────────────────────────────────────────────
PROSPECTOR_STRICTNESS: Final[str]     = "low"
PROSPECTOR_TIMEOUT_SECONDS: Final[int] = 90
PROSPECTOR_WORKERS: Final[int]         = 4


# ── Smell esikleri ─────────────────────────────────────────────────
SMELL_BINARY_PERCENTILE: Final[int]    = 80


# ── Feature sutun siralari ────────────────────────────────────────
# T1 commit (29 ozellik) — proje + statik + turetilmis
FEATURES_COMMIT: Final[tuple[str, ...]] = (
    "loc", "lloc", "sloc", "comments", "multi", "blank", "single_comments",
    "cc_mean", "cc_max", "cc_total", "num_functions",
    "h_vocabulary", "h_length", "h_volume", "h_difficulty",
    "h_effort", "h_bugs", "h_time", "h_calculated_length",
    "maintainability_index",
    "comment_ratio", "doc_ratio",
    "complexity_density", "comment_per_function",
    "avg_function_length", "effort_per_line",
    "stars", "contributor_count", "project_age_days",
)

# T2 bug (36 ozellik) — T1 + process metrikleri
FEATURES_PROCESS: Final[tuple[str, ...]] = (
    "commit_count", "bug_count", "n_authors", "file_age_days",
    "churn_total", "avg_churn_per_commit", "max_single_churn",
    "recent_commits_90d",
)
FEATURES_BUG: Final[tuple[str, ...]]   = FEATURES_COMMIT + (
    "commit_count", "n_authors", "file_age_days",
    "churn_total", "avg_churn_per_commit", "max_single_churn",
    "recent_commits_90d",
)
# T3 smell — T2 ile ayni ozellik seti
FEATURES_SMELL: Final[tuple[str, ...]] = FEATURES_BUG


# ── Logging ────────────────────────────────────────────────────────
LOG_FORMAT: Final[str] = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
LOG_DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"


# ── Env okuma yardimcisi ─────────────────────────────────────────
def github_token() -> str:
    """Ortam degiskeninden guncel GitHub token'i al (bos string olabilir)."""
    return os.environ.get(GITHUB_TOKEN_ENV, "").strip()
