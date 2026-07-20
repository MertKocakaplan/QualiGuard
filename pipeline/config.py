"""
config.py — QualiGuard V2 sabitleri, esikler, yol ve tool ayarlari.

Tum magic number/string degerleri buraya toplanir. Diger modullerden
absolute import ile kullanilir:

    from pipeline.config import DEFAULT_MIN_AGE_DAYS
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# ── Proje kok yolu ─────────────────────────────────────────────────
# Bu dosya <proje-koku>/pipeline/config.py konumundadir.
# parents[1] = proje koku.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

OUTPUT_DIR: Final[Path]     = PROJECT_ROOT / "output"
CHECKPOINT_DIR: Final[Path] = OUTPUT_DIR / "checkpoints"
PROJECTS_DIR: Final[Path]   = OUTPUT_DIR / "projects"
LOGS_DIR: Final[Path]       = OUTPUT_DIR / "logs"
FIGURES_DIR: Final[Path]    = OUTPUT_DIR / "figures"
REPOS_DIR: Final[Path]      = PROJECT_ROOT / "repos"
MODELS_DIR: Final[Path]     = PROJECT_ROOT / "models"
ARCHIVE_DIR: Final[Path]    = PROJECT_ROOT / "archive"


# ── .env loader (modul yuklenirken bir kez calisir) ───────────────
# pipeline.config'i import eden TUM girdi noktalari (run.py, scripts.collect,
# analysis/*.py, tests) otomatik olarak proje kokundeki .env'i ortama yukler.
# Idempotent: zaten os.environ'da olan key'lerin uzerine yazmaz.

def _load_dotenv(env_path: Path = PROJECT_ROOT / ".env") -> None:
    """
    Proje kokundeki .env'i os.environ'a yukle. Dosya yoksa veya bozuksa
    sessizce gec; ekrana hata basmaz.

    Format: KEY=value (her satir, tirnaklar opsiyonel, # ile yorum).
    """
    if not env_path.exists():
        return
    try:
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass  # Bozuk dosya / izin hatasi — sessizce devam


_load_dotenv()


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


# ── Prospector (deprecated — sample validation icin tutuluyor) ────
PROSPECTOR_STRICTNESS: Final[str]      = "low"
PROSPECTOR_TIMEOUT_SECONDS: Final[int] = 90
PROSPECTOR_WORKERS: Final[int]         = 4
PROSPECTOR_ENABLED_FOR_VALIDATION: Final[bool] = False  # --use-prospector ile aktif


# ── Code smell esikleri (Fowler 1999 + Lanza-Marinescu 2006) ──────
LONG_METHOD_LOC: Final[int]          = 50
LARGE_CLASS_LOC: Final[int]          = 500
LARGE_CLASS_METHOD_COUNT: Final[int] = 10
LONG_PARAM_COUNT: Final[int]         = 5
NESTING_DEPTH: Final[int]            = 4
HIGH_CC: Final[int]                  = 10
LOW_MI: Final[int]                   = 20
GOD_FUNC_CC: Final[int]              = 15
GOD_FUNC_LOC: Final[int]             = 80

# ── Smell binary label ──────────────────────────────────────────────
SMELL_BINARY_PERCENTILE: Final[int]    = 80


# ── Feature sutun siralari ────────────────────────────────────────
# FEATURES_COMMIT (35 ozellik) — proje + statik + turetilmis + cognitive + repo-history.
# NOT: V2.1'de T1 commit standalone task'i kaldirildi (quality gate odagi yok).
# Bu tuple "temel feature seti" olarak korunuyor cunku FEATURES_BUG ve
# FEATURES_SMELL bunun ustune eklenerek tanimlaniyor. Ismi tarihsel.
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
    "cognitive_complexity_total", "cognitive_complexity_max",  # F3.1
    # F3.5 — process-history proxies (Mockus et al. 2002)
    "revert_count", "inter_commit_time_cv", "author_entropy", "bug_fix_density",
)

# T2 bug (44 ozellik) — T1 + process + keyword counts
FEATURES_PROCESS: Final[tuple[str, ...]] = (
    "commit_count", "bug_count", "n_authors", "file_age_days",
    "churn_total", "avg_churn_per_commit", "max_single_churn",
    "recent_commits_90d",
)
# T2 bug — bug_kw_*_count'lar burada YOK: bug_keyword label tanimi
# (bug_count = sum(bug_kw_*_count) > 0) bu sutunlardan turedigi icin
# modele dahil edilmeleri label leakage'a yol acar (F1 ~%99'a yapay siçrar).
# Bug_szz da yuksek olasilikla ayni commit'lerden turetildigi icin riskli.
# Smell ise farkli (label smell_count'tan turer); bug_kw_*'lar smell tahmininde
# dogal korelasyondur (Tantithamthavorn et al. 2017) — FEATURES_SMELL'de tutulur.
FEATURES_BUG: Final[tuple[str, ...]] = (
    "loc",
    "multi",
    "single_comments",
    "cc_mean",
    "cc_max",
    "cc_total",
    "num_functions",
    "h_volume",
    "maintainability_index",
    "comment_ratio",
    "doc_ratio",
    "complexity_density",
    "comment_per_function",
    "avg_function_length",
    "effort_per_line",
    "stars",
    "contributor_count",
    "project_age_days",
    "cognitive_complexity_total",
    "cognitive_complexity_max",
    "revert_count",
    "inter_commit_time_cv",
    "author_entropy",
    "bug_fix_density",
    "commit_count",
    "n_authors",
    "file_age_days",
    "churn_total",
    "avg_churn_per_commit",
    "max_single_churn",
    "recent_commits_90d",
)
# T3 smell — FEATURES_BUG + bug keyword separation
# (Antoniol et al. 2008'in ayrik kategorileri; bug-prone <-> smell-prone
# ilişkisi Tantithamthavorn 2017'ye gore meşru korelasyondur, leak değil)
FEATURES_SMELL: Final[tuple[str, ...]] = (
    "lloc",
    "comments",
    "multi",
    "cc_mean",
    "cc_max",
    "num_functions",
    "h_volume",
    "h_effort",
    "maintainability_index",
    "comment_ratio",
    "doc_ratio",
    "complexity_density",
    "comment_per_function",
    "avg_function_length",
    "stars",
    "contributor_count",
    "project_age_days",
    "cognitive_complexity_total",
    "cognitive_complexity_max",
    "inter_commit_time_cv",
    "author_entropy",
    "bug_fix_density",
    "commit_count",
    "file_age_days",
    "churn_total",
    "avg_churn_per_commit",
    "max_single_churn",
    "bug_kw_anomaly_count",
)


# ── Logging ────────────────────────────────────────────────────────
LOG_FORMAT: Final[str] = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
LOG_DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"


# ── Env okuma yardimcisi ─────────────────────────────────────────
def github_token() -> str:
    """Ortam degiskeninden guncel GitHub token'i al (bos string olabilir)."""
    return os.environ.get(GITHUB_TOKEN_ENV, "").strip()
