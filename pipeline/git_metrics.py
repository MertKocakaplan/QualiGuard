"""
git_metrics.py — Git log/churn/bug-keyword metrikleri.

V1'deki app/git_utils.py'den taşindi (PLAN §3.5). Bug keyword baseline
burada kalir; SZZ etiketi `pipeline.szz` icinde hesaplanir.

Tek git-log cagrisi tum dosyalarin istatistiklerini uretir (N sorgu yerine 1).
"""
from __future__ import annotations

import logging
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pipeline.config import GIT_LOG_TIMEOUT_SECONDS, RECENT_COMMIT_WINDOW_DAYS

logger = logging.getLogger(__name__)


# ── Filtre ve etiket regex'leri ──────────────────────────────────

SKIP_PATTERNS = (
    r"__init__\.py$",
    r"setup\.py$",
    r"conftest\.py$",
    r"manage\.py$",
    r"/migrations?/",
    r"/tests?/",
    r"test_[^/]+\.py$",
    r"[^/]+_test\.py$",
    r"/docs?/",
    r"/examples?/",
    r"/vendor/",
    r"conf\.py$",
    r"__main__\.py$",
    r"/\.",
)
SKIP_REGEX = re.compile("|".join(SKIP_PATTERNS))

BUG_KEYWORDS = re.compile(
    r"(?:"
    r"\b(fix(es|ed|ing)?|bug(s|gy)?|defect(s|ive)?|fault(s|y)?)\b"
    r"|\b(error(s)?|patch(es|ed|ing)?|resolv(e|es|ed|ing)?)\b"
    r"|\b(crash(es|ed|ing)?|regression(s)?)\b"
    r"|\b(hotfix|bugfix|workaround|repair(ed|ing)?|typo(s)?)\b"
    r"|\b(broken|failure(s)?|fail(s|ed|ing)?)\b"
    r"|\b(null.?pointer|npe|exception|traceback|segfault|overflow)\b"
    r"|^fix[\(:]"
    r"|\b(fix(?:es|ed)?|close[sd]?|resolve[sd]?)\s*#\d+"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

# Per-keyword gruplar (Antoniol et al. 2008)
BUG_KEYWORD_GROUPS: dict[str, tuple[str, ...]] = {
    "fix":     ("fix", "fixed", "fixes", "fixing", "hotfix", "bugfix"),
    "bug":     ("bug", "bugs", "buggy"),
    "error":   ("error", "errors"),
    "defect":  ("defect", "defects"),
    "issue":   ("issue", "issues"),
    "anomaly": ("anomaly", "anomalies"),
}
_BUG_KW_COMPILED: dict[str, re.Pattern] = {
    group: re.compile(
        r"|".join(rf"\b{w}\b" for w in words),
        re.IGNORECASE,
    )
    for group, words in BUG_KEYWORD_GROUPS.items()
}


def classify_bug_message(message: str) -> dict[str, int]:
    """Mesajda her keyword grubu icin 0/1 dondur."""
    msg = message or ""
    return {
        f"bug_kw_{group}": (1 if pat.search(msg) else 0)
        for group, pat in _BUG_KW_COMPILED.items()
    }

REFACTOR_KEYWORDS = re.compile(
    r"\b("
    r"refactor(ed|ing|s)?|cleanup|clean.?up|"
    r"rename(d|s|ing)?|reorganiz(e|ed|ing)|restructur(e|ed|ing)|"
    r"simplif(y|ied|ies|ying)|extract(ed|ing)?|inline(d|s)?|"
    r"tidy(ing)?|reformat(ted|ting)?"
    r")\b",
    re.IGNORECASE,
)


# ── Yardimci hesap fonksiyonlari (F3.3 – F3.5) ───────────────────

def gini_coefficient(values: list[int]) -> float:
    """
    Gini katsayisi (0 = esit dagilim, 1 = tek kiside konsantrasyon).
    Mockus et al. (2002) power-law contribution distribution.
    """
    if not values or sum(values) == 0:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    cum = sum((i + 1) * v for i, v in enumerate(sorted_v))
    total = sum(sorted_v)
    return (2 * cum) / (n * total) - (n + 1) / n


def inter_commit_time_cv(timestamps: list[int]) -> float:
    """
    Commit arasi surelerin coefficient of variation (std/mean).
    Yuksek = duzensiz gelistirme, dusuk = stabil kadans.
    """
    if len(timestamps) < 2:
        return 0.0
    sorted_ts = sorted(timestamps)
    deltas = [sorted_ts[i + 1] - sorted_ts[i] for i in range(len(sorted_ts) - 1)]
    mean = sum(deltas) / len(deltas)
    if mean == 0:
        return 0.0
    var = sum((d - mean) ** 2 for d in deltas) / len(deltas)
    return (var ** 0.5) / mean


def author_entropy(author_commits: dict[str, int]) -> float:
    """Shannon entropisi — yazar commit dagilimi (bit cinsinden)."""
    import math
    total = sum(author_commits.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in author_commits.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def bug_fix_density(bug_fix_count: int, kloc: float, age_years: float) -> float:
    """Bug-fix sayisi / KLOC / yil (issue density proxy). Mockus (2010)."""
    return bug_fix_count / max(kloc, 0.1) / max(age_years, 0.1)


# ── Public API ───────────────────────────────────────────────────

def should_skip_file(file_path: str) -> bool:
    """Test/init/migration gibi analiz disi dosya mi?"""
    return bool(SKIP_REGEX.search(file_path))


def is_bug_message(commit_message: str) -> bool:
    """Commit mesaji bug anahtar kelimelerini icerir mi?"""
    return bool(BUG_KEYWORDS.search(commit_message or ""))


def is_refactor_message(commit_message: str) -> bool:
    """Commit mesaji refactor/cleanup anahtar kelimelerini icerir mi?"""
    return bool(REFACTOR_KEYWORDS.search(commit_message or ""))


def get_repo_commit_summary(
    repo_path: Path,
    timeout: int = GIT_LOG_TIMEOUT_SECONDS,
) -> dict[str, int]:
    """
    Repo-level commit ozeti — PLAN §17.1 Project Health kartlari icin.

    `git log --pretty=format:%at|%s` cikisini okur, bug ve refactor
    commit'lerini sayar, son `RECENT_COMMIT_WINDOW_DAYS` icindekini
    da dondurur.

    Hata / timeout / bos repo -> tum degerler 0.
    """
    try:
        result = subprocess.run(
            ["git", "log", "--pretty=format:%at|%ae|%s", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.warning("get_repo_commit_summary basarisiz (%s): %s", repo_path, exc)
        return _empty_summary()

    if result.returncode != 0 or not result.stdout.strip():
        return _empty_summary()

    now       = datetime.now(timezone.utc)
    cutoff_ts = (now - timedelta(days=RECENT_COMMIT_WINDOW_DAYS)).timestamp()

    total    = 0
    bugs     = 0
    refacs   = 0
    reverts  = 0
    recent   = 0
    timestamps: list[int] = []
    author_commits: dict[str, int] = {}

    for line in result.stdout.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 2)
        ts_str  = parts[0] if len(parts) > 0 else ""
        author  = parts[1] if len(parts) > 1 else ""
        subject = parts[2] if len(parts) > 2 else ""

        total += 1
        if is_bug_message(subject):
            bugs += 1
        if is_refactor_message(subject):
            refacs += 1
        if re.match(r"^revert\b", subject.strip(), re.IGNORECASE):
            reverts += 1
        if author:
            author_commits[author] = author_commits.get(author, 0) + 1
        try:
            ts = int(ts_str)
            timestamps.append(ts)
            if ts >= cutoff_ts:
                recent += 1
        except ValueError:
            pass

    contribution_gini  = gini_coefficient(list(author_commits.values()))
    commit_cadence_cv  = inter_commit_time_cv(timestamps)
    commit_entropy     = author_entropy(author_commits)

    return {
        "total_commits":       total,
        "bug_fix_commits":     bugs,
        "refactor_commits":    refacs,
        "recent_commits_90d":  recent,
        # F3.3
        "refactor_ratio":      round(refacs / max(total, 1), 4),
        # F3.4
        "contribution_gini":   round(contribution_gini, 4),
        # F3.5
        "revert_count":             reverts,
        "inter_commit_time_cv":     round(commit_cadence_cv, 4),
        "author_entropy":           round(commit_entropy, 4),
    }


def _empty_summary() -> dict:
    return {
        "total_commits":       0,
        "bug_fix_commits":     0,
        "refactor_commits":    0,
        "recent_commits_90d":  0,
        "refactor_ratio":      0.0,
        "contribution_gini":   0.0,
        "revert_count":        0,
        "inter_commit_time_cv": 0.0,
        "author_entropy":      0.0,
    }


def get_head_python_files(repo_path: Path) -> list[str]:
    """HEAD'deki tum .py dosyalarinin repo-relative yollari."""
    try:
        result = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="replace",
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.warning("ls-tree basarisiz (%s): %s", repo_path, exc)
        return []

    if result.returncode != 0:
        return []

    files = result.stdout.strip().split("\n")
    return [f for f in files if f.endswith(".py") and f.strip()]


def get_bulk_git_stats(repo_path: Path, head_files: list[str]) -> dict[str, dict]:
    """
    Tek git-log cagrisi ile her dosyanin process metriklerini uret.

    Returns:
        {file_path: {
            'commit_count', 'bug_count', 'n_authors', 'file_age_days',
            'churn_total', 'avg_churn_per_commit', 'max_single_churn',
            'recent_commits_90d'
        }}
        Git hatasi veya head_files bossa {} doner.
    """
    if not head_files:
        return {}

    try:
        result = subprocess.run(
            ["git", "log", "--format=COMMIT|%H|%ae|%at|%s", "--numstat", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=GIT_LOG_TIMEOUT_SECONDS,
            encoding="utf-8",
            errors="replace",
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.warning("git-log basarisiz (%s): %s", repo_path, exc)
        return {}

    if result.returncode != 0:
        return {}

    head_set   = set(head_files)
    now        = datetime.now(timezone.utc)
    now_ts     = now.timestamp()
    cutoff_ts  = (now - timedelta(days=RECENT_COMMIT_WINDOW_DAYS)).timestamp()

    _kw_groups = list(BUG_KEYWORD_GROUPS.keys())

    file_stats: dict[str, dict] = {
        f: {
            "commit_count": 0,
            "bug_count":    0,
            "authors":      set(),
            "timestamps":   [],
            "churns":       [],
            **{f"bug_kw_{g}_count": 0 for g in _kw_groups},
        }
        for f in head_files
    }

    current_is_bug  = False
    current_ts      = 0
    current_author  = ""
    current_kw_hits: dict[str, int] = {}

    for line in result.stdout.split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("COMMIT|"):
            parts = line.split("|", 4)
            if len(parts) >= 4:
                current_author = parts[2] if len(parts) > 2 else ""
                try:
                    current_ts = int(parts[3])
                except (ValueError, IndexError):
                    current_ts = 0
                subject = parts[4] if len(parts) > 4 else ""
                current_is_bug  = is_bug_message(subject)
                current_kw_hits = classify_bug_message(subject)
        else:
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            added_str, deleted_str, fpath = parts
            if fpath not in head_set:
                continue
            st = file_stats.get(fpath)
            if st is None:
                continue
            st["commit_count"] += 1
            if current_is_bug:
                st["bug_count"] += 1
            for g in _kw_groups:
                st[f"bug_kw_{g}_count"] += current_kw_hits.get(f"bug_kw_{g}", 0)
            st["authors"].add(current_author)
            st["timestamps"].append(current_ts)
            try:
                st["churns"].append(int(added_str) + int(deleted_str))
            except ValueError:
                # Binary dosyalar "-" dondurur — gormezden gel
                pass

    results: dict[str, dict] = {}
    for fpath, st in file_stats.items():
        if st["commit_count"] == 0:
            continue
        churns     = st["churns"]
        timestamps = st["timestamps"]
        oldest     = min(timestamps) if timestamps else now_ts
        age        = max((now_ts - oldest) / 86400, 0.5)
        recent     = sum(1 for ts in timestamps if ts >= cutoff_ts)
        total_ch   = sum(churns) if churns else 0
        results[fpath] = {
            "commit_count":         st["commit_count"],
            "bug_count":            st["bug_count"],
            "n_authors":            len(st["authors"]),
            "file_age_days":        round(age, 1),
            "churn_total":          total_ch,
            "avg_churn_per_commit": round(total_ch / len(churns), 1) if churns else 0,
            "max_single_churn":     max(churns) if churns else 0,
            "recent_commits_90d":   recent,
            **{f"bug_kw_{g}_count": st[f"bug_kw_{g}_count"] for g in _kw_groups},
        }
    return results
