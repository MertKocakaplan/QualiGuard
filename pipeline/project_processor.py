"""
project_processor.py — Tek proje icin tam islem hatti (F3 process fazi).

Akis:
    1. Clone (cloning.clone_repo)
    2. HEAD python dosyalari (git_metrics.get_head_python_files + skip filtre)
    3. Bulk git stats (git_metrics.get_bulk_git_stats)
    4. Statik metrikler (static_metrics.calculate_metrics + calculate_derived)
    5. SZZ (szz.compute_szz_labels) — opsiyonel
    6. Prospector batch (prospector_runner.run_prospector_batch) — opsiyonel
    7. DataFrame insa (PLAN §14.1 semasi)
    8. output/projects/<safe_name>.parquet (atomic yazim)

`process_project()` her cagrisinda bagimsizdir. Hata durumunda
    {"status": "failed", "error": ...}
doner, caller pipeline'i devam ettirir.

NOT: `commits_to_first_bug` su an -1 ile doldurulur. F6'da
`commits_before_bug.compute_stats` ile doldurulup project_stats.json'a
yazilacak.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from pipeline import cloning, code_smells, git_metrics, static_metrics, szz
from pipeline.config import (
    FEATURES_BUG,
    PROJECTS_DIR,
    PROSPECTOR_WORKERS,
    REPOS_DIR,
    SZZ_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


# ── Dahili yardimcilar ──────────────────────────────────────────────

def _bug_fix_hashes(repo_path: Path) -> list[str]:
    """
    HEAD'den geri dogru bug keyword iceren commit hash'leri.

    Tek `git log` cagrisi; regex git_metrics.BUG_KEYWORDS ile eslesenler.
    Pilot scripti ile ayni mantik — F3 production hatti bu hesabi
    processor icine aldi (scripts'de kopya gerekmesin).
    """
    try:
        result = subprocess.run(
            ["git", "log", "--format=%H%x1f%s", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=300,
            encoding="utf-8",
            errors="replace",
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("git log basarisiz (%s): %s", repo_path, exc)
        return []
    if result.returncode != 0:
        return []

    hashes: list[str] = []
    for line in result.stdout.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split("\x1f", 1)
        if len(parts) != 2:
            continue
        h, subject = parts
        if git_metrics.is_bug_message(subject):
            hashes.append(h)
    return hashes


def _atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    """Temp dosyaya yaz -> os.replace. Yarim parquet kalmaz."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{out_path.name}.", suffix=".tmp", dir=str(out_path.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _row_from_file(
    file_path_rel: str,
    *,
    project: dict,
    bulk: dict[str, dict],
    static_map: dict[str, dict],
    bug_szz_map: dict[str, int],
    smell_map: dict[str, dict],
    repo_summary: dict,
    include_szz: bool,
    include_smells: bool,
) -> Optional[dict[str, Any]]:
    """
    Tek dosya icin §14.1 semasina uygun kayit uret.

    static_map[file] None ise (radon parse hatasi / cok kucuk dosya),
    satir uretilmez. bulk[file] yoksa 0 default'larla devam edilir.
    """
    stat = static_map.get(file_path_rel)
    if stat is None:
        return None

    b = bulk.get(file_path_rel) or {}
    row: dict[str, Any] = {
        "file_path":         file_path_rel,
        "project_name":      project.get("full_name", ""),
        "stars":             int(project.get("stars") or 0),
        "contributor_count": int(project.get("contributor_count") or 0),
        "project_age_days":  int(project.get("project_age_days") or 0),

        "commit_count":         int(b.get("commit_count", 0)),
        "bug_count":            int(b.get("bug_count", 0)),
        "n_authors":            int(b.get("n_authors", 0)),
        "file_age_days":        float(b.get("file_age_days", 0.0)),
        "churn_total":          int(b.get("churn_total", 0)),
        "avg_churn_per_commit": float(b.get("avg_churn_per_commit", 0.0)),
        "max_single_churn":     int(b.get("max_single_churn", 0)),
        "recent_commits_90d":   int(b.get("recent_commits_90d", 0)),

        # Bug keyword separation (F3.2)
        "bug_kw_fix_count":     int(b.get("bug_kw_fix_count", 0)),
        "bug_kw_bug_count":     int(b.get("bug_kw_bug_count", 0)),
        "bug_kw_error_count":   int(b.get("bug_kw_error_count", 0)),
        "bug_kw_defect_count":  int(b.get("bug_kw_defect_count", 0)),
        "bug_kw_issue_count":   int(b.get("bug_kw_issue_count", 0)),
        "bug_kw_anomaly_count": int(b.get("bug_kw_anomaly_count", 0)),

        # Process-history proxies — repo-level, her satira yansitilir (F3.5)
        "revert_count":         int(repo_summary.get("revert_count", 0)),
        "inter_commit_time_cv": float(repo_summary.get("inter_commit_time_cv", 0.0)),
        "author_entropy":       float(repo_summary.get("author_entropy", 0.0)),
        "bug_fix_density":      float(repo_summary.get("bug_fix_density", 0.0)),

        # Etiketler
        "bug_keyword":          1 if b.get("bug_count", 0) > 0 else 0,
        "commits_to_first_bug": -1,  # F6'da dolduralacak
    }

    # SZZ (None -> nullable)
    if include_szz:
        row["bug_szz"] = int(bug_szz_map.get(file_path_rel, 0))
    else:
        row["bug_szz"] = None

    # Static + derived
    for key, value in stat.items():
        row[key] = value

    # Code smells (AST + radon)
    if include_smells:
        sm = smell_map.get(file_path_rel) or {}
        row["smell_count"]               = sm.get("smell_count", 0)
        row["smell_long_method"]         = sm.get("smell_long_method", 0)
        row["smell_large_class"]         = sm.get("smell_large_class", 0)
        row["smell_long_param_list"]     = sm.get("smell_long_param_list", 0)
        row["smell_deep_nesting"]        = sm.get("smell_deep_nesting", 0)
        row["smell_high_complexity"]     = sm.get("smell_high_complexity", 0)
        row["smell_low_maintainability"] = sm.get("smell_low_maintainability", 0)
        row["smell_god_function"]        = sm.get("smell_god_function", 0)
    else:
        row["smell_count"]               = None
        row["smell_long_method"]         = None
        row["smell_large_class"]         = None
        row["smell_long_param_list"]     = None
        row["smell_deep_nesting"]        = None
        row["smell_high_complexity"]     = None
        row["smell_low_maintainability"] = None
        row["smell_god_function"]        = None

    return row


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """§14.1 Parquet tip beklentileri."""
    int32_cols = (
        "stars", "contributor_count", "project_age_days",
        "commit_count", "bug_count", "n_authors",
        "churn_total", "max_single_churn", "recent_commits_90d",
        "loc", "lloc", "sloc", "comments", "multi", "blank",
        "single_comments", "num_functions",
        "commits_to_first_bug",
        "cognitive_complexity_total", "cognitive_complexity_max",  # F3.1
        "bug_kw_fix_count", "bug_kw_bug_count", "bug_kw_error_count",  # F3.2
        "bug_kw_defect_count", "bug_kw_issue_count", "bug_kw_anomaly_count",
        "revert_count",  # F3.5
    )
    float32_cols = (
        "file_age_days", "avg_churn_per_commit",
        "cc_mean", "cc_max", "cc_total",
        "h_vocabulary", "h_length", "h_volume", "h_difficulty",
        "h_effort", "h_bugs", "h_time", "h_calculated_length",
        "maintainability_index",
        "comment_ratio", "doc_ratio",
        "complexity_density", "comment_per_function",
        "avg_function_length", "effort_per_line",
        "inter_commit_time_cv", "author_entropy", "bug_fix_density",  # F3.5
    )
    int8_cols = ("bug_keyword",)
    # Nullable: bug_szz (Int8); smell sütunlari (Int32)
    smell_nullable_cols = (
        "smell_count", "smell_long_method", "smell_large_class",
        "smell_long_param_list", "smell_deep_nesting",
        "smell_high_complexity", "smell_low_maintainability",
        "smell_god_function",
    )

    for col in int32_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int32")
    for col in float32_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float32")
    for col in int8_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int8")

    if "bug_szz" in df.columns:
        df["bug_szz"] = pd.to_numeric(df["bug_szz"], errors="coerce").astype("Int8")
    for col in smell_nullable_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")

    return df


# ── Public API ──────────────────────────────────────────────────────

def process_project(
    project: dict,
    *,
    skip_szz: bool = False,
    skip_smells: bool = False,
    skip_prospector: bool = False,  # deprecated alias for skip_smells
    workers: int = PROSPECTOR_WORKERS,
    repos_dir: Path = REPOS_DIR,
    projects_dir: Path = PROJECTS_DIR,
) -> dict:
    """
    Bir projeyi bastan sona isle, per-project parquet yaz, ozet dict'i dondur.

    Args:
        project: discovery.search_projects() urettigi kayit
                 (`full_name`, `clone_url`, `stars`, `contributor_count`,
                 `project_age_days`, `default_branch` ...).
        skip_szz: True ise SZZ calistirilmaz, bug_szz=None.
        skip_smells: True ise smell tespiti calistirilmaz, smell_* sutunlari None.
        skip_prospector: Deprecated alias for skip_smells (backward compat).
        workers: Kullanilmiyor (backward compat, AST multiprocessing gerektirmiyor).
        repos_dir: Klonlarin toplandigi ebeveyn dizin.
        projects_dir: Per-project parquet'lerin yazildigi dizin.

    Returns (§14.4 ile uyumlu):
        {
            "status": "ok" | "failed" | "empty",
            "files": int,                 # yazilan satir sayisi
            "total_loc": int,
            "bugs_keyword": int,
            "bugs_szz": int,
            "smells_total": int,
            "smells_missing": int,
            "parquet": str,               # dosya yolu (ok ise)
            "timing": {...},              # alt adim sureleri
            "completed_at": "iso8601",
            "error": str,                 # failed ise
        }
    """
    _skip_smells = skip_smells or skip_prospector

    name = project.get("full_name") or "unknown"
    started = time.monotonic()
    logger.info("── %s (stars=%d) ──", name, project.get("stars", -1))

    timings: dict[str, float] = {}

    # 1) Klon
    t0 = time.monotonic()
    repo_path, clone_status = cloning.clone_repo(project["clone_url"], repos_dir)
    timings["clone_secs"] = round(time.monotonic() - t0, 1)
    if repo_path is None:
        return {
            "status":       "failed",
            "error":        f"clone: {clone_status}",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "timing":       timings,
        }

    # 2) HEAD dosyalari + skip filtre
    head_all = git_metrics.get_head_python_files(repo_path)
    head_files = [f for f in head_all if not git_metrics.should_skip_file(f)]
    if not head_files:
        return {
            "status":       "empty",
            "error":        "no_python_files",
            "files":        0,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "timing":       timings,
        }

    # 3) Bulk git stats + repo-level commit summary (F3.5)
    t0 = time.monotonic()
    bulk         = git_metrics.get_bulk_git_stats(repo_path, head_files)
    repo_summary = git_metrics.get_repo_commit_summary(repo_path)
    timings["git_secs"] = round(time.monotonic() - t0, 1)

    # 4) Statik metrikler (her dosya icin)
    t0 = time.monotonic()
    static_map: dict[str, dict] = {}
    for rel in head_files:
        abs_path = repo_path / rel
        try:
            source = abs_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeError) as exc:
            logger.debug("dosya okunamadi %s: %s", rel, exc)
            continue
        m = static_metrics.calculate_metrics(source)
        if m is None:
            continue
        static_metrics.calculate_derived(m)
        static_map[rel] = m
    timings["static_secs"] = round(time.monotonic() - t0, 1)

    if not static_map:
        return {
            "status":       "empty",
            "error":        "no_parseable_files",
            "files":        0,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "timing":       timings,
        }

    # bug_fix_density: repo-level, LOC tanimlandi (F3.5)
    total_kloc = sum(m.get("loc", 0) for m in static_map.values()) / 1000
    age_years  = max(project.get("project_age_days", 365), 1) / 365
    repo_summary["bug_fix_density"] = git_metrics.bug_fix_density(
        repo_summary.get("bug_fix_commits", 0), total_kloc, age_years,
    )

    # 5) SZZ (opsiyonel)
    bug_szz_map: dict[str, int] = {}
    szz_fallback = False
    if not skip_szz:
        t0 = time.monotonic()
        fix_hashes = _bug_fix_hashes(repo_path)
        bug_szz_map = szz.compute_szz_labels(
            repo_path, head_files, fix_hashes,
            timeout_seconds=SZZ_TIMEOUT_SECONDS,
        )
        timings["szz_secs"] = round(time.monotonic() - t0, 1)
        if not bug_szz_map:
            logger.warning("[%s] SZZ bos dondu, bug_keyword fallback.", name)
            bug_szz_map = {
                f: 1 if (bulk.get(f, {}).get("bug_count", 0) > 0) else 0
                for f in head_files
            }
            szz_fallback = True
    # skip_szz=True ise bug_szz_map bos, _row_from_file None ile doldurur

    # 6) Code smells — AST + radon (opsiyonel)
    smell_by_rel: dict[str, dict] = {}
    smells_missing = 0
    if not _skip_smells:
        t0 = time.monotonic()
        abs_paths = [repo_path / f for f in head_files if f in static_map]
        smell_results = code_smells.detect_smells_batch(abs_paths)
        timings["smells_secs"] = round(time.monotonic() - t0, 1)
        for abs_p, result in smell_results.items():
            try:
                rel = str(Path(abs_p).relative_to(repo_path)).replace("\\", "/")
            except ValueError:
                continue
            smell_by_rel[rel] = result

    # 7) DataFrame insa
    rows: list[dict] = []
    for rel in head_files:
        row = _row_from_file(
            rel,
            project=project,
            bulk=bulk,
            static_map=static_map,
            bug_szz_map=bug_szz_map,
            smell_map=smell_by_rel,
            repo_summary=repo_summary,
            include_szz=not skip_szz,
            include_smells=not _skip_smells,
        )
        if row is not None:
            rows.append(row)

    if not rows:
        return {
            "status":       "empty",
            "error":        "no_rows_after_merge",
            "files":        0,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "timing":       timings,
        }

    df = pd.DataFrame(rows)
    df = _coerce_types(df)

    # 8) Parquet yaz
    safe_name = cloning.safe_repo_name(project["clone_url"])
    out_path = projects_dir / f"{safe_name}.parquet"
    _atomic_write_parquet(df, out_path)

    # Ozet
    total_loc    = int(df["loc"].sum()) if "loc" in df.columns else 0
    bugs_keyword = int(df["bug_keyword"].sum()) if "bug_keyword" in df.columns else 0
    bugs_szz     = (
        int(df["bug_szz"].fillna(0).sum())
        if "bug_szz" in df.columns and not skip_szz
        else 0
    )
    smells_total = (
        int(df["smell_count"].fillna(0).sum())
        if "smell_count" in df.columns and not _skip_smells
        else 0
    )
    duration = round(time.monotonic() - started, 1)

    logger.info(
        "  [%s] ozet: files=%d loc=%d bug_kw=%d bug_szz=%d smell=%d (miss=%d) %.1fs",
        name, len(df), total_loc, bugs_keyword, bugs_szz,
        smells_total, smells_missing, duration,
    )

    return {
        "status":         "ok",
        "clone":          clone_status,
        "files":          len(df),
        "total_loc":      total_loc,
        "bugs_keyword":   bugs_keyword,
        "bugs_szz":       bugs_szz,
        "szz_fallback":   szz_fallback,
        "smells_total":   smells_total,
        "smells_missing": smells_missing,
        "parquet":        str(out_path),
        "duration_secs":  duration,
        "timing":         timings,
        "completed_at":   datetime.now(timezone.utc).isoformat(),
    }


# Export: T2 feature seti dogrulamasi (parquet sutunlarinin modele uyumu)
EXPECTED_FEATURE_COLUMNS = FEATURES_BUG
