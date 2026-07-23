"""
analyzer.py — Flask ana analiz akisi.

Iki giris noktasi (F7):
  - `analyze_repo(github_url, progress_callback)`
      GitHub URL'den clone + analiz.
  - `analyze_zip(zip_path, progress_callback)`
      Yuklenmis ZIP'ten extract + analiz (.git/ zorunlu).

Iki yol da `_analyze_local_repo(repo_path, project_info, ...)`'a delege eder.
V2'de agir kod `pipeline.*` modullerinde — bu dosya yalniz orkestrasyon.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from pipeline.ci_cd import detect_ci_cd_signals
from pipeline.cloning import clone_repo, safe_repo_name
from pipeline.code_smells import detect_smells_batch
from pipeline.discovery import get_project_info
from pipeline.git_metrics import (
    get_bulk_git_stats,
    get_head_python_files,
    get_repo_commit_summary,
    should_skip_file,
)
from pipeline.static_metrics import calculate_derived, calculate_metrics
from pipeline.zip_handler import (
    ZipValidationError,
    extract_local_meta,
    safe_extract,
)

from . import predictor
from .health import compute_project_health, compute_smell_summary, risk_tier

logger = logging.getLogger(__name__)


# ── Giris noktasi 1: GitHub URL ───────────────────────────────────

def analyze_repo(
    github_url: str,
    progress_callback: Callable,
) -> dict:
    """
    GitHub repo'yu ucundan ucuna analiz et.

    progress_callback(percent: int, message: str) seklinde cagirilir.

    Args:
        github_url:        "https://github.com/user/repo" formatinda
        progress_callback: UI'ye ilerleme bildirimi

    Returns:
        PLAN §5 icinde tanimli sonuc sozlugu.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="metrihunter_"))
    try:
        progress_callback(5, "Fetching GitHub project information...")
        try:
            project_info = get_project_info(github_url)
        except RuntimeError as exc:
            return _error_result(github_url, str(exc))

        progress_callback(10, "Cloning repository...")
        repo_path, clone_status = clone_repo(github_url, tmp_dir)
        if repo_path is None:
            return _error_result(github_url, clone_status)

        return _analyze_local_repo(
            repo_path, project_info, github_url, progress_callback, start_pct=20,
        )

    except Exception as exc:  # noqa: BLE001 — en dis kapan, traceback raporlariz
        tb = traceback.format_exc()
        logger.exception("analyze_repo cokti: %s", exc)
        return {
            **_error_result(github_url, str(exc) or "Unknown error."),
            "error_detail": tb,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Giris noktasi 2: ZIP upload (F7) ──────────────────────────────

def analyze_zip(
    zip_path: Path,
    progress_callback: Callable,
) -> dict:
    """
    Yuklenmis ZIP dosyasini extract et + ucundan ucuna analiz et.

    Akis:
      1. Validate: dosya sayisi, decompressed size, compression ratio,
         path traversal (pipeline.zip_handler.safe_extract).
      2. Repo koku tespit: top-level alt-dir'i takip et, `.git/` ZORUNLU.
      3. extract_local_meta: git log'dan minimum project_info (stars=0,
         contributor_count + project_age_days hesaplanir).
      4. _analyze_local_repo: ortak akis.

    Args:
        zip_path:          Yuklenmis ZIP dosyasi (server tarafinda kayitli).
        progress_callback: UI'ye ilerleme bildirimi.

    Returns:
        analyze_repo ile ayni semadaki sonuc sozlugu. `"source": "upload"`
        ek anahtariyla.
    """
    fallback_name = zip_path.stem.replace(" ", "_") or "uploaded_repo"
    tmp_dir = Path(tempfile.mkdtemp(prefix="metrihunter_zip_"))
    try:
        progress_callback(5, "Validating ZIP (size + zip-bomb protection)...")
        progress_callback(10, "Extracting ZIP and checking for .git/...")
        try:
            repo_path = safe_extract(zip_path, tmp_dir)
        except ZipValidationError as exc:
            return _error_result("", str(exc), repo_name=fallback_name)
        except Exception as exc:  # noqa: BLE001 — extract bozuk ZIP gibi
            logger.warning("ZIP extraction failed: %s", exc)
            return _error_result("", f"Could not open the ZIP: {exc}", repo_name=fallback_name)

        progress_callback(15, "Extracting metadata from local git history...")
        project_info = extract_local_meta(repo_path, fallback_name)

        return _analyze_local_repo(
            repo_path, project_info, "", progress_callback, start_pct=20,
            display_name=fallback_name,
        )

    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.exception("analyze_zip cokti: %s", exc)
        return {
            **_error_result("", str(exc) or "Unknown error.", repo_name=fallback_name),
            "error_detail": tb,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Ortak analiz akisi ────────────────────────────────────────────

def _analyze_local_repo(
    repo_path: Path,
    project_info: dict,
    github_url: str,
    progress_callback: Callable,
    start_pct: int = 20,
    display_name: Optional[str] = None,
) -> dict:
    """
    Local repo (clone ya da extract sonrasi) icin tam analiz.

    Iki giris noktasinin (URL/ZIP) ortak alt-akisi.

    Args:
        repo_path:        `.git/` iceren yerel kok dizini.
        project_info:     GitHub API'den ya da extract_local_meta'dan dict.
        github_url:       "" ise ZIP upload kaynak. Sonuc dict'inde gosterim.
        progress_callback: callable
        start_pct:        Bu helper'in baslangic progress yuzdesi.
        display_name:     UI'da gosterilecek repo adi (ZIP upload icin).
    """
    progress_callback(start_pct, "Listing Python files...")
    all_files = get_head_python_files(repo_path)
    py_files = [f for f in all_files if not should_skip_file(f)]

    if not py_files:
        return _error_result(
            github_url,
            "No Python files to analyse were found; all files may have been filtered out.",
            repo_name=display_name,
        )

    progress_callback(start_pct + 5,
                      f"{len(py_files)} files found. Computing Git statistics...")
    git_stats = get_bulk_git_stats(repo_path, py_files)
    commit_summary = get_repo_commit_summary(repo_path)

    # F7 — CI/CD signals (DevOps practices indicator)
    ci_cd_signals = detect_ci_cd_signals(repo_path)

    # F7 — AST tabanli smell breakdown (her dosya icin 7 smell turu sayisi)
    progress_callback(start_pct + 8, "Detecting code smell types (AST)...")
    abs_paths = [repo_path / f for f in py_files]
    smell_breakdown_raw = detect_smells_batch(abs_paths)
    smell_breakdown: dict[str, dict] = {}
    for rel, abs_p in zip(py_files, abs_paths):
        res = smell_breakdown_raw.get(abs_p) or smell_breakdown_raw.get(str(abs_p))
        if res:
            smell_breakdown[rel] = res

    rows = _compute_per_file_rows(
        repo_path, py_files, project_info, git_stats, progress_callback,
    )
    if not rows:
        return _error_result(
            github_url,
            "No valid metrics could be computed for any file. "
            "The files may be too small or unparsable.",
            repo_name=display_name,
        )

    progress_callback(82, "Running predictions...")
    feature_names = predictor.get_feature_names()
    df = pd.DataFrame(rows)

    df_bug = _prepare_feature_frame(df, feature_names["bug"])
    bug_preds, bug_probs = predictor.predict_bug(df_bug)

    smell_preds = None
    smell_probs = None
    if predictor.smell_available() and feature_names.get("smell"):
        df_smell = _prepare_feature_frame(df, feature_names["smell"])
        smell_preds, smell_probs = predictor.predict_smell(df_smell)

    progress_callback(95, "Preparing results...")
    file_results = _assemble_file_results(
        rows,
        bug_preds, bug_probs,
        smell_preds, smell_probs,
        smell_breakdown=smell_breakdown,
    )
    file_results.sort(key=lambda x: x["bug_prob"], reverse=True)

    project_health = compute_project_health(commit_summary, rows)
    smell_summary  = compute_smell_summary(file_results, rows)

    progress_callback(100, "Completed!")
    # F5 — risk tier distribution for summary card
    tier_counts = {"PASS": 0, "REVIEW": 0, "BLOCK": 0}
    for f in file_results:
        t = f.get("risk_tier", "PASS")
        tier_counts[t] = tier_counts.get(t, 0) + 1

    return {
        "project_info":    project_info,
        "github_url":      github_url,
        "repo_name":       display_name or (safe_repo_name(github_url) if github_url else "local"),
        "files":           file_results,
        "project_health":  project_health,
        "smell_summary":   smell_summary,
        # F7 — CI/CD / DevOps practices karti
        "ci_cd":           ci_cd_signals,
        # F7 — Veri kaynagi: "github" veya "upload" (UI rozet icin)
        "source":          "github" if github_url else "upload",
        "summary": {
            "total_files":      len(file_results),
            "has_bug_risk":     sum(1 for f in file_results if f["bug_pred"] == 1),
            "has_smell_risk":   sum(1 for f in file_results if f.get("smell_pred") == 1),
            "risk_pass":   tier_counts["PASS"],
            "risk_review": tier_counts["REVIEW"],
            "risk_block":  tier_counts["BLOCK"],
        },
        "error":        None,
        "error_detail": None,
    }


# ── Yardimcilar ──────────────────────────────────────────────────

def _compute_per_file_rows(
    repo_path: Path,
    py_files: list[str],
    project_info: dict,
    git_stats: dict,
    progress_callback: Callable,
) -> list[dict]:
    rows: list[dict] = []
    n = len(py_files)
    update_every = max(1, n // 20)

    for i, fpath in enumerate(py_files):
        if i % update_every == 0:
            pct = 30 + int((i / n) * 50)
            progress_callback(pct, f"Computing metrics ({i + 1}/{n})...")

        source = _read_file(repo_path / fpath)
        if source is None:
            continue

        m = calculate_metrics(source)
        if m is None:
            continue
        m = calculate_derived(m)

        gst = git_stats.get(fpath, {})
        rows.append({
            "file_path": fpath,
            **project_info,
            "commit_count":         gst.get("commit_count", 1),
            "bug_count":            gst.get("bug_count", 0),
            "n_authors":            gst.get("n_authors", 1),
            "file_age_days":        gst.get("file_age_days", 30.0),
            "churn_total":          gst.get("churn_total", 0),
            "avg_churn_per_commit": gst.get("avg_churn_per_commit", 0.0),
            "max_single_churn":     gst.get("max_single_churn", 0),
            "recent_commits_90d":   gst.get("recent_commits_90d", 0),
            **m,
        })
    return rows


def _prepare_feature_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Eksik sutunlari 0 ile doldurup sirali frame don."""
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0
    return df[cols].fillna(0.0)


def _assemble_file_results(
    rows: list[dict],
    bug_preds, bug_probs,
    smell_preds, smell_probs,
    smell_breakdown: Optional[dict[str, dict]] = None,
) -> list[dict]:
    smell_breakdown = smell_breakdown or {}
    results = []
    for i, row in enumerate(rows):
        fpath = row["file_path"]
        # F5 — risk_score: bug_prob kalibresini ana sinyal olarak kullan
        r_score = round(float(bug_probs[i]), 4)
        entry = {
            "file_path":             fpath,
            "bug_pred":              int(bug_preds[i]),
            "bug_prob":              round(float(bug_probs[i]), 4),
            "loc":                   int(row.get("loc", 0)),
            "cc_mean":               round(float(row.get("cc_mean") or 0), 2),
            "maintainability_index": round(float(row.get("maintainability_index") or 0), 1),
            # F5 — risk score + 3-tier quality gate
            "risk_score": r_score,
            "risk_tier":  risk_tier(r_score),
        }
        if smell_preds is not None and smell_probs is not None:
            entry["smell_pred"] = int(smell_preds[i])
            entry["smell_prob"] = round(float(smell_probs[i]), 4)

        # F7 — AST smell breakdown (7 tur: long_method, large_class, ...)
        sb = smell_breakdown.get(fpath)
        if sb:
            entry["smell_breakdown"] = {
                "total":               int(sb.get("smell_count", 0) or 0),
                "long_method":         int(sb.get("smell_long_method", 0) or 0),
                "large_class":         int(sb.get("smell_large_class", 0) or 0),
                "long_param_list":     int(sb.get("smell_long_param_list", 0) or 0),
                "deep_nesting":        int(sb.get("smell_deep_nesting", 0) or 0),
                "high_complexity":     int(sb.get("smell_high_complexity", 0) or 0),
                "low_maintainability": int(sb.get("smell_low_maintainability", 0) or 0),
                "god_function":        int(sb.get("smell_god_function", 0) or 0),
            }

        results.append(entry)
    return results


def _read_file(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _error_result(
    github_url: str,
    msg: str,
    repo_name: Optional[str] = None,
) -> dict:
    """Hata sonucunda UI'a donen minimum sema."""
    name = repo_name or (safe_repo_name(github_url) if github_url else "uploaded")
    return {
        "project_info":   {},
        "github_url":     github_url,
        "repo_name":      name,
        "files":          [],
        "project_health": {},
        "smell_summary":  {},
        "ci_cd":          {},
        "source":         "github" if github_url else "upload",
        "summary":        {
            "total_files": 0,
            "has_bug_risk": 0, "has_smell_risk": 0,
        },
        "error":          msg,
        "error_detail":   None,
    }
