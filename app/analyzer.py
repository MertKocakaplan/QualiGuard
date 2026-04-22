"""
analyzer.py — Flask ana analiz akisi.

V2'de agir kod `pipeline.*` modulune tasindi. Bu dosya sadece akisi
orkestre eder: GitHub info -> clone -> ls-tree -> git log -> radon ->
ML tahmin.

Prospector entegrasyonu F7'de eklenir (PLAN §5.2). F1'de sadece
flag hazir tutulur.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from pipeline.cloning import clone_repo, safe_repo_name
from pipeline.discovery import get_project_info
from pipeline.git_metrics import (
    get_bulk_git_stats,
    get_head_python_files,
    should_skip_file,
)
from pipeline.static_metrics import calculate_derived, calculate_metrics

from . import predictor

logger = logging.getLogger(__name__)


def analyze_repo(
    github_url: str,
    progress_callback: Callable,
    prospector_enabled: bool = False,
) -> dict:
    """
    GitHub repo'yu ucundan ucuna analiz et.

    progress_callback(percent: int, message: str) seklinde cagirilir.

    Args:
        github_url:        "https://github.com/user/repo" formatinda
        progress_callback: UI'ye ilerleme bildirimi
        prospector_enabled: F7'de aktiflesir — simdilik no-op

    Returns:
        PLAN §5 icinde tanimli sonuc sozlugu.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="metrihunter_"))
    try:
        progress_callback(5, "GitHub proje bilgileri aliniyor...")
        try:
            project_info = get_project_info(github_url)
        except RuntimeError as exc:
            return _error_result(github_url, str(exc))

        progress_callback(10, "Repo klonlaniyor...")
        repo_path, clone_status = clone_repo(github_url, tmp_dir)
        if repo_path is None:
            return _error_result(github_url, clone_status)

        progress_callback(20, "Python dosyalari listeleniyor...")
        all_files = get_head_python_files(repo_path)
        py_files = [f for f in all_files if not should_skip_file(f)]

        if not py_files:
            return _error_result(
                github_url,
                "Analiz edilecek Python dosyasi bulunamadi. Tum dosyalar filtrelenmis olabilir."
            )

        progress_callback(25, f"{len(py_files)} dosya bulundu. Git istatistikleri hesaplaniyor...")
        git_stats = get_bulk_git_stats(repo_path, py_files)

        rows = _compute_per_file_rows(
            repo_path, py_files, project_info, git_stats, progress_callback,
        )
        if not rows:
            return _error_result(
                github_url,
                "Hicbir dosyadan gecerli metrik hesaplanamadi. Dosyalar cok kucuk veya parse edilemez olabilir."
            )

        progress_callback(82, "Tahminler yapiliyor...")
        feature_names = predictor.get_feature_names()
        df = pd.DataFrame(rows)

        df_commit = _prepare_feature_frame(df, feature_names["commit"])
        commit_preds, commit_probs = predictor.predict_commit(df_commit)

        df_bug = _prepare_feature_frame(df, feature_names["bug"])
        bug_preds, bug_probs = predictor.predict_bug(df_bug)

        # F7'de: smell + prospector entegrasyonu eklenecek
        smell_preds = None
        smell_probs = None
        if predictor.smell_available() and feature_names.get("smell"):
            df_smell = _prepare_feature_frame(df, feature_names["smell"])
            smell_preds, smell_probs = predictor.predict_smell(df_smell)

        progress_callback(95, "Sonuclar hazirlaniyor...")
        file_results = _assemble_file_results(
            rows, commit_preds, commit_probs,
            bug_preds, bug_probs,
            smell_preds, smell_probs,
        )
        file_results.sort(key=lambda x: x["bug_prob"], reverse=True)

        progress_callback(100, "Tamamlandi!")
        return {
            "project_info": project_info,
            "github_url":   github_url,
            "repo_name":    safe_repo_name(github_url),
            "files":        file_results,
            "summary": {
                "total_files":      len(file_results),
                "high_commit_risk": sum(1 for f in file_results if f["commit_pred"] == 1),
                "has_bug_risk":     sum(1 for f in file_results if f["bug_pred"] == 1),
                "has_smell_risk":   sum(1 for f in file_results if f.get("smell_pred") == 1),
            },
            "error":        None,
            "error_detail": None,
        }

    except Exception as exc:  # noqa: BLE001 — en dis kapan, traceback raporlariz
        tb = traceback.format_exc()
        logger.exception("analyze_repo cokti: %s", exc)
        return {
            **_error_result(github_url, str(exc) or "Bilinmeyen hata."),
            "error_detail": tb,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


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
            progress_callback(pct, f"Metrikler hesaplaniyor ({i + 1}/{n})...")

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
    commit_preds, commit_probs,
    bug_preds, bug_probs,
    smell_preds, smell_probs,
) -> list[dict]:
    results = []
    for i, row in enumerate(rows):
        entry = {
            "file_path":             row["file_path"],
            "commit_pred":           int(commit_preds[i]),
            "commit_prob":           round(float(commit_probs[i]), 4),
            "bug_pred":              int(bug_preds[i]),
            "bug_prob":              round(float(bug_probs[i]), 4),
            "loc":                   int(row.get("loc", 0)),
            "cc_mean":               round(float(row.get("cc_mean") or 0), 2),
            "maintainability_index": round(float(row.get("maintainability_index") or 0), 1),
        }
        if smell_preds is not None and smell_probs is not None:
            entry["smell_pred"] = int(smell_preds[i])
            entry["smell_prob"] = round(float(smell_probs[i]), 4)
        results.append(entry)
    return results


def _read_file(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _error_result(github_url: str, msg: str) -> dict:
    return {
        "project_info": {},
        "github_url":   github_url,
        "repo_name":    safe_repo_name(github_url),
        "files":        [],
        "summary":      {"total_files": 0, "high_commit_risk": 0, "has_bug_risk": 0, "has_smell_risk": 0},
        "error":        msg,
        "error_detail": None,
    }
