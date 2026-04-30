"""
analyzer.py — Flask ana analiz akisi.

V2'de agir kod `pipeline.*` modulune tasindi. Bu dosya sadece akisi
orkestre eder: GitHub info -> clone -> ls-tree -> git log -> radon ->
(opsiyonel) prospector -> ML tahmin -> health/smell ozet.

Prospector aktif ise (PLAN §5.2 opt-in), `pipeline.prospector_runner`
paralel olarak calisir, `smell_count` UI'de gosterilir.
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
    get_repo_commit_summary,
    should_skip_file,
)
from pipeline.static_metrics import calculate_derived, calculate_metrics

from . import predictor
from .health import compute_project_health, compute_smell_summary, risk_tier

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
        commit_summary = get_repo_commit_summary(repo_path)

        rows = _compute_per_file_rows(
            repo_path, py_files, project_info, git_stats, progress_callback,
        )
        if not rows:
            return _error_result(
                github_url,
                "Hicbir dosyadan gecerli metrik hesaplanamadi. Dosyalar cok kucuk veya parse edilemez olabilir."
            )

        prospector_results: dict[str, dict] = {}
        if prospector_enabled:
            progress_callback(70, f"Prospector calisiyor ({len(rows)} dosya)...")
            prospector_results = _run_prospector_safe(repo_path, rows)

        progress_callback(82, "Tahminler yapiliyor...")
        feature_names = predictor.get_feature_names()
        df = pd.DataFrame(rows)

        df_commit = _prepare_feature_frame(df, feature_names["commit"])
        commit_preds, commit_probs = predictor.predict_commit(df_commit)

        df_bug = _prepare_feature_frame(df, feature_names["bug"])
        bug_preds, bug_probs = predictor.predict_bug(df_bug)

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
            prospector_results,
        )
        file_results.sort(key=lambda x: x["bug_prob"], reverse=True)

        project_health = compute_project_health(commit_summary, rows)
        smell_summary  = compute_smell_summary(file_results, rows, prospector_results)

        progress_callback(100, "Tamamlandi!")
        # F5 — risk tier distribution for summary card
        tier_counts = {"PASS": 0, "REVIEW": 0, "BLOCK": 0}
        for f in file_results:
            t = f.get("risk_tier", "PASS")
            tier_counts[t] = tier_counts.get(t, 0) + 1

        return {
            "project_info":    project_info,
            "github_url":      github_url,
            "repo_name":       safe_repo_name(github_url),
            "files":           file_results,
            "project_health":  project_health,
            "smell_summary":   smell_summary,
            "prospector_enabled": prospector_enabled,
            "summary": {
                "total_files":      len(file_results),
                "high_commit_risk": sum(1 for f in file_results if f["commit_pred"] == 1),
                "has_bug_risk":     sum(1 for f in file_results if f["bug_pred"] == 1),
                "has_smell_risk":   sum(1 for f in file_results if f.get("smell_pred") == 1),
                # F5 — 3-tier quality gate counts
                "risk_pass":   tier_counts["PASS"],
                "risk_review": tier_counts["REVIEW"],
                "risk_block":  tier_counts["BLOCK"],
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


def _run_prospector_safe(
    repo_path: Path,
    rows: list[dict],
) -> dict[str, dict]:
    """
    Prospector'i rows icindeki dosya yollarina uygula.

    Prospector kurulu degilse veya patlarsa bos sozluk dondurur;
    analiz akisi kesilmez.
    """
    try:
        from pipeline.prospector_runner import run_prospector_batch
    except ImportError as exc:
        logger.warning("prospector_runner import edilemedi: %s", exc)
        return {}

    abs_paths = [repo_path / r["file_path"] for r in rows]
    try:
        raw = run_prospector_batch(abs_paths)
    except Exception as exc:  # noqa: BLE001 — batch cokerse akis devam etsin
        logger.warning("run_prospector_batch cokti: %s", exc)
        return {}

    # repo-relative key'e donustur
    results: dict[str, dict] = {}
    for row, abs_path in zip(rows, abs_paths):
        pres = raw.get(abs_path) or raw.get(str(abs_path))
        if pres:
            results[row["file_path"]] = pres
    return results


def _assemble_file_results(
    rows: list[dict],
    commit_preds, commit_probs,
    bug_preds, bug_probs,
    smell_preds, smell_probs,
    prospector_results: Optional[dict[str, dict]] = None,
) -> list[dict]:
    prospector_results = prospector_results or {}
    results = []
    for i, row in enumerate(rows):
        fpath = row["file_path"]
        # F5 — risk_score: bug_prob kalibresini ana sinyal olarak kullan
        r_score = round(float(bug_probs[i]), 4)
        entry = {
            "file_path":             fpath,
            "commit_pred":           int(commit_preds[i]),
            "commit_prob":           round(float(commit_probs[i]), 4),
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

        pres = prospector_results.get(fpath)
        if pres:
            cnt = pres.get("smell_count")
            entry["prospector_count"] = int(cnt) if cnt is not None else None
            entry["prospector_categories"] = pres.get("categories") or {}
        results.append(entry)
    return results


def _read_file(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _error_result(github_url: str, msg: str) -> dict:
    return {
        "project_info":   {},
        "github_url":     github_url,
        "repo_name":      safe_repo_name(github_url),
        "files":          [],
        "project_health": {},
        "smell_summary":  {},
        "prospector_enabled": False,
        "summary":        {"total_files": 0, "high_commit_risk": 0, "has_bug_risk": 0, "has_smell_risk": 0},
        "error":          msg,
        "error_detail":   None,
    }
