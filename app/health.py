"""
health.py — PLAN §17.1 Project Health + Smell Overview hesaplamalari.

Flask analizi tamamlandiktan sonra UI'deki ozet kartlari icin
toplam istatistikleri uretir. Sadece saf hesaplama — I/O yok.

Iki genel fonksiyon:
  - compute_project_health(commit_summary, rows) -> dict
  - compute_smell_summary(file_results, rows, prospector_results=None) -> dict
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def compute_project_health(
    commit_summary: dict[str, int],
    rows: list[dict],
) -> dict[str, Any]:
    """
    Proje seviyesinde "saglik" gostergeleri.

    Args:
        commit_summary: `pipeline.git_metrics.get_repo_commit_summary` cikisi.
        rows:           `_compute_per_file_rows` cikisi (per-file metrikler).

    Returns dict anahtarlari (UI bu aynen okur):
      - defect_density_per_kloc:  bug_count toplami / (toplam loc / 1000)
      - refactor_ratio:           refactor_commits / total_commits (0..1)
      - recent_commits_90d:       raw sayac
      - total_commits:            raw sayac
      - bug_fix_commits:          raw sayac
      - avg_commits_per_file:     toplam commit / dosya sayisi
      - bug_fix_ratio:            bug_fix_commits / total_commits (0..1)
    """
    total_commits   = int(commit_summary.get("total_commits", 0) or 0)
    bug_fix_commits = int(commit_summary.get("bug_fix_commits", 0) or 0)
    refactor_commits = int(commit_summary.get("refactor_commits", 0) or 0)
    recent_commits  = int(commit_summary.get("recent_commits_90d", 0) or 0)

    total_loc    = sum(int(r.get("loc", 0) or 0) for r in rows)
    total_bugs   = sum(int(r.get("bug_count", 0) or 0) for r in rows)
    total_commit = sum(int(r.get("commit_count", 0) or 0) for r in rows)
    n_files      = len(rows)

    if total_loc > 0:
        defect_density = total_bugs / (total_loc / 1000.0)
    else:
        defect_density = 0.0

    refactor_ratio = (refactor_commits / total_commits) if total_commits else 0.0
    bug_fix_ratio  = (bug_fix_commits / total_commits) if total_commits else 0.0
    avg_commits    = (total_commit / n_files) if n_files else 0.0

    return {
        "defect_density_per_kloc": round(defect_density, 2),
        "refactor_ratio":          round(refactor_ratio, 4),
        "bug_fix_ratio":           round(bug_fix_ratio, 4),
        "recent_commits_90d":      recent_commits,
        "total_commits":           total_commits,
        "bug_fix_commits":         bug_fix_commits,
        "refactor_commits":        refactor_commits,
        "avg_commits_per_file":    round(avg_commits, 2),
        "total_loc":               total_loc,
        "total_bug_commits_file":  total_bugs,
    }


def compute_smell_summary(
    file_results: list[dict],
    rows: list[dict],
    prospector_results: Optional[dict[str, dict]] = None,
) -> dict[str, Any]:
    """
    PLAN §17.1 Smell Overview kartlari icin ozet.

    Args:
        file_results:       `_assemble_file_results` cikisi; smell_pred/bug_pred icerebilir.
        rows:               Ham per-file metrikler (loc okumak icin).
        prospector_results: {file_path: {smell_count, categories, messages}}
                            — yoksa sadece ML tahmin sayaclari dondurulur.

    Returns:
      - prospector_enabled:     bool — prospector cikisi mevcut mu
      - total_smells:           sum(smell_count) veya 0
      - smell_density_per_kloc: total_smells / (total_loc/1000)
      - ml_smell_risk_count:    smell_pred==1 dosya sayisi
      - refactor_priority_count: bug_pred==1 AND smell_pred==1
      - top_smelly_files:       [{file_path, smell_count}, ...] top 5
    """
    prospector_enabled = bool(prospector_results)

    total_loc = sum(int(r.get("loc", 0) or 0) for r in rows)

    total_smells = 0
    by_file: list[tuple[str, int]] = []
    if prospector_enabled:
        for fpath, pres in prospector_results.items():
            if not isinstance(pres, dict):
                continue
            cnt = pres.get("smell_count")
            if cnt is None:
                continue
            try:
                cnt_i = int(cnt)
            except (TypeError, ValueError):
                continue
            total_smells += cnt_i
            by_file.append((str(fpath), cnt_i))

    smell_density = (total_smells / (total_loc / 1000.0)) if total_loc > 0 else 0.0

    ml_smell_count = sum(1 for f in file_results if f.get("smell_pred") == 1)
    refactor_priority = sum(
        1 for f in file_results
        if f.get("smell_pred") == 1 and f.get("bug_pred") == 1
    )

    by_file.sort(key=lambda t: t[1], reverse=True)
    top_smelly = [
        {"file_path": fp, "smell_count": cnt}
        for fp, cnt in by_file[:5]
        if cnt > 0
    ]

    return {
        "prospector_enabled":      prospector_enabled,
        "total_smells":            total_smells,
        "smell_density_per_kloc":  round(smell_density, 2),
        "ml_smell_risk_count":     ml_smell_count,
        "refactor_priority_count": refactor_priority,
        "top_smelly_files":        top_smelly,
    }
