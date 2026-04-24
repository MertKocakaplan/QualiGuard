"""
project_stats.py — global + kategori bazli istatistikler.

PLAN §14.6 kapsami. `scripts/train_final.py` tarafindan cagrilir,
`models/project_stats.json` uretir; Flask `predictor.get_project_stats()`
ile panellere verir.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_mean(series: pd.Series) -> float:
    """NaN atlayan ortalama; sonuc NaN ise 0.0 dondur."""
    if series is None or len(series) == 0:
        return 0.0
    v = pd.to_numeric(series, errors="coerce").mean()
    return 0.0 if pd.isna(v) else float(v)


def _safe_median(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return 0.0
    v = pd.to_numeric(series, errors="coerce").median()
    return 0.0 if pd.isna(v) else float(v)


def _safe_rate(series: pd.Series) -> float:
    """0/1 serisinde pozitif oran; NaN'lar atlanir."""
    if series is None or len(series) == 0:
        return 0.0
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0
    return float((s > 0).mean())


def compute_project_stats(df: pd.DataFrame) -> dict[str, Any]:
    """
    PLAN §14.6 sozlugu uret.

    Beklenen sutunlar:
      - project_name, loc, smell_count, bug_keyword, bug_szz,
        commits_to_first_bug, category_primary

    Eksik sutunlar icin varsayilan deger (0 / 0.0) kullanilir; cagri hata vermez.
    """
    out: dict[str, Any] = {"global": {}, "by_category": {}}

    if df is None or df.empty:
        logger.warning("compute_project_stats: bos DataFrame; varsayilan sozluk.")
        out["global"] = {
            "n_projects": 0, "n_files": 0, "total_loc": 0,
            "avg_commits_to_first_bug": 0.0,
            "median_smell_density": 0.0,
            "bug_rate_keyword": 0.0, "bug_rate_szz": 0.0,
        }
        return out

    total_loc = int(pd.to_numeric(df.get("loc", pd.Series(dtype=float)),
                                   errors="coerce").fillna(0).sum())
    n_projects = int(df["project_name"].nunique()) if "project_name" in df else 0

    # commits_to_first_bug: -1 = bug yok; bunlari haric tut
    ctfb = pd.to_numeric(df.get("commits_to_first_bug", pd.Series(dtype=float)),
                          errors="coerce")
    ctfb_valid = ctfb[ctfb >= 0] if ctfb is not None else pd.Series(dtype=float)

    smell_density = pd.Series(dtype=float)
    if {"smell_count", "loc"}.issubset(df.columns):
        loc_nonzero = pd.to_numeric(df["loc"], errors="coerce").replace(0, np.nan)
        smell_density = (pd.to_numeric(df["smell_count"], errors="coerce")
                           / loc_nonzero * 1000.0).dropna()

    out["global"] = {
        "n_projects":               n_projects,
        "n_files":                  int(len(df)),
        "total_loc":                total_loc,
        "avg_commits_to_first_bug": _safe_mean(ctfb_valid),
        "median_smell_density":     _safe_median(smell_density),
        "bug_rate_keyword":         _safe_rate(df.get("bug_keyword",
                                                       pd.Series(dtype=float))),
        "bug_rate_szz":             _safe_rate(df.get("bug_szz",
                                                       pd.Series(dtype=float))),
    }

    # by_category — category_primary varsa
    if "category_primary" in df.columns:
        for cat, group in df.groupby("category_primary"):
            cat_key = str(cat)
            ctfb_g = pd.to_numeric(group.get("commits_to_first_bug",
                                              pd.Series(dtype=float)),
                                    errors="coerce")
            ctfb_g_valid = ctfb_g[ctfb_g >= 0] if ctfb_g is not None else pd.Series(dtype=float)
            out["by_category"][cat_key] = {
                "n_projects":         int(group["project_name"].nunique())
                                        if "project_name" in group else 0,
                "n_files":            int(len(group)),
                "bug_rate_keyword":   _safe_rate(group.get("bug_keyword",
                                                            pd.Series(dtype=float))),
                "bug_rate_szz":       _safe_rate(group.get("bug_szz",
                                                            pd.Series(dtype=float))),
                "avg_commits_to_first_bug": _safe_mean(ctfb_g_valid),
            }

    return out


def write_project_stats(
    df: pd.DataFrame,
    out_path: Path,
) -> Path:
    """
    `compute_project_stats(df)` sonucunu `out_path`'a JSON olarak yaz.
    Dizin yoksa olusturulur.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = compute_project_stats(df)
    out_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return out_path
