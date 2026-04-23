"""
commits_before_bug.py — Bug oncesi commit istatistikleri.

PLAN §3.8 + §13.6.

Girdi: file_path, commit_idx, is_bug_intro sutunlu DataFrame. commit_idx,
dosyanin kendi commit serisindeki siradir (0'dan baslar). is_bug_intro,
SZZ ciktisindan 0/1'dir.

Cikti (sozluk):
    mean_commits_to_first_bug    — dosyalar arasinda ilk-bug-a-kadar commit
                                    sayilarinin ortalamasi (bug'i olan dosyalar).
    median_commits_to_first_bug  — yukaridakinin medyani.
    mean_commits_between_bugs    — ayni dosyada ardisik bug intro'lari arasi
                                    ortalama commit farki (coklu bug'i olan
                                    dosyalar).
    by_file                      — {file_path: first_bug_commit_idx}. Bug'i
                                    olmayan dosyalar burada YOK.

Bos ya da bug'siz veri icin sifir/NaN yerine gerekli default'lar verilir.
"""
from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLS: tuple[str, ...] = ("file_path", "commit_idx", "is_bug_intro")


def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"commits_before_bug: eksik sutunlar {missing}")


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else 0.0


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    mid = n // 2
    return float(xs_sorted[mid] if n % 2 == 1 else (xs_sorted[mid - 1] + xs_sorted[mid]) / 2)


def compute_stats(commits_df: pd.DataFrame) -> dict:
    """
    Commit serisinden bug-oncesi ozet istatistikleri uret.

    Args:
        commits_df: file_path / commit_idx / is_bug_intro sutunlari olan df.

    Returns:
        {
            'mean_commits_to_first_bug': float,
            'median_commits_to_first_bug': float,
            'mean_commits_between_bugs': float,
            'by_file': {file_path: first_bug_commit_idx},
        }
    """
    _validate(commits_df)

    empty_result = {
        "mean_commits_to_first_bug":   0.0,
        "median_commits_to_first_bug": 0.0,
        "mean_commits_between_bugs":   0.0,
        "by_file":                     {},
    }
    if commits_df.empty:
        return empty_result

    by_file: dict[str, int] = {}
    between_diffs: list[int] = []

    grouped = commits_df.sort_values(["file_path", "commit_idx"]).groupby("file_path")

    for file_path, group in grouped:
        bug_rows = group[group["is_bug_intro"] == 1]
        if bug_rows.empty:
            continue

        indices = bug_rows["commit_idx"].astype(int).tolist()
        by_file[str(file_path)] = int(indices[0])

        if len(indices) > 1:
            for prev, curr in zip(indices[:-1], indices[1:]):
                between_diffs.append(int(curr) - int(prev))

    firsts = list(by_file.values())
    return {
        "mean_commits_to_first_bug":   round(_mean(firsts), 2),
        "median_commits_to_first_bug": round(_median(firsts), 2),
        "mean_commits_between_bugs":   round(_mean(between_diffs), 2),
        "by_file":                     by_file,
    }
