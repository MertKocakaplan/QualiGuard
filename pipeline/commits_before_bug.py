"""
commits_before_bug.py — Bug-oncesi commit istatistikleri.

F1 iskeleti; F2'de SZZ ciktisi hazir olunca tam implement edilir.
API contract (PLAN §13.6):

    compute_stats(commits_df) -> dict
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_stats(commits_df: pd.DataFrame) -> dict:
    """
    Her dosyanin commit serisinden bug-oncesi istatistikleri uret.

    Args:
        commits_df: file_path, commit_idx, is_bug_intro sutunlarini iceren df

    Returns:
        {
            'mean_commits_to_first_bug':    float,
            'median_commits_to_first_bug':  float,
            'mean_commits_between_bugs':    float,
            'by_file': {file_path: int},
        }
    """
    raise NotImplementedError(
        "commits_before_bug.compute_stats F2'de SZZ ciktisi ile implement edilecek."
    )
