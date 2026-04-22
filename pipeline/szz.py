"""
szz.py — SZZ (Sliwerski-Zimmermann-Zeller) bug-introducing commit tespiti.

F1 iskeleti; tam implementasyon F2'de. API contract (PLAN §13.4):

    compute_szz_labels(repo_path, head_files, bug_fix_commits,
                       timeout_seconds=600) -> dict[str, int]

Iskelet not: pydriller F2'de eklenir. Simdilik NotImplementedError ile
caller'a net hata verir; F2 oncesi yanlislikla cagrilirsa fail-fast.
"""
from __future__ import annotations

import logging
from pathlib import Path

from pipeline.config import SZZ_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


def compute_szz_labels(
    repo_path: Path,
    head_files: list[str],
    bug_fix_commits: list[str],
    timeout_seconds: int = SZZ_TIMEOUT_SECONDS,
) -> dict[str, int]:
    """
    Bug-fix commit'lerinden geriye dogru git blame ile bug-introducing
    commit'leri tespit et, bunlarin etkiledigi HEAD dosyalarini 1 olarak
    etiketle. F2'de pydriller ile implement edilecek.

    Returns:
        {file_path: 0|1}. Timeout veya hata durumunda {} — caller
        `bug_keyword` etiketine fallback yapar.
    """
    raise NotImplementedError(
        "szz.compute_szz_labels F2'de implement edilecek (pydriller)."
    )
