"""
prospector_runner.py — Prospector subprocess + JSON parse.

F1 iskeleti; tam implementasyon F2'de. API contract (PLAN §13.5):

    run_prospector(file_path, strictness, timeout_seconds) -> dict
    run_prospector_batch(file_paths, workers, strictness) -> dict

F2'de subprocess + multiprocessing.Pool eklenir.
"""
from __future__ import annotations

import logging
from pathlib import Path

from pipeline.config import (
    PROSPECTOR_STRICTNESS,
    PROSPECTOR_TIMEOUT_SECONDS,
    PROSPECTOR_WORKERS,
)

logger = logging.getLogger(__name__)


def run_prospector(
    file_path: Path,
    strictness: str = PROSPECTOR_STRICTNESS,
    timeout_seconds: int = PROSPECTOR_TIMEOUT_SECONDS,
) -> dict:
    """
    Tek dosya uzerinde prospector calistir. F2'de implement edilecek.

    Returns:
        {'smell_count': int|None, 'categories': dict[str,int], 'messages': list}
    """
    raise NotImplementedError(
        "prospector_runner.run_prospector F2'de implement edilecek."
    )


def run_prospector_batch(
    file_paths: list[Path],
    workers: int = PROSPECTOR_WORKERS,
    strictness: str = PROSPECTOR_STRICTNESS,
) -> dict[Path, dict]:
    """multiprocessing.Pool ile paralel calistirma. F2'de."""
    raise NotImplementedError(
        "prospector_runner.run_prospector_batch F2'de implement edilecek."
    )
