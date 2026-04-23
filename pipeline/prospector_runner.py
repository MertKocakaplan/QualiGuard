"""
prospector_runner.py — Prospector subprocess + JSON parse + paralel.

PLAN §3.7 + §13.5'e gore her Python dosyasi icin prospector'i calistirir,
cikan JSON'u parse eder. Tekli ve batch (multiprocessing.Pool) API sunar.

Cikti sozlugu:

    {
        'smell_count': int | None,          # None = hata/timeout/parse failure
        'categories': dict[str, int],       # source (tool) -> mesaj sayisi
        'messages':   list[dict],           # Phase C icin ham mesaj listesi
    }

Exit code notu:
    prospector mesaj buldugunda 1, bulmadiginda 0 doner. Bu ayri bir hata
    gostergesi degildir — stdout gecerli JSON ise ikisi de success'tir.

Paralel is:
    Pool worker'larinin `run_prospector`'i cagirabilmesi icin `Path`
    parametresi pickle'lanabilir; modul-seviyesinde tanimli `run_prospector`
    fonksiyonu Pool icin uygun.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

from pipeline.config import (
    PROSPECTOR_STRICTNESS,
    PROSPECTOR_TIMEOUT_SECONDS,
    PROSPECTOR_WORKERS,
)

logger = logging.getLogger(__name__)


# ── Subprocess komutunun yapisi ──────────────────────────────────

def _prospector_cmd(file_path: Path, strictness: str) -> list[str]:
    """
    Platformlar arasi guvenli prospector cagrisi.

    `prospector` bazen PATH'te exe olarak, bazen sadece Python modulu
    olarak kuruludur. Once exe'yi ara, yoksa `python -m prospector`.
    """
    exe = shutil.which("prospector")
    if exe:
        return [
            exe,
            "--output-format=json",
            f"--strictness={strictness}",
            str(file_path),
        ]
    return [
        sys.executable,
        "-m",
        "prospector",
        "--output-format=json",
        f"--strictness={strictness}",
        str(file_path),
    ]


def _parse_output(stdout: str) -> Optional[dict]:
    """
    Prospector JSON'unu parse et.

    Prospector bazen lider satirlar (warning'ler) ardindan JSON basiyor.
    Ilk `{` karakterinden sonunu okuyarak toleranslica parse et.
    """
    if not stdout:
        return None
    idx = stdout.find("{")
    if idx < 0:
        return None
    try:
        return json.loads(stdout[idx:])
    except json.JSONDecodeError:
        return None


# ── Public API ───────────────────────────────────────────────────

def run_prospector(
    file_path: Path,
    strictness: str = PROSPECTOR_STRICTNESS,
    timeout_seconds: int = PROSPECTOR_TIMEOUT_SECONDS,
) -> dict:
    """
    Tek bir dosya uzerinde prospector calistir ve JSON ciktisini parse et.

    Args:
        file_path:        Analiz edilecek .py dosyasi.
        strictness:       Prospector sikilik seviyesi.
        timeout_seconds:  Subprocess zaman asim sinirini.

    Returns:
        {'smell_count', 'categories', 'messages'}.
        Hata/timeout durumunda smell_count=None.
    """
    cmd = _prospector_cmd(file_path, strictness)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        logger.warning("prospector timeout (%ds): %s", timeout_seconds, file_path)
        return {"smell_count": None, "categories": {}, "messages": []}
    except FileNotFoundError:
        logger.error("prospector kurulu degil (shutil.which + python -m ikisi de bos).")
        return {"smell_count": None, "categories": {}, "messages": []}
    except OSError as exc:
        logger.warning("prospector OSError (%s): %s", file_path, exc)
        return {"smell_count": None, "categories": {}, "messages": []}

    parsed = _parse_output(proc.stdout)
    if parsed is None:
        logger.debug(
            "prospector gecersiz JSON (%s, rc=%d): %s",
            file_path, proc.returncode, proc.stderr[:200],
        )
        return {"smell_count": None, "categories": {}, "messages": []}

    summary  = parsed.get("summary", {}) or {}
    messages = parsed.get("messages", []) or []

    count = summary.get("message_count")
    if not isinstance(count, int):
        count = len(messages)

    categories: Counter[str] = Counter()
    for m in messages:
        src = m.get("source") or "unknown"
        categories[src] += 1

    return {
        "smell_count": count,
        "categories": dict(categories),
        "messages":   messages,
    }


def _run_prospector_pickleable(args: tuple) -> tuple:
    """Pool worker sarmalayicisi — (path, result) iki'li doner."""
    file_path, strictness, timeout_seconds = args
    result = run_prospector(file_path, strictness, timeout_seconds)
    return file_path, result


def run_prospector_batch(
    file_paths: list[Path],
    workers: int = PROSPECTOR_WORKERS,
    strictness: str = PROSPECTOR_STRICTNESS,
    timeout_seconds: int = PROSPECTOR_TIMEOUT_SECONDS,
) -> dict[Path, dict]:
    """
    Cok sayida dosya icin paralel prospector.

    multiprocessing.Pool ile `workers` kadar es zamanli subprocess calistirir.
    Dosya sayisi workers'dan azsa tek sureclik paralelizm atlanir.

    Args:
        file_paths:       Analiz edilecek dosyalar.
        workers:          Paralel worker sayisi.
        strictness:       Prospector sikilik seviyesi.
        timeout_seconds:  Her dosyanin zaman asim sinirini.

    Returns:
        {Path: result_dict} — sirali degildir; her giris icin kayit.
    """
    if not file_paths:
        return {}

    workers = max(1, min(workers, len(file_paths)))

    args_iter = [
        (fp, strictness, timeout_seconds) for fp in file_paths
    ]

    if workers == 1:
        return dict(_run_prospector_pickleable(a) for a in args_iter)

    with Pool(processes=workers) as pool:
        results = pool.map(_run_prospector_pickleable, args_iter)

    return dict(results)
