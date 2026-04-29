"""
static_metrics.py — Radon + cognitive complexity tabanli statik metrik hesabi.

V1'deki app/metrics.py'den taşınmistir (PLAN §3.4). Imzalar korunur.

Tek dosya icin 22 raw/derivable metrik + 4 derived + 2 cognitive = 28 ozellik uretir.
"""
from __future__ import annotations

import ast
import logging
from typing import Optional

from cognitive_complexity.api import get_cognitive_complexity
from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit
from radon.raw import analyze

from pipeline.config import MAX_FILE_SIZE_BYTES, MIN_SLOC_THRESHOLD

logger = logging.getLogger(__name__)

_HALSTEAD_KEYS = (
    "h_vocabulary", "h_length", "h_volume", "h_difficulty",
    "h_effort", "h_bugs", "h_time", "h_calculated_length",
)


def calculate_metrics(source_code: str) -> Optional[dict]:
    """
    Radon ile statik metrikleri hesapla.

    Args:
        source_code: Dosyanin ham icerigi (UTF-8 decoded)

    Returns:
        Metrik sozlugu (22 raw + halstead + MI + 2 ratio) veya None.
        Cok buyuk dosya, cok kucuk dosya (SLOC < MIN_SLOC_THRESHOLD)
        veya parse hatasi durumunda None.
    """
    if len(source_code) > MAX_FILE_SIZE_BYTES:
        return None

    metrics: dict = {}

    # ── Raw metrikler ──────────────────────────────────────────
    try:
        raw = analyze(source_code)
        metrics["loc"]             = raw.loc
        metrics["lloc"]            = raw.lloc
        metrics["sloc"]            = raw.sloc
        metrics["comments"]        = raw.comments
        metrics["multi"]           = raw.multi
        metrics["blank"]           = raw.blank
        metrics["single_comments"] = raw.single_comments
    except (SyntaxError, TypeError, ValueError):
        return None

    if metrics.get("sloc", 0) < MIN_SLOC_THRESHOLD:
        return None

    # ── Cyclomatic complexity ────────────────────────────────
    try:
        cc_results = cc_visit(source_code)
        if cc_results:
            complexities = [block.complexity for block in cc_results]
            metrics["cc_mean"]      = sum(complexities) / len(complexities)
            metrics["cc_max"]       = max(complexities)
            metrics["cc_total"]     = sum(complexities)
            metrics["num_functions"] = len(cc_results)
        else:
            metrics["cc_mean"] = 0
            metrics["cc_max"]  = 0
            metrics["cc_total"] = 0
            metrics["num_functions"] = 0
    except (SyntaxError, TypeError, ValueError):
        metrics["cc_mean"] = 0
        metrics["cc_max"]  = 0
        metrics["cc_total"] = 0
        metrics["num_functions"] = 0

    # ── Halstead metrikleri ─────────────────────────────────
    try:
        halstead = h_visit(source_code)
        if halstead.total and hasattr(halstead.total, "volume"):
            h = halstead.total
            metrics["h_vocabulary"]       = h.vocabulary
            metrics["h_length"]           = h.length
            metrics["h_volume"]           = h.volume
            metrics["h_difficulty"]       = h.difficulty
            metrics["h_effort"]           = h.effort
            metrics["h_bugs"]             = h.bugs
            metrics["h_time"]             = h.time
            metrics["h_calculated_length"] = h.calculated_length
        else:
            for k in _HALSTEAD_KEYS:
                metrics[k] = 0
    except (SyntaxError, TypeError, ValueError):
        for k in _HALSTEAD_KEYS:
            metrics[k] = 0

    # ── Maintainability Index ────────────────────────────────
    try:
        metrics["maintainability_index"] = mi_visit(source_code, multi=True)
    except (SyntaxError, TypeError, ValueError):
        metrics["maintainability_index"] = 0

    # ── Cognitive Complexity (Campbell 2018) ─────────────────
    try:
        tree = ast.parse(source_code)
        funcs = [
            n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if funcs:
            cog_values = [get_cognitive_complexity(f) for f in funcs]
            metrics["cognitive_complexity_total"] = sum(cog_values)
            metrics["cognitive_complexity_max"]   = max(cog_values)
        else:
            metrics["cognitive_complexity_total"] = 0
            metrics["cognitive_complexity_max"]   = 0
    except (SyntaxError, TypeError, ValueError, RecursionError):
        metrics["cognitive_complexity_total"] = 0
        metrics["cognitive_complexity_max"]   = 0

    # ── Oran metrikleri ──────────────────────────────────────
    loc = max(metrics["loc"], 1)
    metrics["comment_ratio"] = metrics["comments"] / loc
    metrics["doc_ratio"]     = metrics["multi"] / loc

    return metrics


def calculate_derived(metrics: dict) -> dict:
    """
    4 turetilmis metrik ekler ve ayni sozlugu dondurur:
      - complexity_density   = cc_total / loc
      - comment_per_function = comments / num_functions
      - avg_function_length  = sloc / num_functions
      - effort_per_line      = h_effort / loc

    Sifira bolmeler 1'e normalize edilir.
    """
    loc    = max(metrics.get("loc", 1), 1)
    num_fn = max(metrics.get("num_functions", 0), 1)
    sloc   = metrics.get("sloc", 0) or 0

    metrics["complexity_density"]  = (metrics.get("cc_total") or 0) / loc
    metrics["comment_per_function"] = (metrics.get("comments") or 0) / num_fn
    metrics["avg_function_length"]  = sloc / num_fn
    metrics["effort_per_line"]      = (metrics.get("h_effort") or 0) / loc
    return metrics
