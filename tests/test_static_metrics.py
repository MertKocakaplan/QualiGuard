"""
test_static_metrics.py — pipeline.static_metrics birim testleri.

calculate_metrics + cognitive complexity entegrasyonu denetler.
"""
from __future__ import annotations

import pytest

from pipeline.static_metrics import calculate_metrics, calculate_derived


_SIMPLE_SRC = """\
def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def div(a, b):
    if b == 0:
        raise ValueError("zero")
    return a / b


def clamp(v, lo, hi):
    return max(lo, min(hi, v))
"""

_BRANCHY_SRC = """\
def classify(x):
    if x > 100:
        if x > 200:
            return "very high"
        return "high"
    elif x > 50:
        return "medium"
    elif x > 0:
        return "low"
    else:
        return "negative"
"""


def test_calculate_metrics_returns_dict_for_valid_source():
    m = calculate_metrics(_SIMPLE_SRC)
    assert isinstance(m, dict)
    assert "loc" in m
    assert "maintainability_index" in m


def test_cognitive_complexity_keys_present():
    m = calculate_metrics(_SIMPLE_SRC)
    assert m is not None
    assert "cognitive_complexity_total" in m
    assert "cognitive_complexity_max"   in m


def test_cognitive_complexity_zero_for_no_functions():
    src = "x = 1\ny = 2\n"
    m = calculate_metrics(src)
    # SLOC < MIN_SLOC_THRESHOLD ise None donebilir; None ise 0 varsayalim
    if m is None:
        return
    assert m["cognitive_complexity_total"] == 0
    assert m["cognitive_complexity_max"]   == 0


def test_cognitive_complexity_increases_with_nesting():
    flat = "def f(x):\n    return x + 1\n"
    m_flat = calculate_metrics(flat)
    m_branch = calculate_metrics(_BRANCHY_SRC)
    if m_flat is None or m_branch is None:
        pytest.skip("kaynak cok kucuk veya parse edilemiyor")
    assert m_branch["cognitive_complexity_total"] > m_flat["cognitive_complexity_total"]


def test_cognitive_complexity_max_lte_total():
    m = calculate_metrics(_BRANCHY_SRC)
    assert m is not None
    assert m["cognitive_complexity_max"] <= m["cognitive_complexity_total"]


def test_calculate_derived_fills_four_fields():
    m = calculate_metrics(_SIMPLE_SRC)
    assert m is not None
    calculate_derived(m)
    for key in ("complexity_density", "comment_per_function",
                "avg_function_length", "effort_per_line"):
        assert key in m


def test_calculate_metrics_returns_none_for_syntax_error():
    m = calculate_metrics("def f(\n  # broken")
    assert m is None
