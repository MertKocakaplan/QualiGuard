"""
test_code_smells.py — pipeline.code_smells birim testleri.

7 smell tipi, batch API ve hata yollarini denetler.
Esik degerleri config'den okunur; test kaynaklari
her test icinde tmp_path'e yazilir.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.code_smells import detect_smells, detect_smells_batch, detect_smells_from_source
from pipeline.config import (
    GOD_FUNC_CC,
    GOD_FUNC_LOC,
    HIGH_CC,
    LARGE_CLASS_LOC,
    LARGE_CLASS_METHOD_COUNT,
    LONG_METHOD_LOC,
    LONG_PARAM_COUNT,
    LOW_MI,
    NESTING_DEPTH,
)


# ── Kaynak uretici yardimcilar ────────────────────────────────────

def _write(tmp_path: Path, name: str, src: str) -> Path:
    p = tmp_path / name
    p.write_text(src, encoding="utf-8")
    return p


def _long_method_src(loc: int = LONG_METHOD_LOC + 10) -> str:
    lines = [f"def long_func():"]
    for i in range(loc - 1):
        lines.append(f"    x{i} = {i}")
    lines.append("    return x0")
    return "\n".join(lines)


def _large_class_src(
    loc: int = LARGE_CLASS_LOC + 10,
    methods: int = LARGE_CLASS_METHOD_COUNT + 2,
) -> str:
    # AST end_lineno yorumsuz satirlara kadar gittiginden
    # padding icin class-seviyesinde assignment kullan
    lines = ["class BigClass:"]
    for i in range(methods):
        lines.append(f"    def method_{i}(self): pass")
    pad = loc - len(lines)
    for j in range(max(pad, 0)):
        lines.append(f"    _attr_{j} = {j}")
    return "\n".join(lines)


def _long_param_src(params: int = LONG_PARAM_COUNT + 2) -> str:
    args = ", ".join(f"p{i}" for i in range(params))
    return f"def f({args}): pass\n"


def _deep_nesting_src(depth: int = NESTING_DEPTH + 1) -> str:
    lines = ["def f(x):"]
    indent = "    "
    for i in range(depth):
        lines.append(f"{indent * (i + 1)}if x > {i}:")
    lines.append(f"{indent * (depth + 1)}pass")
    return "\n".join(lines)


def _high_cc_src(branches: int = HIGH_CC + 5) -> str:
    lines = ["def complex_func(x):"]
    lines.append("    r = 0")
    for i in range(branches):
        lines.append(f"    if x == {i}:")
        lines.append(f"        r += {i}")
    lines.append("    return r")
    return "\n".join(lines)


def _god_function_src() -> str:
    # CC > GOD_FUNC_CC (15) ve LOC > GOD_FUNC_LOC (80)
    branches = GOD_FUNC_CC + 2   # 17 dallanma → CC=18
    base_lines = GOD_FUNC_LOC + 10  # 90 satir
    lines = ["def god_func(x, y):"]
    lines.append("    r = 0")
    for i in range(branches):
        lines.append(f"    if x == {i}:")
        lines.append(f"        r += y * {i + 1}")
    pad = base_lines - len(lines) - 1
    for j in range(max(pad, 0)):
        lines.append(f"    r += {j}")
    lines.append("    return r")
    return "\n".join(lines)


def _low_mi_src() -> str:
    # Uzun, dallanan, yorumsuz — MI < LOW_MI (20) saglamasi icin
    # 300+ satir, 60+ dallanma, sifir yorum
    lines = ["def huge_complex(a, b, c, d, e, f, g, h, i, j, k, l):"]
    lines.append("    result = 0")
    for n in range(60):
        lines += [
            f"    v{n}a = a * {n} + b - c",
            f"    v{n}b = d / max({n + 1}, e) + f",
            f"    v{n}c = g ** 2 - h * {n} + i",
            f"    if v{n}a > v{n}b:",
            f"        result += v{n}a * v{n}c - j * {n}",
            f"    elif v{n}b > v{n}c:",
            f"        result -= v{n}b + k * {n} // max({n + 1}, 1)",
        ]
    lines.append("    return result")
    return "\n".join(lines)


# ── Birim testler ─────────────────────────────────────────────────

def test_long_method_detected(tmp_path):
    p = _write(tmp_path, "lm.py", _long_method_src())
    r = detect_smells(p)
    assert r["smell_long_method"] >= 1


def test_long_method_not_detected_for_short_func(tmp_path):
    src = "def short(): return 1\n"
    p = _write(tmp_path, "clean.py", src)
    r = detect_smells(p)
    assert r["smell_long_method"] == 0


def test_large_class_detected(tmp_path):
    p = _write(tmp_path, "lc.py", _large_class_src())
    r = detect_smells(p)
    assert r["smell_large_class"] >= 1


def test_large_class_not_detected_few_methods(tmp_path):
    # LOC > esik ama method sayisi az → smell yok
    lines = ["class Small:"]
    for i in range(3):
        lines.append(f"    def m{i}(self): pass")
    for j in range(LARGE_CLASS_LOC + 5):
        lines.append(f"    _x{j} = {j}")
    p = _write(tmp_path, "small.py", "\n".join(lines))
    r = detect_smells(p)
    assert r["smell_large_class"] == 0


def test_long_param_list_detected(tmp_path):
    p = _write(tmp_path, "lp.py", _long_param_src())
    r = detect_smells(p)
    assert r["smell_long_param_list"] >= 1


def test_deep_nesting_detected(tmp_path):
    p = _write(tmp_path, "dn.py", _deep_nesting_src())
    r = detect_smells(p)
    assert r["smell_deep_nesting"] >= 1


def test_deep_nesting_not_detected_shallow(tmp_path):
    src = "def f(x):\n    if x > 0:\n        return x\n"
    p = _write(tmp_path, "shallow.py", src)
    r = detect_smells(p)
    assert r["smell_deep_nesting"] == 0


def test_high_complexity_detected(tmp_path):
    p = _write(tmp_path, "hc.py", _high_cc_src())
    r = detect_smells(p)
    assert r["smell_high_complexity"] >= 1


def test_god_function_detected(tmp_path):
    p = _write(tmp_path, "god.py", _god_function_src())
    r = detect_smells(p)
    assert r["smell_god_function"] >= 1


def test_low_maintainability_detected():
    src = _low_mi_src()
    r = detect_smells_from_source(src)
    assert r["smell_low_maintainability"] == 1, (
        "MI >= LOW_MI (%d) — kod uretecini guclendir veya esigi kontrol et" % LOW_MI
    )


def test_no_smells_in_clean_code(tmp_path):
    src = "def add(a, b):\n    return a + b\n"
    p = _write(tmp_path, "clean.py", src)
    r = detect_smells(p)
    assert r["smell_count"] == 0


def test_smell_count_aggregates_correctly(tmp_path):
    # 2 uzun fonksiyon + 1 uzun param → smell_count en az 3
    lines = []
    # Fonksiyon 1: uzun
    lines.append("def func_a():")
    for i in range(LONG_METHOD_LOC + 5):
        lines.append(f"    x{i} = {i}")
    lines.append("    return x0")
    # Fonksiyon 2: uzun
    lines.append("def func_b():")
    for i in range(LONG_METHOD_LOC + 5):
        lines.append(f"    y{i} = {i}")
    lines.append("    return y0")
    # Fonksiyon 3: uzun param
    args = ", ".join(f"p{i}" for i in range(LONG_PARAM_COUNT + 2))
    lines.append(f"def func_c({args}): pass")

    p = _write(tmp_path, "multi.py", "\n".join(lines))
    r = detect_smells(p)
    assert r["smell_long_method"] == 2
    assert r["smell_long_param_list"] == 1
    assert r["smell_count"] >= 3


def test_detect_smells_batch_handles_malformed(tmp_path):
    """Bozuk Python, skip_errors=True ile exception firlatmamali."""
    bad = tmp_path / "bad.py"
    bad.write_text("def f(\n  # bozuk", encoding="utf-8")
    results = detect_smells_batch([bad], skip_errors=True)
    assert bad in results
    assert results[bad]["smell_count"] == 0  # empty fallback


def test_detect_smells_batch_missing_file(tmp_path):
    """Olmayan dosya, skip_errors=True ile bos sozluk donmeli."""
    ghost = tmp_path / "ghost.py"
    results = detect_smells_batch([ghost], skip_errors=True)
    assert ghost in results
    assert results[ghost]["smell_count"] == 0


def test_detect_smells_batch_aggregates_multiple(tmp_path):
    """3 dosyalik batch — her dosya icin sonuc var."""
    files = []
    for i in range(3):
        p = _write(tmp_path, f"f{i}.py", f"def add_{i}(a, b): return a + b\n")
        files.append(p)
    results = detect_smells_batch(files)
    assert len(results) == 3
    for f in files:
        assert f in results
        assert results[f]["smell_count"] == 0


def test_all_smell_keys_present(tmp_path):
    """Her sonuc sozlugunde 8 beklenen anahtar olmali."""
    p = _write(tmp_path, "x.py", "def f(): pass\n")
    r = detect_smells(p)
    expected_keys = {
        "smell_count", "smell_long_method", "smell_large_class",
        "smell_long_param_list", "smell_deep_nesting",
        "smell_high_complexity", "smell_low_maintainability",
        "smell_god_function",
    }
    assert expected_keys == set(r.keys())
