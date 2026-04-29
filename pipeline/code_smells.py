"""
code_smells.py — AST + radon tabanli klasik code smell tespiti.

Fowler (1999) Refactoring + Lanza & Marinescu (2006) Object-Oriented Metrics
in Practice'tan secilmis 7 smell:

1. Long Method          — fonksiyon LOC > LONG_METHOD_LOC (50)
2. Large Class          — sinif LOC > LARGE_CLASS_LOC (500) ve method >= 10
3. Long Parameter List  — parametre sayisi > LONG_PARAM_COUNT (5)
4. Deep Nesting         — max indent depth > NESTING_DEPTH (4)
5. High Complexity      — radon CC > HIGH_CC (10)
6. Low Maintainability  — radon MI < LOW_MI (20)
7. God Function         — CC > GOD_FUNC_CC (15) ve LOC > GOD_FUNC_LOC (80)

Kullanim:
    from pipeline.code_smells import detect_smells, detect_smells_batch
    result = detect_smells(Path("foo.py"))
    # {'smell_count': 3, 'smell_long_method': 1, ...}
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path

from radon.complexity import cc_visit
from radon.metrics import mi_visit

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

logger = logging.getLogger(__name__)

_NEST_TYPES = (
    ast.If, ast.For, ast.AsyncFor, ast.While,
    ast.With, ast.AsyncWith, ast.Try,
)


def _max_nesting(node: ast.AST, depth: int = 0) -> int:
    """if/for/while/with/try ic ice en derin seviye."""
    children = list(ast.iter_child_nodes(node))
    if not children:
        return depth
    return max(
        _max_nesting(child, depth + (1 if isinstance(child, _NEST_TYPES) else 0))
        for child in children
    )


class _SmellVisitor(ast.NodeVisitor):
    """AST visitor — statik smell sayaci."""

    def __init__(self) -> None:
        self.long_method = 0
        self.large_class = 0
        self.long_param_list = 0
        self.deep_nesting = 0
        self._func_data: list[tuple[str, int]] = []  # (name, loc)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        loc = (node.end_lineno or node.lineno) - node.lineno + 1
        if loc > LONG_METHOD_LOC:
            self.long_method += 1
        self._func_data.append((node.name, loc))

        n_params = len(node.args.args) + len(node.args.kwonlyargs)
        if node.args.vararg:
            n_params += 1
        if node.args.kwarg:
            n_params += 1
        if n_params > LONG_PARAM_COUNT:
            self.long_param_list += 1

        if _max_nesting(node) > NESTING_DEPTH:
            self.deep_nesting += 1

        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        loc = (node.end_lineno or node.lineno) - node.lineno + 1
        method_count = sum(
            1 for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        if loc > LARGE_CLASS_LOC and method_count >= LARGE_CLASS_METHOD_COUNT:
            self.large_class += 1
        self.generic_visit(node)


def _empty_smells() -> dict:
    return {
        "smell_count":               0,
        "smell_long_method":         0,
        "smell_large_class":         0,
        "smell_long_param_list":     0,
        "smell_deep_nesting":        0,
        "smell_high_complexity":     0,
        "smell_low_maintainability": 0,
        "smell_god_function":        0,
    }


def detect_smells(file_path: Path) -> dict:
    """
    Tek dosyada 7 smell'i tespit et.

    Returns dict with keys:
        smell_count: int (toplam)
        smell_long_method: int
        smell_large_class: int
        smell_long_param_list: int
        smell_deep_nesting: int
        smell_high_complexity: int
        smell_low_maintainability: int  (0 veya 1, dosya seviyesinde)
        smell_god_function: int
    """
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.debug("dosya okunamadi %s: %s", file_path, exc)
        return _empty_smells()

    return detect_smells_from_source(source)


def detect_smells_from_source(source: str) -> dict:
    """Kaynak metin uzerinde smell tespiti (dosya I/O olmadan)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _empty_smells()

    visitor = _SmellVisitor()
    visitor.visit(tree)

    # radon CC — fonksiyon seviyesinde
    try:
        cc_results = cc_visit(source)
    except Exception:
        cc_results = []

    cc_map: dict[str, float] = {r.name: r.complexity for r in cc_results}

    high_cc = 0
    god_func = 0
    for name, loc in visitor._func_data:
        c = cc_map.get(name, 0)
        if c > HIGH_CC:
            high_cc += 1
        if c > GOD_FUNC_CC and loc > GOD_FUNC_LOC:
            god_func += 1

    # radon MI — dosya seviyesinde
    try:
        mi = mi_visit(source, multi=True)
    except Exception:
        mi = 100.0

    low_mi = 1 if mi < LOW_MI else 0

    total = (
        visitor.long_method
        + visitor.large_class
        + visitor.long_param_list
        + visitor.deep_nesting
        + high_cc
        + low_mi
        + god_func
    )

    return {
        "smell_count":               total,
        "smell_long_method":         visitor.long_method,
        "smell_large_class":         visitor.large_class,
        "smell_long_param_list":     visitor.long_param_list,
        "smell_deep_nesting":        visitor.deep_nesting,
        "smell_high_complexity":     high_cc,
        "smell_low_maintainability": low_mi,
        "smell_god_function":        god_func,
    }


def detect_smells_batch(
    file_paths: list[Path],
    skip_errors: bool = True,
) -> dict[Path, dict]:
    """Batch tespit — multiprocessing kullanmaz, AST zaten hizli."""
    results: dict[Path, dict] = {}
    for fp in file_paths:
        try:
            results[fp] = detect_smells(fp)
        except Exception as exc:
            if not skip_errors:
                raise
            logger.debug("smell tespiti basarisiz %s: %s", fp, exc)
            results[fp] = _empty_smells()
    return results
