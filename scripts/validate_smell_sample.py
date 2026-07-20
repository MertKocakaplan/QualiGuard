"""
validate_smell_sample.py — AST+radon vs Prospector Cohen's kappa validation.

PLAN §8 (F6):
    Stratified sample of Python files (default: 50).
    Runs detect_smells() and optionally run_prospector() on each file.
    Reports Cohen's kappa (Cohen 1960; Landis & Koch 1977) for paper's
    methodology section.

    Acceptance criterion: kappa >= 0.4 (Landis & Koch "fair agreement").

Usage:
    # With prospector (requires: pip install prospector)
    python -m scripts.validate_smell_sample

    # Without prospector — AST-only report, kappa skipped
    python -m scripts.validate_smell_sample --skip-prospector

    # Custom sample size + source directory
    python -m scripts.validate_smell_sample --n 100 --source-dir pipeline/

    # Use specific file list
    python -m scripts.validate_smell_sample --file-list path/to/files.txt

    # Dry-run with synthetic files (for CI/testing)
    python -m scripts.validate_smell_sample --dry-run

Output:
    output/figures/smell_validation_kappa.csv
    output/figures/smell_validation_report.txt
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
OUTPUT_DIR   = PROJECT_ROOT / "output" / "figures"

sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.code_smells import detect_smells  # noqa: E402


# ── Prospector integration (optional) ───────────────────────────

def _try_import_prospector():
    """Return run_prospector callable or None if not available."""
    try:
        from pipeline.prospector_runner import run_prospector  # noqa: F401
        return run_prospector
    except ImportError:
        return None


# ── Synthetic dry-run helpers ────────────────────────────────────

_CLEAN_SNIPPET = textwrap.dedent("""\
    def add(a: int, b: int) -> int:
        return a + b

    def greet(name: str) -> str:
        return f"Hello, {name}!"
""")

_SMELLY_SNIPPET = textwrap.dedent("""\
    def long_complex_function(a, b, c, d, e, f, g):
        result = 0
        if a > 0:
            for i in range(b):
                if i % 2 == 0:
                    if c > 0:
                        if d > 0:
                            result += a * b * c * d
                        else:
                            result -= a
                    else:
                        result += c
                else:
                    while result < 100:
                        if result > 50:
                            result -= 1
                        else:
                            result += 2
        elif b < 0:
            for j in range(abs(b)):
                result -= j * a
        else:
            result = c + d + e + f + g
        return result
""")

_SMELLY_SNIPPET = _SMELLY_SNIPPET + "\n    pass\n" * 30  # force LOC > 50


def _write_dry_run_files(tmp_dir: Path, n: int, seed: int) -> list[Path]:
    """Write n synthetic .py files: ~half clean, ~half smelly."""
    rng = random.Random(seed)
    files = []
    for i in range(n):
        fname = tmp_dir / f"sample_{i:04d}.py"
        content = _SMELLY_SNIPPET if rng.random() > 0.5 else _CLEAN_SNIPPET
        fname.write_text(content, encoding="utf-8")
        files.append(fname)
    return files


# ── File collection ──────────────────────────────────────────────

def _collect_project_files(source_dir: Path, n: int, seed: int) -> list[Path]:
    """Collect up to n Python files from source_dir (excluding venv/__pycache__)."""
    skip_parts = {"venv", "__pycache__", ".git", "node_modules", "site-packages"}
    all_py = [
        f for f in source_dir.rglob("*.py")
        if not any(p in f.parts for p in skip_parts)
        and f.stat().st_size > 0
    ]
    if not all_py:
        raise FileNotFoundError(f"No .py files found in {source_dir}")
    rng = random.Random(seed)
    rng.shuffle(all_py)
    return all_py[:n]


def _load_file_list(path: Path) -> list[Path]:
    return [
        Path(line.strip()) for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


# ── Cohen's kappa ────────────────────────────────────────────────

def cohen_kappa(a: list[int], b: list[int]) -> float:
    """Compute Cohen's kappa for two binary label sequences."""
    if not a or not b:
        return 0.0
    try:
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(a, b))
    except ImportError:
        # Manual fallback
        n = len(a)
        if n == 0:
            return 0.0
        po = sum(x == y for x, y in zip(a, b)) / n
        pa = (sum(a) / n) * (sum(b) / n) + (sum(1 - x for x in a) / n) * (sum(1 - y for y in b) / n)
        return (po - pa) / (1 - pa) if pa < 1.0 else 1.0


def landis_koch_label(kappa: float) -> str:
    """Landis & Koch (1977) agreement category."""
    if kappa < 0.00:
        return "Poor (<0)"
    if kappa < 0.20:
        return "Slight (0–0.20)"
    if kappa < 0.40:
        return "Fair (0.20–0.40)"
    if kappa < 0.60:
        return "Moderate (0.40–0.60)"
    if kappa < 0.80:
        return "Substantial (0.60–0.80)"
    return "Almost Perfect (0.80–1.0)"


# ── Core validation ──────────────────────────────────────────────

def _run_ast_smell(file_path: Path) -> Optional[int]:
    """Return smell_count or None on error."""
    try:
        result = detect_smells(file_path)
        return int(result.get("smell_count", 0))
    except Exception as exc:
        logger.debug("AST smell error on %s: %s", file_path, exc)
        return None


def _run_prospector_smell(run_prospector_fn, file_path: Path) -> Optional[int]:
    """Return smell_count (None on error / timeout)."""
    try:
        result = run_prospector_fn(file_path)
        cnt = result.get("smell_count")
        return int(cnt) if cnt is not None else None
    except Exception as exc:
        logger.debug("Prospector error on %s: %s", file_path, exc)
        return None


def validate(
    files: list[Path],
    prospector_fn,
    output_dir: Path,
) -> dict:
    """
    Run AST smell detection (and optionally Prospector) on all files.

    Returns summary dict.
    """
    rows: list[dict] = []
    ast_errors   = 0
    pros_errors  = 0
    use_prospector = prospector_fn is not None

    for i, fpath in enumerate(files):
        if i % 10 == 0:
            logger.info("  [%d/%d] %s", i + 1, len(files), fpath.name)

        ast_cnt = _run_ast_smell(fpath)
        if ast_cnt is None:
            ast_errors += 1

        pros_cnt = None
        if use_prospector:
            pros_cnt = _run_prospector_smell(prospector_fn, fpath)
            if pros_cnt is None:
                pros_errors += 1

        rows.append({
            "file_path":      str(fpath),
            "ast_smell_count":        ast_cnt if ast_cnt is not None else -1,
            "ast_has_smell":          1 if (ast_cnt is not None and ast_cnt > 0) else 0,
            "prospector_smell_count": pros_cnt if pros_cnt is not None else -1,
            "prospector_has_smell":   1 if (pros_cnt is not None and pros_cnt > 0) else 0,
        })

    # kappa — only for rows where both detectors returned a value
    valid_pairs = [
        (r["ast_has_smell"], r["prospector_has_smell"])
        for r in rows
        if r["ast_smell_count"] >= 0 and r["prospector_smell_count"] >= 0
    ]

    kappa: Optional[float] = None
    if use_prospector and valid_pairs:
        ast_labels  = [p[0] for p in valid_pairs]
        pros_labels = [p[1] for p in valid_pairs]
        kappa = cohen_kappa(ast_labels, pros_labels)

    # ── Save CSV ─────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "smell_validation_kappa.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "file_path",
            "ast_smell_count", "ast_has_smell",
            "prospector_smell_count", "prospector_has_smell",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Save report ──────────────────────────────────────────────
    report_path = output_dir / "smell_validation_report.txt"
    n_total  = len(rows)
    n_valid  = len(valid_pairs)
    ast_pos  = sum(r["ast_has_smell"] for r in rows if r["ast_smell_count"] >= 0)
    pros_pos = sum(r["prospector_has_smell"] for r in rows if r["prospector_smell_count"] >= 0)

    lines = [
        "=" * 60,
        "QualiGuard — Smell Validation Report (F6)",
        "=" * 60,
        f"Total files sampled  : {n_total}",
        f"AST analysis errors  : {ast_errors}",
        f"AST smelly files     : {ast_pos} / {n_total - ast_errors}",
        "",
    ]
    if use_prospector:
        lines += [
            f"Prospector errors    : {pros_errors}",
            f"Prospector smelly    : {pros_pos} / {n_total - pros_errors}",
            f"Valid pairs (both ok): {n_valid}",
            "",
        ]
        if kappa is not None:
            agreement = landis_koch_label(kappa)
            passed    = kappa >= 0.4
            lines += [
                f"Cohen's kappa        : {kappa:.4f}",
                f"Agreement level      : {agreement}",
                f"Acceptance (>= 0.4)  : {'PASS ✓' if passed else 'FAIL ✗'}",
            ]
        else:
            lines.append("Cohen's kappa        : N/A (no valid pairs)")
    else:
        lines.append("Prospector           : not run (--skip-prospector)")
        lines.append("Cohen's kappa        : N/A")

    lines += ["", f"Output CSV           : {csv_path}"]
    report = "\n".join(lines)
    report_path.write_text(report, encoding="utf-8")
    print(report)

    return {
        "n_total":    n_total,
        "n_valid":    n_valid,
        "ast_errors": ast_errors,
        "kappa":      kappa,
        "csv_path":   str(csv_path),
    }


# ── CLI ──────────────────────────────────────────────────────────

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate smell detection: AST vs Prospector.")
    p.add_argument("--n",             type=int, default=50,   help="Sample size (default 50)")
    p.add_argument("--seed",          type=int, default=42,   help="Random seed")
    p.add_argument("--source-dir",    type=Path, default=PROJECT_ROOT, help="Root to sample .py files from")
    p.add_argument("--file-list",     type=Path, default=None, help="Text file with one path per line")
    p.add_argument("--skip-prospector", action="store_true",  help="Run AST only, skip Prospector")
    p.add_argument("--dry-run",         action="store_true",  help="Use synthetic files (for CI/testing)")
    p.add_argument("--output-dir",    type=Path, default=OUTPUT_DIR, help="Output directory")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    # ── File collection ──────────────────────────────────────────
    if args.dry_run:
        logger.info("Dry-run mode: generating %d synthetic files...", args.n)
        tmp_dir = Path(tempfile.mkdtemp(prefix="smell_validation_"))
        files = _write_dry_run_files(tmp_dir, args.n, args.seed)
    elif args.file_list:
        files = _load_file_list(args.file_list)[: args.n]
        logger.info("Loaded %d files from %s", len(files), args.file_list)
    else:
        logger.info("Collecting %d .py files from %s ...", args.n, args.source_dir)
        try:
            files = _collect_project_files(args.source_dir, args.n, args.seed)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            return 1
        logger.info("Collected %d files", len(files))

    # ── Prospector setup ─────────────────────────────────────────
    prospector_fn = None
    if not args.skip_prospector:
        prospector_fn = _try_import_prospector()
        if prospector_fn is None:
            logger.warning(
                "Prospector not available (run `pip install prospector` to enable). "
                "Running AST-only analysis."
            )
        else:
            logger.info("Prospector available — running full comparison.")

    # ── Validation ───────────────────────────────────────────────
    logger.info("Starting validation on %d files...", len(files))
    summary = validate(files, prospector_fn, args.output_dir)

    kappa = summary["kappa"]
    if kappa is not None and kappa < 0.4:
        logger.warning(
            "Cohen's kappa = %.4f < 0.4 — academic defence may be weakened. "
            "Consider tuning smell detection thresholds.",
            kappa,
        )
        return 2  # non-zero but not error — signals failed acceptance criterion

    return 0


if __name__ == "__main__":
    sys.exit(main())
