"""
test_validate_smell_sample.py — F6 sample validation script unit tests.

Tests the pure functions and the CLI entrypoint (--dry-run --skip-prospector)
without requiring Prospector to be installed or GitHub API access.
"""
from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import pytest

from scripts.validate_smell_sample import (
    cohen_kappa,
    landis_koch_label,
    main,
    validate,
)


# ── cohen_kappa ──────────────────────────────────────────────────

def test_cohen_kappa_perfect_agreement():
    a = [0, 0, 1, 1, 1]
    assert cohen_kappa(a, a) == pytest.approx(1.0, rel=1e-3)


def test_cohen_kappa_complete_disagreement():
    a = [0, 0, 1, 1]
    b = [1, 1, 0, 0]
    k = cohen_kappa(a, b)
    assert k < 0.0


def test_cohen_kappa_chance_level():
    """All-zeros vs all-ones should give kappa near -1."""
    a = [1] * 10
    b = [0] * 10
    k = cohen_kappa(a, b)
    assert k <= 0.0


def test_cohen_kappa_partial_agreement():
    a = [0, 1, 1, 0, 1]
    b = [0, 1, 0, 0, 1]
    k = cohen_kappa(a, b)
    assert 0.0 < k < 1.0


def test_cohen_kappa_empty_lists():
    """Edge case: empty lists should return 0 without crashing."""
    k = cohen_kappa([], [])
    assert k == 0.0


# ── landis_koch_label ────────────────────────────────────────────

def test_landis_koch_poor():
    assert landis_koch_label(-0.05) == "Poor (<0)"


def test_landis_koch_slight():
    assert landis_koch_label(0.10) == "Slight (0–0.20)"


def test_landis_koch_fair():
    assert landis_koch_label(0.30) == "Fair (0.20–0.40)"


def test_landis_koch_moderate():
    assert landis_koch_label(0.50) == "Moderate (0.40–0.60)"


def test_landis_koch_substantial():
    assert landis_koch_label(0.70) == "Substantial (0.60–0.80)"


def test_landis_koch_almost_perfect():
    assert landis_koch_label(0.90) == "Almost Perfect (0.80–1.0)"


# ── validate() — no prospector ───────────────────────────────────

def test_validate_ast_only_no_crash(tmp_path):
    """validate() with no prospector_fn produces a valid CSV and returns summary."""
    # Write a trivial .py file
    f = tmp_path / "tiny.py"
    f.write_text("def add(a, b): return a + b\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    summary = validate([f], prospector_fn=None, output_dir=out_dir)

    assert summary["n_total"] == 1
    assert summary["kappa"] is None
    assert Path(summary["csv_path"]).exists()


def test_validate_csv_has_expected_columns(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    validate([f], prospector_fn=None, output_dir=out_dir)
    csv_path = out_dir / "smell_validation_kappa.csv"
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames
    assert "ast_smell_count" in cols
    assert "ast_has_smell"   in cols
    assert "prospector_smell_count" in cols


def test_validate_report_written(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("def f(): return 1\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    validate([f], prospector_fn=None, output_dir=out_dir)
    assert (out_dir / "smell_validation_report.txt").exists()


def test_validate_with_mock_prospector(tmp_path):
    """validate() with a mock prospector_fn computes kappa."""
    files = []
    for i in range(4):
        f = tmp_path / f"f{i}.py"
        f.write_text("def g(): pass\n", encoding="utf-8")
        files.append(f)

    # Mock: first 2 smelly, last 2 clean
    call_idx = [0]
    def mock_prospector(path):
        idx = call_idx[0]
        call_idx[0] += 1
        return {"smell_count": 3 if idx < 2 else 0}

    out_dir = tmp_path / "out"
    summary = validate(files, prospector_fn=mock_prospector, output_dir=out_dir)

    assert summary["n_valid"] == 4
    assert summary["kappa"] is not None


def test_validate_prospector_errors_handled(tmp_path):
    """Prospector errors are counted but do not crash the run."""
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")

    def bad_prospector(path):
        raise RuntimeError("connection refused")

    out_dir = tmp_path / "out"
    summary = validate([f], prospector_fn=bad_prospector, output_dir=out_dir)
    assert summary["n_total"] == 1
    assert summary["kappa"] is None  # no valid pairs


# ── CLI dry-run ──────────────────────────────────────────────────

def test_cli_dry_run_skip_prospector(tmp_path):
    """CLI --dry-run --skip-prospector exits 0 and writes output."""
    rc = main([
        "--dry-run",
        "--skip-prospector",
        "--n", "10",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert rc == 0
    assert (tmp_path / "out" / "smell_validation_kappa.csv").exists()


def test_cli_dry_run_kappa_with_mock(tmp_path, monkeypatch):
    """CLI dry-run kappa path: mock prospector to return fast results."""
    import scripts.validate_smell_sample as vss

    def mock_try_import():
        def mock_prospector(path):
            return {"smell_count": 1}
        return mock_prospector

    monkeypatch.setattr(vss, "_try_import_prospector", mock_try_import)

    rc = main([
        "--dry-run",
        "--n", "8",
        "--output-dir", str(tmp_path / "out"),
    ])
    # kappa might be 0 or positive — just ensure no crash and output written
    assert rc in (0, 2)  # 2 = kappa < 0.4 (acceptable)
    assert (tmp_path / "out" / "smell_validation_kappa.csv").exists()


def test_cli_missing_source_dir(tmp_path):
    """Non-existent source-dir returns exit code 1."""
    rc = main([
        "--source-dir", str(tmp_path / "nonexistent"),
        "--skip-prospector",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert rc == 1


def test_cli_from_file_list(tmp_path):
    """--file-list argument selects specific files."""
    # Create a tiny .py file
    code_file = tmp_path / "sample.py"
    code_file.write_text("def f(): pass\n", encoding="utf-8")

    # Write file list
    flist = tmp_path / "files.txt"
    flist.write_text(str(code_file) + "\n", encoding="utf-8")

    rc = main([
        "--file-list", str(flist),
        "--skip-prospector",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert rc == 0
