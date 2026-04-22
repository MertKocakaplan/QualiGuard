"""
scripts/train_final.py — Final model egitimi CLI.

CLI contract (PLAN §12.2):

    python -m scripts.train_final [OPTIONS]

F1 kapsami: argparse + --dry-run. Egitim mantigi F5/F6'da eklenir.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pipeline.config import MODELS_DIR, OUTPUT_DIR

logger = logging.getLogger("train_final")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scripts.train_final",
        description="MetricHunter V2 — Final model egitimi (F6).",
    )
    p.add_argument("--dataset", type=Path, default=None,
                   help="Filtered parquet (varsayilan: son bulunan)")
    p.add_argument("--tasks", type=str, default="commit,bug,smell",
                   help="Virgulle ayrilmis: commit,bug,smell")
    p.add_argument("--bug-label", choices=("keyword", "szz"), default="szz",
                   help="Bug etiket kaynagi")
    p.add_argument("--smell-label", choices=("binary", "count"), default="binary",
                   help="Smell etiket turu")
    p.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                   help="Model artifact cikti dizini")
    p.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                   default="INFO")
    p.add_argument("--dry-run", action="store_true",
                   help="Sadece config raporu; egitim yapma")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    if args.dry_run:
        print("=" * 60)
        print("  scripts.train_final  [--dry-run]")
        print("=" * 60)
        print(f"  dataset     : {args.dataset or '(son filtered)'}")
        print(f"  tasks       : {tasks}")
        print(f"  bug-label   : {args.bug_label}")
        print(f"  smell-label : {args.smell_label}")
        print(f"  models-dir  : {args.models_dir}")
        print(f"  log-level   : {args.log_level}")
        print("=" * 60)
        print("NOT: F1'de dry-run disinda egitim kodu yok; F5/F6'da eklenir.")
        return 0

    logger.warning("train_final egitim kodu F5/F6'da implement edilecek.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
