"""
scripts/collect.py — Full veri toplama CLI.

CLI contract (PLAN §12.1):

    python -m scripts.collect [OPTIONS]

F1 kapsami:
    - Argparse + logging + ortam kurulumu
    - --dry-run: config'i rapor et, hicbir sey yazma
    - --phase discovery: 10 projelik mini discovery (dogrulama)
    - --phase process / build / all: F2+F3'te tamamlanir

Exit codes (PLAN §12.1):
    0   basarili
    1   genel hata
    2   config hatasi
    130 user interrupt (SIGINT)
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

from pipeline import checkpoint as checkpoint_mod
from pipeline import discovery
from pipeline.config import (
    DEFAULT_MAX_AGE_DAYS,
    DEFAULT_MAX_CONTRIBUTORS,
    DEFAULT_MIN_AGE_DAYS,
    DEFAULT_MIN_STARS,
    DEFAULT_TARGET_COUNT,
    LOG_DATEFMT,
    LOG_FORMAT,
    LOGS_DIR,
    OUTPUT_DIR,
    PROSPECTOR_WORKERS,
    ensure_runtime_dirs,
)
from pipeline.rate_limit import current_quota, github_token_configured, refresh_quota

logger = logging.getLogger("collect")


PHASES = ("discovery", "process", "build", "all")


def build_parser() -> argparse.ArgumentParser:
    """CLI parser — PLAN §12.1 ile uyumlu."""
    p = argparse.ArgumentParser(
        prog="scripts.collect",
        description="MetricHunter V2 — Full veri toplama CLI (F1-F3).",
    )
    p.add_argument("--target", type=int, default=DEFAULT_TARGET_COUNT,
                   help="Hedef proje sayisi")
    p.add_argument("--min-age-days", type=int, default=DEFAULT_MIN_AGE_DAYS,
                   help="Minimum proje yasi")
    p.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS,
                   help="Maksimum proje yasi")
    p.add_argument("--max-contributors", type=int, default=DEFAULT_MAX_CONTRIBUTORS,
                   help="Maksimum contributor sayisi")
    p.add_argument("--min-stars", type=int, default=DEFAULT_MIN_STARS,
                   help="Minimum yildiz")
    p.add_argument("--phase", choices=PHASES, default="all",
                   help="Calistirilacak faz")
    p.add_argument("--resume", action="store_true",
                   help="Checkpoint'ten devam et")
    p.add_argument("--skip-szz", action="store_true",
                   help="SZZ adimini atla")
    p.add_argument("--skip-prospector", action="store_true",
                   help="Prospector adimini atla")
    p.add_argument("--workers", type=int, default=PROSPECTOR_WORKERS,
                   help="Prospector paralel worker sayisi")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                   help="Cikti dizini")
    p.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                   default="INFO", help="Log seviyesi")
    p.add_argument("--dry-run", action="store_true",
                   help="Hicbir sey yazma, sadece config raporu uret")
    return p


def _setup_logging(level: str, log_file: Path | None) -> None:
    """stdout + opsiyonel dosya handler."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level),
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        handlers=handlers,
        force=True,
    )


def _validate_args(args: argparse.Namespace) -> None:
    """Config dogrulamasi. Hatayi `SystemExit(2)` ile raporlar."""
    if args.target <= 0:
        raise SystemExit("HATA: --target pozitif olmali. (exit 2)")
    if args.min_age_days < 0 or args.max_age_days < 0:
        raise SystemExit("HATA: --*-age-days negatif olamaz. (exit 2)")
    if args.min_age_days > args.max_age_days:
        raise SystemExit("HATA: --min-age-days > --max-age-days. (exit 2)")
    if args.max_contributors <= 0:
        raise SystemExit("HATA: --max-contributors pozitif olmali. (exit 2)")
    if args.min_stars < 0:
        raise SystemExit("HATA: --min-stars negatif olamaz. (exit 2)")
    if args.workers <= 0:
        raise SystemExit("HATA: --workers pozitif olmali. (exit 2)")


def _print_dry_run(args: argparse.Namespace) -> None:
    """--dry-run: config'i ekrana bas, hicbir sey yazma."""
    token_note = "ayarli" if github_token_configured() else "YOK (rate limit cok dusuk)"
    print("=" * 60)
    print("  MetricHunter V2 — scripts.collect  [--dry-run]")
    print("=" * 60)
    print(f"  target           : {args.target}")
    print(f"  min-age-days     : {args.min_age_days}")
    print(f"  max-age-days     : {args.max_age_days}")
    print(f"  max-contributors : {args.max_contributors}")
    print(f"  min-stars        : {args.min_stars}")
    print(f"  phase            : {args.phase}")
    print(f"  resume           : {args.resume}")
    print(f"  skip-szz         : {args.skip_szz}")
    print(f"  skip-prospector  : {args.skip_prospector}")
    print(f"  workers          : {args.workers}")
    print(f"  output-dir       : {args.output_dir}")
    print(f"  log-level        : {args.log_level}")
    print(f"  GITHUB_TOKEN     : {token_note}")
    print("=" * 60)
    print("NOT: --dry-run aktif, hicbir yazim/cagri yapilmadi.")
    print(f"Beklenen log yolu : {LOGS_DIR}/collect_<ts>.log")


def _run_discovery(args: argparse.Namespace) -> int:
    """Discovery fazini calistir. F2+F3'te process/build eklenir."""
    refresh_quota()
    quota = current_quota()
    logger.info("GitHub quota: %s", quota)

    found = discovery.search_projects(
        target_count=args.target,
        min_age_days=args.min_age_days,
        max_age_days=args.max_age_days,
        max_contributors=args.max_contributors,
        min_stars=args.min_stars,
    )
    logger.info("discovery sonucu: %d proje", len(found))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point. argv=None ise sys.argv[1:] kullanilir."""
    args = build_parser().parse_args(argv)
    _validate_args(args)

    # SIGINT graceful exit (exit 130)
    def _sigint(sig, frame):
        logger.warning("SIGINT alindi, cikiliyor.")
        sys.exit(130)
    signal.signal(signal.SIGINT, _sigint)

    if args.dry_run:
        _print_dry_run(args)
        return 0

    ensure_runtime_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    _setup_logging(args.log_level, LOGS_DIR / f"collect_{ts}.log")

    logger.info("scripts.collect basladi (phase=%s)", args.phase)

    try:
        if args.phase in ("discovery", "all"):
            rc = _run_discovery(args)
            if rc != 0:
                return rc

        if args.phase in ("process", "all"):
            logger.warning("phase=process F2'de implement edilecek (SZZ + Prospector).")
            if args.phase == "process":
                return 1

        if args.phase in ("build", "all"):
            logger.warning("phase=build F3'te implement edilecek (dataset_full birlesimi).")
            if args.phase == "build":
                return 1

        logger.info("scripts.collect tamamlandi.")
        return 0

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt yakalandi.")
        return 130
    except Exception as exc:  # noqa: BLE001 — CLI'nin dis kabugu
        logger.exception("Genel hata: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
