"""
scripts/collect.py — Full veri toplama CLI.

CLI contract (PLAN §12.1):

    python -m scripts.collect [OPTIONS]

Fazlar:
    - discovery → GitHub search + contributor filter + output/checkpoints/discovery.json
    - process   → her proje icin clone + radon + git + SZZ + prospector
                  → output/projects/<safe>.parquet (atomic)
                  → output/checkpoints/processed_projects.json
    - build     → tum per-project parquet'leri birlestir
                  → output/dataset_full_<ts>.parquet
    - all       → sirayla discovery + process + build

Exit codes (PLAN §12.1):
    0   basarili
    1   genel hata
    2   config hatasi
    130 user interrupt (SIGINT)
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from pipeline import checkpoint as checkpoint_mod
from pipeline import dataset_builder, discovery, project_processor
from pipeline.config import (
    CHECKPOINT_DIR,
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
    p.add_argument("--skip-smells", action="store_true",
                   help="Smell tespit adimini atla")
    p.add_argument("--skip-prospector", action="store_true",
                   help="(deprecated) --skip-smells alias")
    p.add_argument("--workers", type=int, default=PROSPECTOR_WORKERS,
                   help="(deprecated) AST smell detection worker gerektirmiyor")
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
    print(f"  skip-smells      : {args.skip_smells or args.skip_prospector}")
    print(f"  workers          : {args.workers}")
    print(f"  output-dir       : {args.output_dir}")
    print(f"  log-level        : {args.log_level}")
    print(f"  GITHUB_TOKEN     : {token_note}")
    print("=" * 60)
    print("NOT: --dry-run aktif, hicbir yazim/cagri yapilmadi.")
    print(f"Beklenen log yolu : {LOGS_DIR}/collect_<ts>.log")


def _run_discovery(args: argparse.Namespace) -> int:
    """Discovery fazi — GitHub search + contributor filter."""
    out_path = CHECKPOINT_DIR / "discovery.json"

    # Mevcut meta'yi oku — yeni run'dan once, sonradan merge icin
    existing_by_name: dict[str, dict] = {}
    if out_path.exists():
        try:
            raw = json.loads(out_path.read_text(encoding="utf-8"))
            existing_by_name = {
                p["full_name"]: p
                for p in raw.get("found", [])
                if p.get("full_name")
            }
        except (json.JSONDecodeError, OSError):
            pass

    refresh_quota()
    quota = current_quota()
    logger.info("GitHub quota: %s", quota)

    new_results = discovery.search_projects(
        target_count=args.target,
        min_age_days=args.min_age_days,
        max_age_days=args.max_age_days,
        max_contributors=args.max_contributors,
        min_stars=args.min_stars,
    )
    logger.info("discovery sonucu: %d proje", len(new_results))

    # Merge: yeni meta kazanir, onceki run'dan gelen projeler korunur.
    # Boylece kucuk --target ile calisinca buyuk onceki dataset kaybolmaz.
    merged = dict(existing_by_name)
    new_count = sum(1 for p in new_results if p["full_name"] not in merged)
    for p in new_results:
        merged[p["full_name"]] = p

    checkpoint_mod.save_checkpoint("discovery", {
        "found": list(merged.values()),
        "stats": {
            "total":        len(merged),
            "this_run_new": new_count,
            "previous":     len(existing_by_name),
        },
    })
    logger.info("discovery.json guncellendi: toplam=%d (yeni=%d)", len(merged), new_count)
    return 0


def _load_discovered_projects() -> list[dict]:
    """discovery.json'dan bulunan proje listesini oku. Yoksa [] doner."""
    path = CHECKPOINT_DIR / "discovery.json"
    if not path.exists():
        logger.error("discovery.json yok: %s (once --phase discovery calistir)", path)
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("discovery.json okunamadi: %s", exc)
        return []
    return list(data.get("found", []))


def _run_process(args: argparse.Namespace) -> int:
    """
    Process fazi — her proje icin full hat calistir.

    - discovery.json'dan proje listesini al
    - --resume aktifse processed_projects'teki status=ok'lari atla
    - pipeline.project_processor.process_project ile isle
    - her proje sonrasi checkpoint.mark_project_done (atomic)
    - KeyboardInterrupt yakalanir, son checkpoint'ten devam edilebilir
    """
    projects = _load_discovered_projects()
    if not projects:
        return 1

    processed = checkpoint_mod.get_processed_set() if args.resume else set()
    total = len(projects)
    if processed:
        logger.info("resume aktif — %d proje atlanacak (status=ok)", len(processed))

    stats = {"ok": 0, "failed": 0, "empty": 0, "skipped": 0}
    t_start = time.monotonic()
    durations: list[float] = []
    total_loc = 0
    total_files = 0

    for idx, proj in enumerate(projects, 1):
        name = proj.get("full_name", "<noname>")

        if name in processed:
            stats["skipped"] += 1
            continue

        logger.info("[%d/%d] proje: %s", idx, total, name)
        try:
            result = project_processor.process_project(
                proj,
                skip_szz=args.skip_szz,
                skip_smells=args.skip_smells or args.skip_prospector,
                workers=args.workers,
            )
        except KeyboardInterrupt:
            logger.warning("SIGINT — ara ciktida. Devam icin --resume kullan.")
            return 130
        except Exception as exc:  # noqa: BLE001 — her proje bagimsiz
            logger.exception("proje patladi: %s", name)
            result = {
                "status":       "failed",
                "error":        f"unhandled: {exc}",
                "completed_at": datetime.now().isoformat(),
            }

        checkpoint_mod.mark_project_done(name, result)
        status = result.get("status", "failed")
        stats[status] = stats.get(status, 0) + 1
        if "duration_secs" in result:
            durations.append(float(result["duration_secs"]))
        total_loc   += int(result.get("total_loc", 0) or 0)
        total_files += int(result.get("files", 0) or 0)

        # Periyodik progress
        if idx % 10 == 0:
            logger.info(
                "progress %d/%d — ok=%d failed=%d empty=%d skipped=%d",
                idx, total, stats["ok"], stats["failed"],
                stats["empty"], stats["skipped"],
            )

    # Ozet (PLAN F3 DoD: toplam LOC, proje sayisi, basari orani, ortalama sure)
    elapsed    = round(time.monotonic() - t_start, 1)
    avg        = round(sum(durations) / len(durations), 1) if durations else 0.0
    attempted  = total - stats["skipped"]
    success_pc = (stats["ok"] / attempted * 100.0) if attempted else 0.0
    fail_ratio = (stats["failed"] / attempted) if attempted else 0.0
    logger.info(
        "process ozeti: total=%d ok=%d failed=%d empty=%d skipped=%d  "
        "(success %.1f%%, elapsed %.1fs, avg/proj %.1fs)",
        total, stats["ok"], stats["failed"], stats["empty"], stats["skipped"],
        success_pc, elapsed, avg,
    )
    logger.info(
        "veri ozeti: toplam_dosya=%d toplam_loc=%d  (yalnizca status=ok projelerden)",
        total_files, total_loc,
    )
    if attempted and fail_ratio > 0.10:
        logger.warning("failed orani >%%10 — log'u incele.")
    return 0


def _run_build(args: argparse.Namespace) -> int:
    """Build fazi — per-project parquet'leri birlestir."""
    out = dataset_builder.build_full_dataset(output_dir=args.output_dir)
    if out is None:
        logger.error("build: birlestirilecek parquet yok")
        return 1
    logger.info("build: dataset_full yazildi: %s", out)
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
            rc = _run_process(args)
            if rc != 0:
                return rc

        if args.phase in ("build", "all"):
            rc = _run_build(args)
            if rc != 0:
                return rc

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
