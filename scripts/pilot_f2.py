"""
pilot_f2.py — F2 DoD pilot: SZZ + Prospector 5 projelik gecerlilik testi.

Adimlar:
    1. discovery.json'dan ilk N projeyi oku.
    2. Her projeyi `repos/` altina klonla (veya mevcut klonu kullan).
    3. HEAD .py dosyalarini listele, skip filtresi uygula.
    4. git log'dan bug-fix commit'lerini bul (keyword regex).
    5. SZZ ile bug_szz etiketleri uret, gerekirse keyword fallback.
    6. Prospector batch (4 worker) ile smell_count hesapla.
    7. Proje basina ozet tablo yazdir (dosya/bug_keyword/bug_szz/smell_total).

Kullanim:
    python -m scripts.pilot_f2            # 5 proje, varsayilan limit
    python -m scripts.pilot_f2 --limit 3  # 3 proje
    python -m scripts.pilot_f2 --skip-clone  # mevcut klon kullanilir

Bu script once elle gerceklestirilen bir testtir; scripts/collect.py'ye
daha sonra bu is akisi eklenir (F3).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

from pipeline import cloning, git_metrics, prospector_runner, szz
from pipeline.config import (
    CHECKPOINT_DIR,
    LOG_FORMAT,
    LOG_DATEFMT,
    PROSPECTOR_STRICTNESS,
    PROSPECTOR_TIMEOUT_SECONDS,
    PROSPECTOR_WORKERS,
    REPOS_DIR,
    ensure_runtime_dirs,
)


logger = logging.getLogger("pilot_f2")


def _bug_fix_hashes(repo_path: Path) -> list[str]:
    """
    Bug keyword iceren commit hash'lerini dondur (HEAD'den geri dogru).

    Tek `git log` cagrisi; regex git_metrics.BUG_KEYWORDS ile eslesenler.
    """
    try:
        result = subprocess.run(
            ["git", "log", "--format=%H%x1f%s", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=300,
            encoding="utf-8",
            errors="replace",
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("git log basarisiz (%s): %s", repo_path, exc)
        return []

    if result.returncode != 0:
        return []

    hashes: list[str] = []
    for line in result.stdout.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split("\x1f", 1)
        if len(parts) != 2:
            continue
        h, subject = parts
        if git_metrics.is_bug_message(subject):
            hashes.append(h)
    return hashes


def _keyword_labels(head_files: list[str], bulk_stats: dict[str, dict]) -> dict[str, int]:
    """bug_count > 0 olan HEAD dosyasi = 1, digerleri = 0."""
    out: dict[str, int] = {}
    for f in head_files:
        st = bulk_stats.get(f, {})
        out[f] = 1 if st.get("bug_count", 0) > 0 else 0
    return out


def _pick_projects(limit: int) -> list[dict]:
    path = CHECKPOINT_DIR / "discovery.json"
    if not path.exists():
        raise SystemExit(
            f"discovery.json bulunamadi: {path}. Once `scripts.collect "
            "--phase discovery --target N` calistir."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    found = data.get("found", [])
    if len(found) < limit:
        logger.warning("Sadece %d proje var (istek: %d); hepsini kullanilacak.",
                       len(found), limit)
    return found[:limit]


def _process_project(proj: dict, skip_clone: bool) -> dict:
    name = proj["full_name"]
    logger.info("── %s (stars=%d) ──", name, proj.get("stars", -1))

    # 1) Klon
    if skip_clone:
        repo_path = REPOS_DIR / cloning.safe_repo_name(proj["clone_url"])
        if not (repo_path / ".git").exists():
            return {"project": name, "status": "skipped_not_cloned"}
        clone_status = "zaten_var"
    else:
        repo_path, clone_status = cloning.clone_repo(proj["clone_url"], REPOS_DIR)
        if repo_path is None:
            logger.warning("Klon hatali (%s): %s", name, clone_status)
            return {"project": name, "status": f"clone_failed: {clone_status}"}

    # 2) HEAD dosyalari, skip filtresi
    head_files_all = git_metrics.get_head_python_files(repo_path)
    head_files = [f for f in head_files_all if not git_metrics.should_skip_file(f)]
    if not head_files:
        return {"project": name, "status": "no_python_files"}

    # 3) Bulk git stats (bug_keyword etiketinin temeli)
    bulk = git_metrics.get_bulk_git_stats(repo_path, head_files)
    bug_kw = _keyword_labels(head_files, bulk)

    # 4) SZZ icin bug-fix commit listesi
    fix_hashes = _bug_fix_hashes(repo_path)
    logger.info("  bug-fix commit: %d, head .py: %d", len(fix_hashes), len(head_files))

    # 5) SZZ — timeout ya da hata durumunda keyword fallback
    t0 = time.monotonic()
    bug_szz_labels = szz.compute_szz_labels(
        repo_path, head_files, fix_hashes,
    )
    szz_secs = round(time.monotonic() - t0, 1)
    if not bug_szz_labels:
        logger.warning("  SZZ bos geldi, bug_keyword'e fallback.")
        bug_szz_labels = dict(bug_kw)
        szz_fallback = True
    else:
        szz_fallback = False

    # 6) Prospector batch — yolları mutlak al
    abs_paths = [repo_path / f for f in head_files]
    t0 = time.monotonic()
    pros_results = prospector_runner.run_prospector_batch(
        abs_paths,
        workers=PROSPECTOR_WORKERS,
        strictness=PROSPECTOR_STRICTNESS,
        timeout_seconds=PROSPECTOR_TIMEOUT_SECONDS,
    )
    pros_secs = round(time.monotonic() - t0, 1)

    # Toplam smell_count (None'lar atilir)
    smell_total    = 0
    smell_none_cnt = 0
    for p in abs_paths:
        r = pros_results.get(p, {})
        c = r.get("smell_count")
        if c is None:
            smell_none_cnt += 1
        else:
            smell_total += c

    # 7) Ozet
    summary = {
        "project":        name,
        "status":         "ok",
        "clone":          clone_status,
        "files":          len(head_files),
        "bug_keyword":    sum(bug_kw.values()),
        "bug_szz":        sum(bug_szz_labels.values()),
        "szz_fallback":   szz_fallback,
        "szz_secs":       szz_secs,
        "smell_total":    smell_total,
        "smell_missing":  smell_none_cnt,
        "prospector_secs": pros_secs,
    }
    logger.info(
        "  ozet: files=%d bug_kw=%d bug_szz=%d smell=%d (miss=%d)",
        summary["files"], summary["bug_keyword"], summary["bug_szz"],
        summary["smell_total"], summary["smell_missing"],
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F2 pilot runner")
    parser.add_argument("--limit", type=int, default=5, help="Proje sayisi")
    parser.add_argument("--skip-clone", action="store_true",
                        help="Klonlama atla (mevcut klonlari kullan)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, datefmt=LOG_DATEFMT)
    ensure_runtime_dirs()

    projects = _pick_projects(args.limit)
    logger.info("Pilot: %d proje", len(projects))

    results: list[dict] = []
    for proj in projects:
        try:
            results.append(_process_project(proj, args.skip_clone))
        except KeyboardInterrupt:
            logger.warning("Kullanici iptal etti.")
            return 130
        except Exception as exc:  # Her proje bagimsiz — biri patlarsa devam.
            logger.exception("Proje basarisiz: %s", proj.get("full_name"))
            results.append({"project": proj.get("full_name"), "status": f"error: {exc}"})

    # Ozet tablo
    print()
    print(f"{'Project':42s} {'Files':>6s} {'BugKW':>6s} {'BugSZZ':>7s} {'Smell':>7s} {'Status'}")
    print("-" * 100)
    for r in results:
        if r.get("status") != "ok":
            print(f"{r.get('project',''):42s} {'':>6s} {'':>6s} {'':>7s} {'':>7s} {r.get('status','?')}")
            continue
        print(
            f"{r['project']:42s} {r['files']:>6d} "
            f"{r['bug_keyword']:>6d} {r['bug_szz']:>7d} "
            f"{r['smell_total']:>7d} ok "
            f"(szz {r['szz_secs']}s{' fb' if r['szz_fallback'] else ''}, "
            f"pros {r['prospector_secs']}s, miss {r['smell_missing']})"
        )

    out_path = CHECKPOINT_DIR / "f2_pilot.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Pilot ozeti yazildi: %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
