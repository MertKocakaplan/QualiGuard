"""
inspect_dataset.py — Dataset doğrulama yardimcisi.

Smoke test/full run sirasinda her fazdan sonra parquet'lerin durumunu
ozetler. Tek komut: `python -m scripts.inspect_dataset`.

Cikti:
    - dataset_full_*.parquet boyutu, proje sayisi, kolon sayisi
    - per-project parquet sayisi (output/projects/)
    - checkpoint durumu (discovery + processed)
    - feature kolonlarinin var/yok ozeti
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from pipeline.config import CHECKPOINT_DIR, OUTPUT_DIR, PROJECTS_DIR


def _format_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def inspect_discovery() -> None:
    """Discovery checkpoint'ini ozetle."""
    path = CHECKPOINT_DIR / "discovery.json"
    print(f"\n-- Discovery ({path.name}) --")
    if not path.exists():
        print("  HENUZ YOK — once `python -m scripts.collect --phase discovery` calistirin")
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    found = data.get("found", [])
    stats = data.get("stats", {})
    print(f"  Toplam proje:    {len(found)}")
    print(f"  Bu run yeni:     {stats.get('this_run_new', '?')}")
    print(f"  Onceki run:      {stats.get('previous', '?')}")
    if found:
        print(f"  Ilk 3 proje:")
        for p in found[:3]:
            print(f"    - {p.get('full_name', '?'):40s} stars={p.get('stars', 0):4d}  contrib={p.get('contributor_count', 0)}")


def inspect_processed() -> None:
    """processed_projects.json'u ozetle."""
    path = CHECKPOINT_DIR / "processed_projects.json"
    print(f"\n-- Process ({path.name}) --")
    if not path.exists():
        print("  HENUZ YOK — once `python -m scripts.collect --phase process` calistirin")
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    processed = data.get("processed", {})

    counts = {"ok": 0, "failed": 0, "empty": 0}
    for entry in processed.values():
        if isinstance(entry, dict):
            counts[entry.get("status", "unknown")] = counts.get(entry.get("status", "unknown"), 0) + 1

    total = len(processed)
    success_rate = (counts["ok"] / total * 100) if total else 0
    print(f"  Toplam islem: {total}")
    for status, n in counts.items():
        print(f"    {status:8s} {n:4d}")
    print(f"  Basari orani: {success_rate:.1f}%")


def inspect_per_project_parquets() -> None:
    """output/projects/ altindaki parquet dosyalari."""
    print(f"\n-- Per-project parquet'ler ({PROJECTS_DIR}) --")
    if not PROJECTS_DIR.exists():
        print("  HENUZ YOK")
        return

    files = sorted(PROJECTS_DIR.glob("*.parquet"))
    if not files:
        print("  Bos")
        return

    total_size = sum(f.stat().st_size for f in files)
    total_rows = 0
    for f in files[:5]:  # ilk 5'in satir sayisini bak
        try:
            n = len(pd.read_parquet(f))
            total_rows += n
        except Exception:
            pass

    print(f"  Dosya sayisi: {len(files)}")
    print(f"  Toplam boyut: {_format_size(total_size)}")
    print(f"  Ilk 5 dosya:")
    for f in files[:5]:
        try:
            df = pd.read_parquet(f)
            print(f"    {f.name:50s} {len(df):6d} satir, {len(df.columns):3d} kolon")
        except Exception as exc:  # noqa: BLE001
            print(f"    {f.name:50s} HATA: {exc}")


def inspect_full_dataset() -> None:
    """dataset_full_*.parquet (build cikisi)."""
    print(f"\n-- Full dataset ({OUTPUT_DIR}) --")
    files = sorted(OUTPUT_DIR.glob("dataset_full_*.parquet"))
    if not files:
        print("  HENUZ YOK — once `python -m scripts.collect --phase build` calistirin")
        return

    latest = files[-1]
    df = pd.read_parquet(latest)
    print(f"  Dosya:       {latest.name}")
    print(f"  Boyut:       {_format_size(latest.stat().st_size)}")
    print(f"  Satir:       {len(df):,}")
    print(f"  Kolon:       {len(df.columns)}")
    if "project_name" in df.columns:
        print(f"  Proje:       {df['project_name'].nunique()}")

    # Etiket dagilimi
    print(f"\n  Etiket dagilimi:")
    for label_col in ("bug_keyword", "bug_szz", "smell_count", "commit_count"):
        if label_col in df.columns:
            s = df[label_col]
            if label_col in ("bug_keyword", "bug_szz"):
                pos = (s == 1).sum()
                pct = (pos / len(df) * 100) if len(df) else 0
                print(f"    {label_col:18s} pozitif: {pos:6d} ({pct:.1f}%)")
            elif label_col == "smell_count":
                non_null = s.notna().sum()
                avg = s.mean() if non_null else 0
                max_v = s.max() if non_null else 0
                print(f"    {label_col:18s} ort: {avg:.1f}, max: {max_v}, dolu: {non_null}")
            else:  # commit_count
                print(f"    {label_col:18s} ort: {s.mean():.1f}, median: {s.median():.0f}, max: {s.max()}")

    # Onemli feature'lar var mi
    print(f"\n  Kritik feature'lar:")
    expected = (
        "loc", "cc_mean", "h_volume", "maintainability_index",
        "cognitive_complexity_total", "churn_total",
        "bug_kw_fix_count", "refactor_ratio", "contribution_gini",
        "revert_count", "inter_commit_time_cv", "author_entropy", "bug_fix_density",
        "smell_count",
    )
    for col in expected:
        mark = "[OK]" if col in df.columns else "[EKSIK]"
        print(f"    {col:32s} {mark}")


def inspect_filtered_dataset() -> None:
    """dataset_model_filtered_*.parquet (analysis cikisi)."""
    print(f"\n-- Filtered dataset ({OUTPUT_DIR}) --")
    files = sorted(OUTPUT_DIR.glob("dataset_model_filtered_*.parquet"))
    if not files:
        print("  HENUZ YOK — analysis/01_filter_categorize.py calistirin")
        return
    latest = files[-1]
    df = pd.read_parquet(latest)
    print(f"  Dosya:       {latest.name}")
    print(f"  Satir:       {len(df):,}")
    print(f"  Kolon:       {len(df.columns)}")
    if "project_name" in df.columns:
        print(f"  Proje:       {df['project_name'].nunique()}")
    if "category_primary" in df.columns:
        print(f"  Kategori dagilimi:")
        for cat, n in df.drop_duplicates("project_name")["category_primary"].value_counts().items():
            print(f"    {cat:18s} {n}")


def main() -> int:
    print("=" * 60)
    print("  QualiGuard — Dataset Durum Raporu")
    print("=" * 60)
    inspect_discovery()
    inspect_processed()
    inspect_per_project_parquets()
    inspect_full_dataset()
    inspect_filtered_dataset()
    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
