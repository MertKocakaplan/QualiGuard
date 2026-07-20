"""
backfill_discovery_meta.py — Eski parquet'lerin proje isimlerini alip
GitHub API'sinden topics + description doldurarak discovery.json olusturur.

Kullanim:
    python -m scripts.backfill_discovery_meta

Ne ise yarar:
    Smoke test'te discovery topics+description'siz uretildi → kategoriler %85
    "Diger" cikti. Yeniden discovery cektigimizde proje listesi degisti
    (GitHub gunluk yeniliyor). Bu script mevcut output/projects/ altindaki
    parquet'lerin proje isimlerini kullanip ayni projeler icin enrichment
    yapar — sifirdan re-process gerekmeden.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from pipeline.config import (
    CHECKPOINT_DIR,
    GITHUB_REPO_URL,
    PROJECTS_DIR,
)
from pipeline.rate_limit import current_quota, guarded_get, refresh_quota

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("backfill")


def collect_project_names() -> list[str]:
    """output/projects/ altindaki parquet'lerden benzersiz proje listesi."""
    if not PROJECTS_DIR.exists():
        logger.error("PROJECTS_DIR yok: %s", PROJECTS_DIR)
        return []
    names: set[str] = set()
    for path in sorted(PROJECTS_DIR.glob("*.parquet")):
        try:
            df = pd.read_parquet(path, columns=["project_name"])
            names.update(df["project_name"].dropna().unique())
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("parquet okunamadi: %s (%s)", path.name, exc)
    return sorted(names)


def fetch_repo_meta(full_name: str) -> dict:
    """GET /repos/{full_name} ile topics + description + diger meta."""
    resp = guarded_get(f"{GITHUB_REPO_URL}/{full_name}")
    if resp.status_code != 200:
        logger.warning("404/hata: %s -> HTTP %d", full_name, resp.status_code)
        return {}
    try:
        data = resp.json()
    except ValueError:
        return {}
    return data


def build_discovery_record(meta: dict, full_name: str) -> dict:
    """GitHub API yanitindan discovery.json icin tek kayit uret."""
    topics = meta.get("topics") or []
    return {
        "full_name":         full_name,
        "clone_url":         meta.get("clone_url") or f"https://github.com/{full_name}.git",
        "stars":             int(meta.get("stargazers_count", 0)),
        "created_at":        meta.get("created_at", "") or "",
        "project_age_days":  365,  # Cogunlukla parquet'te zaten dogru deger var
        "contributor_count": 1,    # Bilinmiyor, ML feature degil
        "default_branch":    meta.get("default_branch") or "main",
        "topics":            [str(t) for t in topics if t],
        "description":       (meta.get("description") or "").strip(),
    }


def main() -> int:
    refresh_quota()
    logger.info("GitHub quota: %s", current_quota())

    names = collect_project_names()
    if not names:
        logger.error("Hicbir proje bulunamadi. Once 'collect --phase process' calistir.")
        return 1
    logger.info("Toplam %d proje bulundu, GitHub API'den meta cekiliyor...", len(names))

    found: list[dict] = []
    for i, name in enumerate(names, 1):
        meta = fetch_repo_meta(name)
        if not meta:
            continue
        found.append(build_discovery_record(meta, name))
        if i % 10 == 0 or i == len(names):
            logger.info("  ilerleme: %d/%d (topics dolu: %d, desc dolu: %d)",
                        i, len(names),
                        sum(1 for r in found if r["topics"]),
                        sum(1 for r in found if r["description"]))

    out_path = CHECKPOINT_DIR / "discovery.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "started_at":   "",
        "completed_at": "",
        "criteria":     {"backfill": True},
        "target_count": len(found),
        "found_count":  len(found),
        "found":        found,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("discovery.json yazildi: %s (%d proje)", out_path, len(found))

    n_topics = sum(1 for r in found if r["topics"])
    n_desc   = sum(1 for r in found if r["description"])
    logger.info("Sonuc: topics dolu %d/%d, description dolu %d/%d",
                n_topics, len(found), n_desc, len(found))

    refresh_quota()
    logger.info("Sonrasi quota: %s", current_quota())
    return 0


if __name__ == "__main__":
    sys.exit(main())
