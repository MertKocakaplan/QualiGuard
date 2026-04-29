"""
dataset_builder.py — Per-project parquet'leri birlestirip final dataset'i
uret.

F1 kapsaminda iskelet; F3'te per-project akis uretimi; F4'te kategori
atama + sensitivity filtresi burada toplanir.

PLAN §3.11 ve §14.1/14.2 seması uygulanir.

Kullanim:
    from pipeline.dataset_builder import build_full_dataset
    out_path = build_full_dataset()  # output/dataset_full_<ts>.parquet
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd

from pipeline.categories import OTHER_CATEGORY, assign_categories, primary_category
from pipeline.config import OUTPUT_DIR, PROJECTS_DIR, SMELL_BINARY_PERCENTILE

logger = logging.getLogger(__name__)


def list_project_files() -> list[Path]:
    """output/projects/ altindaki tum .parquet dosyalari."""
    if not PROJECTS_DIR.exists():
        return []
    return sorted(PROJECTS_DIR.glob("*.parquet"))


def load_project_parquets(files: Optional[list[Path]] = None) -> pd.DataFrame:
    """Tum per-project parquet'leri tek DataFrame'e birlestir."""
    if files is None:
        files = list_project_files()
    if not files:
        return pd.DataFrame()
    frames = []
    for path in files:
        try:
            frames.append(pd.read_parquet(path))
        except (OSError, ValueError) as exc:
            logger.warning("parquet okunamadi: %s (%s)", path.name, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def add_dynamic_smell_binary(
    df: pd.DataFrame,
    percentile: int = SMELL_BINARY_PERCENTILE,
) -> pd.DataFrame:
    """
    Her proje icin smell_count dagiliminin P{percentile} esiginden buyuk/esit
    olan dosyalara smell_binary=1 ata.
    """
    if df.empty or "smell_count" not in df.columns or "project_name" not in df.columns:
        df["smell_binary"] = pd.Series(0, index=df.index, dtype="int8")
        return df

    # Tum satirlar NA ise (--skip-prospector durumu) hicbir esik hesaplanamaz —
    # smell_binary=0 ile cik, T3 anlamsizlasir ama pipeline crash etmez.
    if df["smell_count"].isna().all():
        df["smell_binary"] = pd.Series(0, index=df.index, dtype="int8")
        return df

    thresholds = df.groupby("project_name")["smell_count"].transform(
        lambda s: s.dropna().quantile(percentile / 100.0) if s.notna().any() else float("nan")
    )
    # Bool maskede NA olabilir (esik NaN olan projelerde) — once False ile doldur.
    mask = (df["smell_count"].fillna(-1) >= thresholds).fillna(False)
    df["smell_binary"] = mask.astype("int8")
    return df


def add_commit_label(df: pd.DataFrame) -> pd.DataFrame:
    """label_commit = commit_count >= global median(commit_count)."""
    if df.empty or "commit_count" not in df.columns:
        df["label_commit"] = 0
        return df
    median = float(df["commit_count"].median())
    df["label_commit"] = (df["commit_count"] >= median).astype("int8")
    return df


def add_project_categories(
    df: pd.DataFrame,
    project_meta: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> pd.DataFrame:
    """
    Her satira projesinin `category_primary` + `categories_all` sutunlarini ekle.

    Proje icin arama metni:
        - project_meta[name]["topics"]  (iterable[str])
        - project_meta[name]["description"]  (str)
        - project_name  (her zaman)

    project_meta None veya eksikse yalnizca project_name kullanilir —
    bu durumda cogu proje `"Diger"` dusebilir, sensitivity analizinde
    bu goz ardi edilebilir (filtresiz sonuclara ek destek).
    """
    if df.empty or "project_name" not in df.columns:
        df["category_primary"] = OTHER_CATEGORY
        df["categories_all"]   = OTHER_CATEGORY
        return df

    meta = project_meta or {}
    cache: dict[str, list[str]] = {}
    for name in df["project_name"].dropna().unique():
        entry   = meta.get(name, {}) if isinstance(meta, Mapping) else {}
        topics  = entry.get("topics", ()) if isinstance(entry, Mapping) else ()
        if not isinstance(topics, Iterable) or isinstance(topics, (str, bytes)):
            topics = ()
        desc    = entry.get("description", "") if isinstance(entry, Mapping) else ""
        cache[name] = assign_categories(
            full_name=name,
            topics=[str(t) for t in topics],
            description=str(desc or ""),
        )

    df["categories_all"]   = df["project_name"].map(
        lambda n: ",".join(cache.get(n, [OTHER_CATEGORY]))
    ).astype("string")
    df["category_primary"] = df["project_name"].map(
        lambda n: primary_category(cache.get(n, [OTHER_CATEGORY]))
    ).astype("string")
    return df


def apply_commit_filter(
    df: pd.DataFrame,
    min_commits: Optional[int] = None,
    max_commits: Optional[int] = None,
) -> pd.DataFrame:
    """
    Dosya seviyesinde `commit_count` araligina gore filtrele.

    None sinir dokunulmaz birakilir. F4 sensitivity analizi bu fonksiyonu
    uc sekilde cagirir: (None, None) = filtresiz, (10, 100), (25, 80).

    Orijinal DataFrame'e dokunmaz, yeni bir kopya dondurur.
    """
    if df.empty or "commit_count" not in df.columns:
        return df.copy()

    mask = pd.Series(True, index=df.index)
    if min_commits is not None:
        mask &= df["commit_count"] >= int(min_commits)
    if max_commits is not None:
        mask &= df["commit_count"] <= int(max_commits)
    return df.loc[mask].copy()


def sensitivity_summary(
    df: pd.DataFrame,
    filters: Iterable[tuple[Optional[int], Optional[int]]] = (
        (None, None), (10, 100), (25, 80),
    ),
) -> pd.DataFrame:
    """
    Uc filtre senaryosu icin ozet tablo: satir/proje sayisi, pozitif sinif
    orani, smell oran. F4 interactive hucresinde plot + CSV export icin.

    Kaynak df'i degistirmez; hem `label_commit` hem `smell_binary`
    yoksa o sutunu gormezden gelir.
    """
    rows: list[dict] = []
    for (lo, hi) in filters:
        sub = apply_commit_filter(df, lo, hi)
        label = (
            float(sub["label_commit"].mean())
            if "label_commit" in sub.columns and len(sub) else float("nan")
        )
        smell = (
            float(sub["smell_binary"].mean())
            if "smell_binary" in sub.columns and len(sub) else float("nan")
        )
        rows.append({
            "min_commits":    lo,
            "max_commits":    hi,
            "files":          int(len(sub)),
            "projects":       int(sub["project_name"].nunique())
                               if "project_name" in sub.columns else 0,
            "pct_label_pos":  round(label * 100.0, 2) if label == label else float("nan"),
            "pct_smell_pos":  round(smell * 100.0, 2) if smell == smell else float("nan"),
        })
    return pd.DataFrame(rows)


def build_full_dataset(
    output_dir: Path = OUTPUT_DIR,
    timestamp: Optional[str] = None,
) -> Optional[Path]:
    """
    Tum per-project parquet'lerini birlestir, label sutunlarini ekle,
    `dataset_full_<ts>.parquet` olarak yaz.

    Returns:
        Yazilan dosyanin Path'i; kaynak bos ise None.
    """
    df = load_project_parquets()
    if df.empty:
        logger.warning("dataset_builder: birlestirilecek parquet bulunamadi")
        return None

    df = add_dynamic_smell_binary(df)
    df = add_commit_label(df)

    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"dataset_full_{ts}.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("dataset_full yazildi: %s (%d satir)", out_path.name, len(df))
    return out_path
