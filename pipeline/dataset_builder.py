"""
dataset_builder.py — Per-project parquet'leri birlestirip final dataset'i
uret.

F1 kapsaminda iskelet seklinde; F3'te veri toplamayla senkron calisir.
PLAN §3.11 ve §14.1/14.2 seması uygulanir.

Kullanim:
    from pipeline.dataset_builder import build_full_dataset
    out_path = build_full_dataset()  # output/dataset_full_<ts>.parquet
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

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
        df["smell_binary"] = 0
        return df

    thresholds = df.groupby("project_name")["smell_count"].transform(
        lambda s: s.dropna().quantile(percentile / 100.0) if s.notna().any() else float("nan")
    )
    df["smell_binary"] = (df["smell_count"].fillna(-1) >= thresholds).astype("int8")
    return df


def add_commit_label(df: pd.DataFrame) -> pd.DataFrame:
    """label_commit = commit_count >= global median(commit_count)."""
    if df.empty or "commit_count" not in df.columns:
        df["label_commit"] = 0
        return df
    median = float(df["commit_count"].median())
    df["label_commit"] = (df["commit_count"] >= median).astype("int8")
    return df


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
