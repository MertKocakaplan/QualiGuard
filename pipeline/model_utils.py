"""
model_utils.py — Split, scaler ve metric yardimcilari.

F1 iskeleti — F5/F6'da genisletilir. Sadece temel imzalari yaziyoruz
ki `from pipeline.model_utils import ...` calissin.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def project_based_split(
    df: pd.DataFrame,
    project_col: str = "project_name",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Proje bazli 70/15/15 split. Ayni projenin dosyalari tek split'te kalir.

    Returns:
        (train_df, val_df, test_df)
    """
    if project_col not in df.columns:
        raise ValueError(f"Sutun bulunamadi: {project_col}")

    projects = df[project_col].unique()
    rng = np.random.default_rng(random_state)
    rng.shuffle(projects)

    n = len(projects)
    n_test = int(round(n * test_size))
    n_val  = int(round(n * val_size))
    test_projs  = set(projects[:n_test])
    val_projs   = set(projects[n_test:n_test + n_val])
    train_projs = set(projects[n_test + n_val:])

    train_df = df[df[project_col].isin(train_projs)].copy()
    val_df   = df[df[project_col].isin(val_projs)].copy()
    test_df  = df[df[project_col].isin(test_projs)].copy()
    return train_df, val_df, test_df


def group_kfold_indices(
    df: pd.DataFrame,
    group_col: str = "project_name",
    n_splits: int = 5,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Proje gruplamasiyla GroupKFold index ciftleri."""
    gkf = GroupKFold(n_splits=n_splits)
    return gkf.split(df, groups=df[group_col].values)


def fit_scaler(X: np.ndarray) -> StandardScaler:
    """StandardScaler fit — train icin kullan, val/test icin transform."""
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """F1, accuracy, MCC; olasilik varsa PR-AUC."""
    out = {
        "f1":       float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mcc":      float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_proba is not None:
        try:
            out["pr_auc"] = float(average_precision_score(y_true, y_proba))
        except ValueError:
            out["pr_auc"] = float("nan")
    return out


# Plain shim — sklearn'un train_test_split'i tek cagri ile cozer.
# Ama caller flat veri kullanirsa bu daha kolay:
random_split = train_test_split
