"""
model_utils.py — Split, scaler, SMOTE ve metric yardimcilari.

PLAN §3.11 + §4.3 F5 kapsami: ablation harness'i icin kullanilir.
Hafif yardimcilar — sklearn + numpy/pandas disinda zorunlu bagimlilik yok.
SMOTE opsiyoneldir; `imbalanced-learn` kurulu degilse `apply_smote_train_only`
`ImportError` firlatir.
"""
from __future__ import annotations

import logging
from typing import Iterable, NamedTuple, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from pipeline.config import FEATURES_BUG, FEATURES_COMMIT, FEATURES_PROCESS

logger = logging.getLogger(__name__)


# ── Feature set secimi (ablation) ────────────────────────────────

# Radon raw + cognitive (22 + 2 = 24)
STATIC_FEATURES: tuple[str, ...] = (
    "loc", "lloc", "sloc", "comments", "multi", "blank", "single_comments",
    "cc_mean", "cc_max", "cc_total", "num_functions",
    "h_vocabulary", "h_length", "h_volume", "h_difficulty",
    "h_effort", "h_bugs", "h_time", "h_calculated_length",
    "maintainability_index",
    "comment_ratio", "doc_ratio",
    "cognitive_complexity_total", "cognitive_complexity_max",  # F3.1
)

# Derived (+4)
DERIVED_FEATURES: tuple[str, ...] = (
    "complexity_density", "comment_per_function",
    "avg_function_length", "effort_per_line",
)

# Proje meta + repo-history proxies (+3 +4 = +7)  F3.5
PROJECT_META_FEATURES: tuple[str, ...] = (
    "stars", "contributor_count", "project_age_days",
    "revert_count", "inter_commit_time_cv", "author_entropy", "bug_fix_density",
)

# Process metrics (bug_count haric, data leakage riski)
PROCESS_FEATURES_SAFE: tuple[str, ...] = tuple(
    f for f in FEATURES_PROCESS if f != "bug_count"
)


def get_feature_set(task: str, variant: str) -> tuple[str, ...]:
    """
    Ablation icin feature seti secimi.

    Args:
        task: "commit" | "bug" | "smell"
        variant: "static" | "derived" | "process" | "all"

    Returns:
        Sutun isimlerinin tuple'i. Variant "all" icin task'a gore
        `FEATURES_COMMIT` (35) ya da `FEATURES_BUG` (48) doner.
    """
    task = task.lower()
    variant = variant.lower()
    if task not in ("commit", "bug", "smell"):
        raise ValueError(f"Gecersiz task: {task}")
    if variant == "static":
        return STATIC_FEATURES
    if variant == "derived":
        return STATIC_FEATURES + DERIVED_FEATURES
    if variant == "process":
        # Static + Derived + Proje meta (T1 'all' ile esdeger)
        return STATIC_FEATURES + DERIVED_FEATURES + PROJECT_META_FEATURES
    if variant == "all":
        return FEATURES_COMMIT if task == "commit" else FEATURES_BUG
    raise ValueError(f"Gecersiz variant: {variant}")


def extract_xy(
    df: pd.DataFrame,
    features: Sequence[str],
    label_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    DataFrame -> (X, y). Eksik sutunlar NaN ile doldurulur, sonra NaN'lar 0.0.
    Caller olmamasi gereken sutunlari `features`'ten cikarir.
    """
    if label_col not in df.columns:
        raise KeyError(f"Etiket sutunu yok: {label_col}")
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Eksik feature sutunu: {missing}")
    X = df[list(features)].to_numpy(dtype="float64", copy=True, na_value=0.0)
    y = df[label_col].to_numpy(dtype="int64", copy=True)
    return X, y


# ── Split'ler ─────────────────────────────────────────────────────

class TwoStageSplit(NamedTuple):
    """
    Project-based 70/15/15 holdout (Tantithamthavorn et al. 2017).

    train_dev: kagit %70 — GroupKFold(5) burada calisir.
    val:       kagit %15 — model secimi sonrasi tek seferlik.
    test:      kagit %15 — final paper tablosu.
    *_pids:    her bolumun proje isimleri dizisi.
    """
    train_dev:  pd.DataFrame
    val:        pd.DataFrame
    test:       pd.DataFrame
    train_pids: np.ndarray
    val_pids:   np.ndarray
    test_pids:  np.ndarray


def two_stage_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    project_col: str = "project_name",
    seed: int = 42,
) -> TwoStageSplit:
    """
    Project-based 70/15/15 holdout — F4 (Tantithamthavorn et al. 2017).

    Projeler project_col'a gore gruplandırilir; ayni projenin tum dosyalari
    tek bir bolume duşer (veri sizdirmaz). Deterministik: seed sabit tutuldukca
    ayni bolunum uretilir.

    Args:
        df:          Ham veri cercevesi. project_col zorunlu.
        val_frac:    Validation icin proje orani (default 0.15).
        test_frac:   Test icin proje orani (default 0.15).
        project_col: Proje kimlik sutunu.
        seed:        Rastgelelik tohumu.

    Returns:
        TwoStageSplit NamedTuple.

    Raises:
        ValueError: project_col yoksa veya train bolumu bos kalirsa.
    """
    if project_col not in df.columns:
        raise ValueError(f"project_col bulunamadi: {project_col!r}")

    projects = df[project_col].unique()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(projects))
    projects = projects[perm]

    n = len(projects)
    n_test = max(1, int(test_frac * n))
    n_val  = max(1, int(val_frac  * n))

    if n_test + n_val >= n:
        raise ValueError(
            f"val_frac={val_frac} + test_frac={test_frac} >= 1.0 — "
            f"train bolumu bos kalir ({n} proje var)"
        )

    test_pids  = projects[:n_test]
    val_pids   = projects[n_test:n_test + n_val]
    train_pids = projects[n_test + n_val:]

    return TwoStageSplit(
        train_dev = df[df[project_col].isin(train_pids)].copy(),
        val       = df[df[project_col].isin(val_pids)].copy(),
        test      = df[df[project_col].isin(test_pids)].copy(),
        train_pids = train_pids,
        val_pids   = val_pids,
        test_pids  = test_pids,
    )


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

    projects = df[project_col].unique().copy()
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


def time_based_split(
    df: pd.DataFrame,
    time_col: str = "created_at",
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Kronolojik 70/15/15 split. `time_col` datetime'a parse edilebilir olmali.

    Tarihe gore sirali: erken -> train, orta -> val, gec -> test.
    Robustness raporlari (project-based'le karsilastirma) icin secondary.

    Returns:
        (train_df, val_df, test_df)
    """
    if time_col not in df.columns:
        raise ValueError(f"Sutun bulunamadi: {time_col}")
    work = df.copy()
    work["_ts"] = pd.to_datetime(work[time_col], utc=True, errors="coerce")
    # NaT'lari en sona at (deterministik davranis)
    work = work.sort_values("_ts", kind="stable", na_position="last").reset_index(drop=True)

    n = len(work)
    n_test = int(round(n * test_size))
    n_val  = int(round(n * val_size))
    n_train = n - n_test - n_val
    if n_train < 0:
        raise ValueError("val_size + test_size 1.0'i asamaz")

    train_df = work.iloc[:n_train].drop(columns="_ts").copy()
    val_df   = work.iloc[n_train:n_train + n_val].drop(columns="_ts").copy()
    test_df  = work.iloc[n_train + n_val:].drop(columns="_ts").copy()
    return train_df, val_df, test_df


def group_kfold_indices(
    df: pd.DataFrame,
    group_col: str = "project_name",
    n_splits: int = 5,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Proje gruplamasiyla GroupKFold index ciftleri."""
    gkf = GroupKFold(n_splits=n_splits)
    return gkf.split(df, groups=df[group_col].values)


# ── Scaler + SMOTE ────────────────────────────────────────────────

def fit_scaler(X: np.ndarray) -> StandardScaler:
    """StandardScaler fit — train icin kullan, val/test icin transform."""
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def apply_smote_train_only(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    SMOTE'u SADECE train fold'unda uygula. val/test verisine dokunma.

    Tek sinifli veya yetersiz ornek durumunda `imblearn` hata firlatir;
    caller bu durumda orjinali kullanmayi tercih edebilir.

    Raises:
        ImportError: `imbalanced-learn` kurulu degilse.
    """
    from imblearn.over_sampling import SMOTE  # type: ignore[import-not-found]

    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res


# ── Metrics ───────────────────────────────────────────────────────

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
    else:
        out["pr_auc"] = float("nan")
    return out


def confusion_quadrants(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, int]:
    """tn/fp/fn/tp — 2-sinif. Cok-sinif icin cagirma."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PR curve: (precision, recall, thresholds)."""
    return precision_recall_curve(y_true, y_proba)


# ── Basit shim'ler ─────────────────────────────────────────────────

# Flat veri icin sklearn'un kendisi en basit yol:
random_split = train_test_split
