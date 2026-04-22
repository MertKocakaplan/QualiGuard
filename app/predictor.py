"""
predictor.py — Model yukleme (singleton) ve tahmin.

V2'de F6 sonunda 3. gorev (smell) eklenir — F1'de T1 (commit) ve T2 (bug)
mevcut artifact'lardan yuklenir; smell lazy olarak kontrol edilir.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from pipeline.config import MODELS_DIR

logger = logging.getLogger(__name__)

_lock = threading.Lock()

_commit_rf         = None
_scaler_commit     = None
_bug_rf            = None
_bug_ag            = None
_bug_meta_lr       = None
_scaler_bug        = None
_smell_model       = None
_scaler_smell      = None
_feature_names     = None
_project_stats     = None
_loaded            = False


# Flask temel sanity icin gerekli dosyalar (commit + bug)
_CORE_REQUIRED = (
    "commit_rf.joblib",
    "scaler_commit.joblib",
    "bug_rf_base.joblib",
    "bug_meta_lr.joblib",
    "scaler_bug.joblib",
    "feature_names.json",
)


def _load_models() -> None:
    """Modelleri tek seferlik yukler (thread-safe, double-checked locking)."""
    global _commit_rf, _scaler_commit, _bug_rf, _bug_ag, _bug_meta_lr
    global _scaler_bug, _smell_model, _scaler_smell, _feature_names, _project_stats
    global _loaded

    if _loaded:
        return

    with _lock:
        if _loaded:
            return

        missing = [f for f in _CORE_REQUIRED if not (MODELS_DIR / f).exists()]
        if not (MODELS_DIR / "bug_ag_base").exists():
            missing.append("bug_ag_base/")

        if missing:
            raise FileNotFoundError(
                f"Eksik model dosyalari: {', '.join(missing)}\n"
                "Once scripts/train_final.py calistirin."
            )

        _commit_rf     = joblib.load(MODELS_DIR / "commit_rf.joblib")
        _scaler_commit = joblib.load(MODELS_DIR / "scaler_commit.joblib")
        _bug_rf        = joblib.load(MODELS_DIR / "bug_rf_base.joblib")
        _bug_meta_lr   = joblib.load(MODELS_DIR / "bug_meta_lr.joblib")
        _scaler_bug    = joblib.load(MODELS_DIR / "scaler_bug.joblib")

        # AutoGluon lazy import — buyuk kutuphane.
        # require_py_version_match=False: egitim farkli py ile yapilmis
        # olabilir (notebook 3.10, runtime 3.13).
        from autogluon.tabular import TabularPredictor
        _bug_ag = TabularPredictor.load(
            str(MODELS_DIR / "bug_ag_base"),
            verbosity=0,
            require_py_version_match=False,
        )

        with open(MODELS_DIR / "feature_names.json", encoding="utf-8") as f:
            _feature_names = json.load(f)

        # Smell modeli opsiyonel (F6 sonrasi)
        smell_path  = MODELS_DIR / "smell_rf.joblib"
        scaler_path = MODELS_DIR / "scaler_smell.joblib"
        if smell_path.exists() and scaler_path.exists():
            _smell_model  = joblib.load(smell_path)
            _scaler_smell = joblib.load(scaler_path)

        # Proje istatistikleri (Flask panelleri icin, F6/F7'de dolar)
        stats_path = MODELS_DIR / "project_stats.json"
        if stats_path.exists():
            try:
                with open(stats_path, encoding="utf-8") as f:
                    _project_stats = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("project_stats.json okunamadi: %s", exc)

        _loaded = True


def get_feature_names() -> dict:
    _load_models()
    return _feature_names  # type: ignore[return-value]


def get_project_stats() -> Optional[dict]:
    """Global proje istatistikleri, F6 sonrasi dolu olur."""
    _load_models()
    return _project_stats


def predict_commit(features_df: pd.DataFrame):
    """
    Commit tahmini (Random Forest).
    Returns: (predictions ndarray, probabilities ndarray)
    """
    _load_models()
    cols = _feature_names["commit"]  # type: ignore[index]
    X = features_df[cols].values.astype(float)
    X_s = _scaler_commit.transform(X)
    preds = _commit_rf.predict(X_s)
    probs = _commit_rf.predict_proba(X_s)[:, 1]
    return preds, probs


def predict_bug(features_df: pd.DataFrame):
    """
    Bug tahmini (Stacking: RF + AutoGluon -> LR meta).
    Returns: (predictions ndarray, probabilities ndarray)
    """
    _load_models()
    cols = _feature_names["bug"]  # type: ignore[index]
    X = features_df[cols].values.astype(float)
    X_s = _scaler_bug.transform(X)

    rf_proba = _bug_rf.predict_proba(X_s)[:, 1]

    ag_input = pd.DataFrame(X_s, columns=cols)
    ag_pred = _bug_ag.predict_proba(ag_input)
    if isinstance(ag_pred, pd.DataFrame):
        if 1 in ag_pred.columns:
            ag_proba = ag_pred[1].values
        else:
            ag_proba = ag_pred.iloc[:, -1].values
    else:
        ag_proba = np.array(ag_pred)[:, 1]

    meta_features = np.column_stack([rf_proba, ag_proba])
    preds = _bug_meta_lr.predict(meta_features)
    probs = _bug_meta_lr.predict_proba(meta_features)[:, 1]
    return preds, probs


def predict_smell(features_df: pd.DataFrame):
    """
    Smell tahmini (F6 sonrasi). Model yoksa RuntimeError.
    """
    _load_models()
    if _smell_model is None or _scaler_smell is None:
        raise RuntimeError(
            "Smell modeli henuz egitilmemis. scripts/train_final.py ile egitim gerekli."
        )
    cols = _feature_names.get("smell") if _feature_names else None  # type: ignore[union-attr]
    if not cols:
        raise RuntimeError("feature_names.json icinde 'smell' tanimli degil.")
    X = features_df[cols].values.astype(float)
    X_s = _scaler_smell.transform(X)
    preds = _smell_model.predict(X_s)
    probs = _smell_model.predict_proba(X_s)[:, 1]
    return preds, probs


def smell_available() -> bool:
    """Smell modeli models/ altinda mevcut mu?"""
    return (MODELS_DIR / "smell_rf.joblib").exists() and \
           (MODELS_DIR / "scaler_smell.joblib").exists()


def models_ready() -> bool:
    """Commit+Bug icin tum artifact'lar mevcut mu? (smell opsiyonel)"""
    required: list[Path] = [MODELS_DIR / f for f in _CORE_REQUIRED]
    required.append(MODELS_DIR / "bug_ag_base")
    return all(p.exists() for p in required)
