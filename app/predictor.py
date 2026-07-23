"""
predictor.py — Model yukleme (singleton) ve tahmin.

V2.1: T1 commit standalone task'i kaldirildi. Core required = T2 bug stacking
artifact'lari + feature_names.json. T3 smell lazy olarak kontrol edilir.
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

_bug_base          = None  # RF / LightGBM / XGBoost — model_config.json'dan belirlenir
_bug_ag            = None  # AutoML model (AutoGluon / H2O / TPOT)
_bug_meta_lr       = None
_scaler_bug        = None
_smell_model       = None
_scaler_smell      = None
_feature_names     = None
_project_stats     = None
_model_config      = None  # model_config.json icerigi
_loaded            = False


def _read_model_config() -> dict:
    """
    models/model_config.json oku; yoksa varsayilan RF+AutoGluon dondur.
    Backward compatible: eski artifact'lar (sadece bug_rf_base.joblib) hala calisir.
    """
    config_path = MODELS_DIR / "model_config.json"
    defaults = {
        "stacking_base_bug":   "rf",
        "stacking_base_smell": "rf",
        "stacking_automl":     "autogluon",
        # F5 threshold-opt: CV-mean optimal threshold (cv_summary_*.json'dan).
        # Production stacking modelinin meta-LR cikis dagilimi 0.5 etrafinda
        # dengelenmis degil; 0.5 ile tahmin yaparsak F1 0.611, opt thr ile 0.644.
        # train_final ileride bu degerleri model_config.json'a yazinca override eder.
        "bug_threshold":   0.42,   # cv_summary_20260531_061755 mean=0.420
        "smell_threshold": 0.34,   # cv_summary_20260531_061755 mean=0.340
    }
    if not config_path.exists():
        return defaults
    try:
        return {**defaults, **json.load(open(config_path, encoding="utf-8"))}
    except (json.JSONDecodeError, OSError):
        return defaults


# Flask temel sanity icin gerekli dosyalar dinamik olarak belirlenir.
# _CORE_REQUIRED sabit listesi _load_models() icinde olusturulur.
_CORE_REQUIRED_STATIC = (
    "bug_meta_lr.joblib",
    "scaler_bug.joblib",
    "feature_names.json",
)


def _load_models() -> None:
    """Modelleri tek seferlik yukler (thread-safe, double-checked locking)."""
    global _bug_base, _bug_ag, _bug_meta_lr
    global _scaler_bug, _smell_model, _scaler_smell
    global _feature_names, _project_stats, _model_config
    global _loaded

    if _loaded:
        return

    with _lock:
        if _loaded:
            return

        cfg = _read_model_config()
        _model_config = cfg
        base_bug   = cfg["stacking_base_bug"]
        base_smell = cfg["stacking_base_smell"]
        automl     = cfg["stacking_automl"]

        # Dinamik artifact isimleri
        base_bug_artifact   = f"bug_{base_bug}_base.joblib"
        automl_bug_artifact = f"bug_{automl}_base"  # dir veya .pkl
        # Backward compat: eski 'bug_rf_base.joblib' ismiyle egitilmis modeller
        if not (MODELS_DIR / base_bug_artifact).exists():
            base_bug_artifact = "bug_rf_base.joblib"

        missing = [f for f in _CORE_REQUIRED_STATIC if not (MODELS_DIR / f).exists()]
        if not (MODELS_DIR / base_bug_artifact).exists():
            missing.append(base_bug_artifact)

        # AutoML artifact varligini kontrol et (H2O dir, TPOT pkl, AG dir)
        automl_path = MODELS_DIR / automl_bug_artifact
        if automl != "h2o" and not automl_path.exists():
            # H2O artifact format farkli olabilir; AG ve TPOT icin zorunlu
            missing.append(automl_bug_artifact)

        if missing:
            raise FileNotFoundError(
                f"Missing model files: {', '.join(missing)}\n"
                "Run scripts/train_final.py first."
            )

        _bug_base    = joblib.load(MODELS_DIR / base_bug_artifact)
        _bug_meta_lr = joblib.load(MODELS_DIR / "bug_meta_lr.joblib")
        _scaler_bug  = joblib.load(MODELS_DIR / "scaler_bug.joblib")

        # AutoML model yukle
        if automl == "autogluon":
            from autogluon.tabular import TabularPredictor
            _bug_ag = TabularPredictor.load(
                str(MODELS_DIR / automl_bug_artifact),
                verbosity=0,
                require_py_version_match=False,
            )
        elif automl == "tpot":
            _bug_ag = joblib.load(MODELS_DIR / automl_bug_artifact / "tpot_pipeline.pkl")
        elif automl == "h2o":
            try:
                import h2o  # type: ignore[import-not-found]
                h2o.init(nthreads=-1, verbose=False)
                model_files = list((MODELS_DIR / automl_bug_artifact).glob("*"))
                if model_files:
                    _bug_ag = h2o.load_model(str(model_files[0]))
                else:
                    logger.warning("H2O artifact directory is empty: %s", automl_bug_artifact)
            except ImportError:
                logger.warning("H2O could not be loaded; bug prediction is disabled.")

        with open(MODELS_DIR / "feature_names.json", encoding="utf-8") as f:
            _feature_names = json.load(f)

        # Smell modeli opsiyonel — model_config'deki base_smell ile dinamik isim
        smell_artifact = MODELS_DIR / f"smell_{base_smell}.joblib"
        if not smell_artifact.exists():
            smell_artifact = MODELS_DIR / "smell_rf.joblib"  # eski isim
        scaler_path = MODELS_DIR / "scaler_smell.joblib"
        if smell_artifact.exists() and scaler_path.exists():
            _smell_model  = joblib.load(smell_artifact)
            _scaler_smell = joblib.load(scaler_path)

        # Proje istatistikleri (Flask panelleri icin)
        stats_path = MODELS_DIR / "project_stats.json"
        if stats_path.exists():
            try:
                with open(stats_path, encoding="utf-8") as f:
                    _project_stats = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read project_stats.json: %s", exc)

        _loaded = True


def get_feature_names() -> dict:
    _load_models()
    return _feature_names  # type: ignore[return-value]


def get_project_stats() -> Optional[dict]:
    """Global proje istatistikleri, F6 sonrasi dolu olur."""
    _load_models()
    return _project_stats


def predict_bug(features_df: pd.DataFrame):
    """
    Bug tahmini (Stacking: base_ml + AutoML -> LR meta).
    base_ml ve AutoML tipi model_config.json'dan belirlenir.
    Returns: (predictions ndarray, probabilities ndarray)
    """
    _load_models()
    cols = _feature_names["bug"]  # type: ignore[index]
    X = features_df[cols].values.astype(float)
    X_s = _scaler_bug.transform(X)

    base_proba = _bug_base.predict_proba(X_s)[:, 1]

    # AutoML tahmin — yuklu modelin tipine gore
    automl_type = (_model_config or {}).get("stacking_automl", "autogluon")
    ag_input = pd.DataFrame(X_s, columns=cols)

    if automl_type == "autogluon":
        ag_pred = _bug_ag.predict_proba(ag_input)
        if isinstance(ag_pred, pd.DataFrame):
            ag_proba = ag_pred.iloc[:, -1].values
        else:
            ag_proba = np.array(ag_pred)[:, 1]
    elif automl_type == "h2o":
        import h2o  # type: ignore[import-not-found]
        h_input = h2o.H2OFrame(ag_input)
        preds = _bug_ag.predict(h_input).as_data_frame()
        ag_proba = preds["p1"].values if "p1" in preds.columns else np.zeros(len(X_s))
    elif automl_type == "tpot":
        ag_proba = _bug_ag.predict_proba(X_s)[:, 1]
    else:
        ag_proba = np.zeros(len(X_s))

    meta_features = np.column_stack([base_proba, ag_proba])
    probs = _bug_meta_lr.predict_proba(meta_features)[:, 1]
    # Threshold-opt: meta_lr.predict() varsayilan 0.5 kullanir; CV'de optimal
    # mean 0.42 (production thr). model_config.json'da varsa o, yoksa default.
    thr = float((_model_config or {}).get("bug_threshold", 0.5))
    preds = (probs >= thr).astype("int64")
    return preds, probs


def predict_smell(features_df: pd.DataFrame):
    """
    Smell tahmini (F6 sonrasi). Model yoksa RuntimeError.
    """
    _load_models()
    if _smell_model is None or _scaler_smell is None:
        raise RuntimeError(
            "The smell model has not been trained yet; train it with scripts/train_final.py."
        )
    cols = _feature_names.get("smell") if _feature_names else None  # type: ignore[union-attr]
    if not cols:
        raise RuntimeError("'smell' is not defined in feature_names.json.")
    X = features_df[cols].values.astype(float)
    X_s = _scaler_smell.transform(X)
    probs = _smell_model.predict_proba(X_s)[:, 1]
    # Threshold-opt: CV optimal mean 0.34; varsayilan 0.5 yerine.
    thr = float((_model_config or {}).get("smell_threshold", 0.5))
    preds = (probs >= thr).astype("int64")
    return preds, probs


def predict_proba_calibrated(features_df: pd.DataFrame) -> np.ndarray:
    """
    Kalibre edilmis bug olasıligi — F5 risk score kaynagi.

    Stacking cikisi (RF + AutoGluon -> isotonic-calibrated meta LR)
    zaten kalibredir; bu fonksiyon temiz bir arayuz saglar.
    Bug modeli yuklu degilse RuntimeError firlatir.

    Returns:
        1-D ndarray, her satirda [0, 1] bug olasıligi.
    """
    _, probs = predict_bug(features_df)
    return probs


def smell_available() -> bool:
    """Smell modeli models/ altinda mevcut mu?"""
    cfg    = _read_model_config()
    base   = cfg.get("stacking_base_smell", "rf")
    paths  = [
        MODELS_DIR / f"smell_{base}.joblib",
        MODELS_DIR / "smell_rf.joblib",  # backward compat
    ]
    return any(p.exists() for p in paths) and (MODELS_DIR / "scaler_smell.joblib").exists()


def models_ready() -> bool:
    """T2 bug icin tum artifact'lar mevcut mu? (smell opsiyonel, T1 commit kaldirildi)"""
    cfg       = _read_model_config()
    base_bug  = cfg.get("stacking_base_bug",  "rf")
    automl    = cfg.get("stacking_automl",    "autogluon")
    candidates = [
        MODELS_DIR / f"bug_{base_bug}_base.joblib",
        MODELS_DIR / "bug_rf_base.joblib",  # backward compat
    ]
    has_base  = any(p.exists() for p in candidates)
    has_automl = (MODELS_DIR / f"bug_{automl}_base").exists()
    required = [MODELS_DIR / f for f in _CORE_REQUIRED_STATIC]
    return has_base and has_automl and all(p.exists() for p in required)
