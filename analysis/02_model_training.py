# %% [markdown]
# # 02 - Model Training (Ablation Matrix)
#
# **Faz:** F5 - Model training (PLAN §4.3, §15 F5 DoD)
#
# Ablation boyutlari (PLAN §4.3):
#
#   - Gorev: T1 commit / T2 bug / T3 smell
#   - Label varyanti (T2): keyword / szz
#   - Feature seti: static / derived / process / all
#   - Split: project-based (primary) / time-based (secondary)
#   - Model: LR, RF, SVM, XGBoost, LightGBM, AutoGluon, MLP, CNN1D, LSTM, Stacking
#
# **Agir bagimliliklar opsiyoneldir.** xgboost, lightgbm, autogluon ve
# keras/tensorflow kurulu degilse ilgili model satiri `status=skipped`
# olarak raporlanir, diger modeller calismaya devam eder.
#
# **Smart pruning (PLAN §4.3):** tam carpim pahalidir. Default konfigurasyon
# "all" feature seti + "project" split uzerinde tum modelleri dener; kucuk
# varyantlari (static/derived/process) yalnizca RF + AutoGluon + Stacking
# uzerinde calistirir. `ABLATION["feature_sets"]` veya `ABLATION["models"]`
# satirini degistirerek genisletilebilir.
#
# Calistirmadan once: `analysis/01_filter_categorize.py` ile
# `output/dataset_model_filtered_*.parquet` uretilmis olmali.

# %% Hucre 1 - Imports
from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from pipeline.config import OUTPUT_DIR
from pipeline.model_utils import (
    apply_smote_train_only,
    classification_metrics,
    extract_xy,
    fit_scaler,
    get_feature_set,
    project_based_split,
    time_based_split,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("analysis.02")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
AUTOGLUON_TIME_LIMIT = 60  # ablation icin kisa; final training'de artirilabilir

# %% Hucre 2 - En guncel filtered parquet'i yukle
filtered = sorted(OUTPUT_DIR.glob("dataset_model_filtered_*.parquet"))
if not filtered:
    raise FileNotFoundError(
        "dataset_model_filtered_*.parquet bulunamadi. "
        "Once `analysis/01_filter_categorize.py` hucrelerini calistirin."
    )
DATA_PATH = filtered[-1]
df = pd.read_parquet(DATA_PATH)
print(f"Kaynak : {DATA_PATH.name}")
print(f"Kayit  : {len(df):,} dosya")
print(f"Proje  : {df['project_name'].nunique() if 'project_name' in df else 0}")

# %% Hucre 3 - Ablation konfigurasyonu
# Her bir kombinasyon bir satir uretir. Gerekirse bu sozlugu duzenleyin.
ABLATION: dict = {
    # Gorev + kullanilacak etiket sutunu
    "tasks": [
        {"name": "commit", "label_col": "label_commit", "label_variant": "median"},
        {"name": "bug",    "label_col": "bug_keyword",  "label_variant": "keyword"},
        {"name": "bug",    "label_col": "bug_szz",      "label_variant": "szz"},
        {"name": "smell",  "label_col": "smell_binary", "label_variant": "p80"},
    ],
    # Feature setleri (PLAN §4.3 cumulative)
    "feature_sets": ["static", "derived", "process", "all"],
    # Split'ler
    "splits": ["project", "time"],
    # Modeller (agir olanlar opsiyonel bagimliliklarla calisir)
    "models": [
        "lr", "rf", "svm",
        "xgboost", "lightgbm",
        "autogluon",
        "mlp", "cnn1d", "lstm",
        "stacking_rf_ag_meta_lr",
    ],
    # Smart pruning — tam matris yerine "all" feature set'te tum modeller,
    # kucuk varyantlarda yalnizca RF + AutoGluon + Stacking.
    "prune_models_on_small_feature_sets": True,
    "small_feature_model_subset": ["rf", "autogluon", "stacking_rf_ag_meta_lr"],
    # SMOTE train-only (azinlik sinifini dengeler)
    "use_smote": True,
}
print(f"Toplam kombinasyon (kaba): "
      f"{len(ABLATION['tasks']) * len(ABLATION['feature_sets']) "
      f"* len(ABLATION['splits']) * len(ABLATION['models'])}")

# %% Hucre 4 - Model fabrikalari (opsiyonel bagimliliklar try/except ile)

@dataclass
class ModelRun:
    """Tek bir model kosusundan donen sonuc kabi."""
    y_pred:    Optional[np.ndarray] = None
    y_proba:   Optional[np.ndarray] = None
    duration:  float                 = 0.0
    status:    str                   = "ok"  # ok | skipped | failed
    error:     str                   = ""
    notes:     dict                  = field(default_factory=dict)


ModelFn = Callable[[np.ndarray, np.ndarray, np.ndarray], ModelRun]


def _time_it(fn: Callable[[], ModelRun]) -> ModelRun:
    """Wrapper: fn calistir, duration olc."""
    t0 = time.monotonic()
    run = fn()
    run.duration = round(time.monotonic() - t0, 2)
    return run


def _fit_predict_sklearn(clf, Xtr, ytr, Xte) -> ModelRun:
    """Ortak sklearn wrapper."""
    clf.fit(Xtr, ytr)
    y_pred = clf.predict(Xte)
    y_proba: Optional[np.ndarray] = None
    if hasattr(clf, "predict_proba"):
        try:
            y_proba = clf.predict_proba(Xte)[:, 1]
        except (ValueError, IndexError):
            y_proba = None
    elif hasattr(clf, "decision_function"):
        try:
            y_proba = clf.decision_function(Xte)
        except ValueError:
            y_proba = None
    return ModelRun(y_pred=np.asarray(y_pred), y_proba=y_proba)


def model_lr(Xtr, ytr, Xte) -> ModelRun:
    return _time_it(lambda: _fit_predict_sklearn(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        Xtr, ytr, Xte,
    ))


def model_rf(Xtr, ytr, Xte) -> ModelRun:
    return _time_it(lambda: _fit_predict_sklearn(
        RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE,
        ),
        Xtr, ytr, Xte,
    ))


def model_svm(Xtr, ytr, Xte) -> ModelRun:
    return _time_it(lambda: _fit_predict_sklearn(
        SVC(probability=True, random_state=RANDOM_STATE),
        Xtr, ytr, Xte,
    ))


def model_xgboost(Xtr, ytr, Xte) -> ModelRun:
    try:
        from xgboost import XGBClassifier  # type: ignore[import-not-found]
    except ImportError as exc:
        return ModelRun(status="skipped", error=f"xgboost yok: {exc}")
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, use_label_encoder=False,
        eval_metric="logloss", random_state=RANDOM_STATE,
        tree_method="hist", n_jobs=-1,
    )
    return _time_it(lambda: _fit_predict_sklearn(clf, Xtr, ytr, Xte))


def model_lightgbm(Xtr, ytr, Xte) -> ModelRun:
    try:
        from lightgbm import LGBMClassifier  # type: ignore[import-not-found]
    except ImportError as exc:
        return ModelRun(status="skipped", error=f"lightgbm yok: {exc}")
    clf = LGBMClassifier(
        n_estimators=300, num_leaves=31, random_state=RANDOM_STATE,
        n_jobs=-1, verbose=-1,
    )
    return _time_it(lambda: _fit_predict_sklearn(clf, Xtr, ytr, Xte))


def model_autogluon(Xtr, ytr, Xte) -> ModelRun:
    try:
        from autogluon.tabular import TabularPredictor  # type: ignore[import-not-found]
    except ImportError as exc:
        return ModelRun(status="skipped", error=f"autogluon yok: {exc}")
    import tempfile

    def _run():
        feat_cols = [f"f{i}" for i in range(Xtr.shape[1])]
        train_df = pd.DataFrame(Xtr, columns=feat_cols)
        train_df["target"] = ytr
        test_df  = pd.DataFrame(Xte,  columns=feat_cols)
        with tempfile.TemporaryDirectory() as tmp:
            pred = TabularPredictor(
                label="target", path=tmp, problem_type="binary", verbosity=0,
            ).fit(train_df, time_limit=AUTOGLUON_TIME_LIMIT)
            y_pred  = pred.predict(test_df).to_numpy()
            try:
                y_proba = pred.predict_proba(test_df).iloc[:, 1].to_numpy()
            except (ValueError, IndexError):
                y_proba = None
        return ModelRun(y_pred=y_pred, y_proba=y_proba,
                        notes={"time_limit": AUTOGLUON_TIME_LIMIT})
    return _time_it(_run)


def _keras_dense(input_dim: int):
    """Basit MLP (keras). Cagri oncesi keras importorskip gibi davranilir."""
    from tensorflow import keras  # type: ignore[import-not-found]
    m = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1,  activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m


def _keras_cnn1d(input_dim: int):
    from tensorflow import keras  # type: ignore[import-not-found]
    m = keras.Sequential([
        keras.layers.Input(shape=(input_dim, 1)),
        keras.layers.Conv1D(32, 3, activation="relu", padding="same"),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1,  activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m


def _keras_lstm(input_dim: int):
    from tensorflow import keras  # type: ignore[import-not-found]
    m = keras.Sequential([
        keras.layers.Input(shape=(input_dim, 1)),
        keras.layers.LSTM(32),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1,  activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m


def _keras_run(builder, Xtr, ytr, Xte, reshape_3d: bool) -> ModelRun:
    try:
        from tensorflow import keras  # noqa: F401 — import check
    except ImportError as exc:
        return ModelRun(status="skipped", error=f"tensorflow/keras yok: {exc}")

    def _run():
        Xtr_in = Xtr[..., None] if reshape_3d else Xtr
        Xte_in = Xte[..., None] if reshape_3d else Xte
        m = builder(Xtr.shape[1])
        m.fit(Xtr_in, ytr, epochs=10, batch_size=64, verbose=0)
        y_proba = m.predict(Xte_in, verbose=0).ravel()
        y_pred = (y_proba >= 0.5).astype("int64")
        return ModelRun(y_pred=y_pred, y_proba=y_proba)
    return _time_it(_run)


def model_mlp(Xtr, ytr, Xte) -> ModelRun:
    return _keras_run(_keras_dense, Xtr, ytr, Xte, reshape_3d=False)


def model_cnn1d(Xtr, ytr, Xte) -> ModelRun:
    return _keras_run(_keras_cnn1d, Xtr, ytr, Xte, reshape_3d=True)


def model_lstm(Xtr, ytr, Xte) -> ModelRun:
    return _keras_run(_keras_lstm, Xtr, ytr, Xte, reshape_3d=True)


def model_stacking_rf_ag_meta_lr(Xtr, ytr, Xte) -> ModelRun:
    """RF + AutoGluon base, LR meta-learner. V1 hibrit pattern'inden uyarlama."""
    try:
        from autogluon.tabular import TabularPredictor  # type: ignore[import-not-found]
    except ImportError as exc:
        return ModelRun(status="skipped", error=f"autogluon yok: {exc}")
    import tempfile
    from sklearn.model_selection import KFold

    def _run():
        kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        rf_oof = np.zeros(len(Xtr))
        ag_oof = np.zeros(len(Xtr))
        feat_cols = [f"f{i}" for i in range(Xtr.shape[1])]
        for tr_idx, va_idx in kf.split(Xtr):
            rf = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                        random_state=RANDOM_STATE)
            rf.fit(Xtr[tr_idx], ytr[tr_idx])
            rf_oof[va_idx] = rf.predict_proba(Xtr[va_idx])[:, 1]

            tr_df = pd.DataFrame(Xtr[tr_idx], columns=feat_cols)
            tr_df["target"] = ytr[tr_idx]
            va_df = pd.DataFrame(Xtr[va_idx], columns=feat_cols)
            with tempfile.TemporaryDirectory() as tmp:
                p = TabularPredictor(
                    label="target", path=tmp, problem_type="binary", verbosity=0,
                ).fit(tr_df, time_limit=AUTOGLUON_TIME_LIMIT)
                ag_oof[va_idx] = p.predict_proba(va_df).iloc[:, 1].to_numpy()

        # Meta-learner: LR on OOF predictions
        meta = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        meta.fit(np.c_[rf_oof, ag_oof], ytr)

        # Base modelleri tum train uzerine yeniden fit + test
        rf_full = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                          random_state=RANDOM_STATE)
        rf_full.fit(Xtr, ytr)
        rf_test = rf_full.predict_proba(Xte)[:, 1]

        tr_df_full = pd.DataFrame(Xtr, columns=feat_cols)
        tr_df_full["target"] = ytr
        te_df = pd.DataFrame(Xte, columns=feat_cols)
        with tempfile.TemporaryDirectory() as tmp:
            p_full = TabularPredictor(
                label="target", path=tmp, problem_type="binary", verbosity=0,
            ).fit(tr_df_full, time_limit=AUTOGLUON_TIME_LIMIT)
            ag_test = p_full.predict_proba(te_df).iloc[:, 1].to_numpy()

        y_proba = meta.predict_proba(np.c_[rf_test, ag_test])[:, 1]
        y_pred = (y_proba >= 0.5).astype("int64")
        return ModelRun(y_pred=y_pred, y_proba=y_proba,
                        notes={"base": ["rf", "autogluon"], "meta": "lr"})
    return _time_it(_run)


MODEL_REGISTRY: dict[str, ModelFn] = {
    "lr":                      model_lr,
    "rf":                      model_rf,
    "svm":                     model_svm,
    "xgboost":                 model_xgboost,
    "lightgbm":                model_lightgbm,
    "autogluon":               model_autogluon,
    "mlp":                     model_mlp,
    "cnn1d":                   model_cnn1d,
    "lstm":                    model_lstm,
    "stacking_rf_ag_meta_lr":  model_stacking_rf_ag_meta_lr,
}

# %% Hucre 5 - Tek bir kombinasyonu calistiran yardimci
def run_one_combo(
    df: pd.DataFrame,
    task: str,
    label_col: str,
    label_variant: str,
    feature_set: str,
    split: str,
    model_name: str,
    use_smote: bool = True,
) -> dict:
    """
    Tek (task, label, feature, split, model) kombinasyonunu calistir, metric
    sozlugu dondur. Hata durumunda status=failed/skipped + error doldurulur.
    """
    row = {
        "task":           task,
        "label_variant":  label_variant,
        "feature_set":    feature_set,
        "split":          split,
        "model":          model_name,
        "status":         "pending",
        "error":          "",
        "n_train":        0,
        "n_val":          0,
        "n_test":         0,
        "duration_secs":  0.0,
    }
    try:
        if label_col not in df.columns:
            row["status"] = "skipped"
            row["error"]  = f"etiket sutunu yok: {label_col}"
            return row

        # Label tek sinifliysa ogretilemez
        if df[label_col].dropna().nunique() < 2:
            row["status"] = "skipped"
            row["error"]  = f"etiket tek sinifli: {label_col}"
            return row

        features = get_feature_set(task, feature_set)
        if split == "project":
            train, val, test = project_based_split(df)
        elif split == "time":
            if "created_at" not in df.columns:
                row["status"] = "skipped"
                row["error"]  = "created_at yok, time-split atlandi"
                return row
            train, val, test = time_based_split(df)
        else:
            raise ValueError(f"gecersiz split: {split}")

        Xtr, ytr = extract_xy(train, features, label_col)
        Xv,  yv  = extract_xy(val,   features, label_col)  # noqa: F841 — val ileri asamada
        Xte, yte = extract_xy(test,  features, label_col)
        row["n_train"] = len(ytr)
        row["n_val"]   = len(yv)
        row["n_test"]  = len(yte)

        # Scaler (train uzerinde fit — val/test transform)
        scaler = fit_scaler(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)

        # SMOTE (train only)
        if use_smote:
            try:
                Xtr_s, ytr = apply_smote_train_only(Xtr_s, ytr, RANDOM_STATE)
            except ImportError:
                row["error"] = "imblearn yok — SMOTE atlandi"
            except ValueError as exc:
                row["error"] = f"SMOTE atlandi: {exc}"

        # Model calistir
        model_fn = MODEL_REGISTRY[model_name]
        mrun = model_fn(Xtr_s, ytr, Xte_s)
        row["duration_secs"] = mrun.duration
        if mrun.status != "ok":
            row["status"] = mrun.status
            row["error"]  = mrun.error or row["error"]
            return row

        metrics = classification_metrics(yte, mrun.y_pred, mrun.y_proba)
        row.update(metrics)
        row["status"] = "ok"
        return row
    except Exception as exc:  # noqa: BLE001 — ablation rowu bagimsiz raporlanir
        row["status"] = "failed"
        row["error"]  = f"{type(exc).__name__}: {exc}"
        return row

# %% Hucre 6 - Ablation loop
def iter_combos(cfg: dict):
    """Smart pruning ile kombinasyon uret."""
    big_set = "all"
    small_subset = set(cfg["small_feature_model_subset"])
    for task_spec in cfg["tasks"]:
        for fset in cfg["feature_sets"]:
            for split in cfg["splits"]:
                models = cfg["models"]
                if cfg.get("prune_models_on_small_feature_sets") and fset != big_set:
                    models = [m for m in models if m in small_subset]
                for model in models:
                    yield {
                        "task":          task_spec["name"],
                        "label_col":     task_spec["label_col"],
                        "label_variant": task_spec["label_variant"],
                        "feature_set":   fset,
                        "split":         split,
                        "model":         model,
                    }


combos = list(iter_combos(ABLATION))
print(f"Etkin kombinasyon sayisi (pruning sonrasi): {len(combos)}")

results: list[dict] = []
for i, combo in enumerate(combos, 1):
    logger.info(
        "[%d/%d] task=%s label=%s feat=%s split=%s model=%s",
        i, len(combos),
        combo["task"], combo["label_variant"],
        combo["feature_set"], combo["split"], combo["model"],
    )
    res = run_one_combo(
        df=df,
        task=combo["task"],
        label_col=combo["label_col"],
        label_variant=combo["label_variant"],
        feature_set=combo["feature_set"],
        split=combo["split"],
        model_name=combo["model"],
        use_smote=ABLATION["use_smote"],
    )
    results.append(res)

results_df = pd.DataFrame(results)
print(results_df.head())

# %% Hucre 7 - Ozet + CSV yaz
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_csv = OUTPUT_DIR / f"ablation_results_{ts}.csv"
results_df.to_csv(out_csv, index=False)
print(f"kaydedildi: {out_csv}")

ok = results_df[results_df["status"] == "ok"]
if len(ok):
    print("\nEn iyi 10 satir (f1 azalan):")
    print(
        ok.sort_values("f1", ascending=False)
          .head(10)
          [["task", "label_variant", "feature_set", "split", "model",
            "f1", "pr_auc", "mcc", "accuracy", "duration_secs"]]
          .to_string(index=False)
    )
else:
    print("UYARI: 'ok' statuslu kosu yok. Log ve error sutununu inceleyin.")

# %% Hucre 8 - Skipped/failed raporu (hangi bagimlilik eksikse yakalar)
nonok = results_df[results_df["status"] != "ok"]
if len(nonok):
    print("Atlanan/basarisiz kosular:")
    print(nonok[["task", "label_variant", "feature_set", "split", "model",
                 "status", "error"]].to_string(index=False))
else:
    print("Tum kombinasyonlar 'ok' dondu.")
