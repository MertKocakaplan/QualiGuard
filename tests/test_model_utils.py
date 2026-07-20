"""
test_model_utils.py — pipeline.model_utils split, scaler, SMOTE, metric yardimcilari.

PLAN §4.3 F5 DoD:
  - GroupKFold boundary (proje sizintisi yok)
  - time_based_split kronolojik sira
  - SMOTE sadece train'de uygulanir
  - feature set variant (static/derived/process/all)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.config import FEATURES_BUG, FEATURES_COMMIT
from pipeline import model_utils as mu


# ── Helpers ───────────────────────────────────────────────────────

def _sample_df(n_projects: int = 5, rows_per_project: int = 8) -> pd.DataFrame:
    """Sentetik mini dataset: project_name + created_at + numeric feature'lar."""
    rng = np.random.default_rng(0)
    rows = []
    base_ts = pd.Timestamp("2024-01-01", tz="UTC")
    for pi in range(n_projects):
        for ri in range(rows_per_project):
            rows.append({
                "project_name":  f"u/p{pi}",
                "created_at":    (base_ts + pd.Timedelta(days=pi * 30 + ri)).isoformat(),
                "loc":           int(rng.integers(10, 500)),
                "cc_mean":       float(rng.uniform(1, 10)),
                "commit_count":  int(rng.integers(1, 50)),
                "label_commit":  int(rng.integers(0, 2)),
                "smell_binary":  int(rng.integers(0, 2)),
            })
    return pd.DataFrame(rows)


# ── project_based_split ───────────────────────────────────────────

def test_project_based_split_sizes_and_no_overlap():
    df = _sample_df(n_projects=10, rows_per_project=4)
    train, val, test = mu.project_based_split(df, random_state=42)
    # Her split'te dosya var
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    # Toplam orijinal satira esit
    assert len(train) + len(val) + len(test) == len(df)
    # Proje sizintisi yok
    tr_set = set(train["project_name"])
    va_set = set(val["project_name"])
    te_set = set(test["project_name"])
    assert tr_set.isdisjoint(va_set)
    assert tr_set.isdisjoint(te_set)
    assert va_set.isdisjoint(te_set)


def test_project_based_split_raises_on_missing_column():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError):
        mu.project_based_split(df)


def test_project_based_split_deterministic_with_same_seed():
    df = _sample_df(n_projects=6, rows_per_project=3)
    a1, _, _ = mu.project_based_split(df, random_state=7)
    a2, _, _ = mu.project_based_split(df, random_state=7)
    assert set(a1["project_name"]) == set(a2["project_name"])


# ── time_based_split ─────────────────────────────────────────────

def test_time_based_split_chronological_order():
    df = _sample_df(n_projects=4, rows_per_project=6)
    train, val, test = mu.time_based_split(df)
    ts_train = pd.to_datetime(train["created_at"], utc=True).max()
    ts_val   = pd.to_datetime(val["created_at"],   utc=True).min()
    ts_val_max = pd.to_datetime(val["created_at"], utc=True).max()
    ts_test  = pd.to_datetime(test["created_at"],  utc=True).min()
    assert ts_train <= ts_val
    assert ts_val_max <= ts_test


def test_time_based_split_drops_ts_column():
    df = _sample_df()
    train, val, test = mu.time_based_split(df)
    for part in (train, val, test):
        assert "_ts" not in part.columns


def test_time_based_split_handles_nat_at_end():
    df = _sample_df(n_projects=3, rows_per_project=4)
    # Bir satiri bozuk tarih ile degistir -> NaT
    df.loc[0, "created_at"] = "not-a-date"
    train, val, test = mu.time_based_split(df)
    # Toplam kayip yok
    assert len(train) + len(val) + len(test) == len(df)


def test_time_based_split_raises_on_missing_column():
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(ValueError):
        mu.time_based_split(df, time_col="created_at")


def test_time_based_split_raises_on_oversized_eval():
    df = _sample_df(n_projects=2, rows_per_project=2)  # n=4
    with pytest.raises(ValueError):
        mu.time_based_split(df, val_size=0.8, test_size=0.8)


# ── group_kfold_indices ──────────────────────────────────────────

def test_group_kfold_indices_no_leakage():
    df = _sample_df(n_projects=5, rows_per_project=4)
    splits = list(mu.group_kfold_indices(df, n_splits=5))
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        train_projs = set(df.iloc[train_idx]["project_name"])
        test_projs  = set(df.iloc[test_idx]["project_name"])
        assert train_projs.isdisjoint(test_projs), (
            f"Proje sizintisi: {train_projs & test_projs}"
        )


# ── get_feature_set ──────────────────────────────────────────────

def test_get_feature_set_static_is_24():
    # F3.1: +2 cognitive complexity → 22 → 24
    assert len(mu.get_feature_set("commit", "static")) == 24


def test_get_feature_set_derived_is_static_plus_4():
    static = mu.get_feature_set("commit", "static")
    derived = mu.get_feature_set("commit", "derived")
    assert len(derived) == len(static) + 4
    # Derived, static'i icermeli (cumulative)
    assert set(static).issubset(set(derived))


def test_get_feature_set_process_equals_all_for_commit():
    process = mu.get_feature_set("commit", "process")
    all_set = mu.get_feature_set("commit", "all")
    # Commit taski icin process = FEATURES_COMMIT (proje meta + repo-history dahil)
    assert len(process) == len(all_set) == len(FEATURES_COMMIT)


def test_get_feature_set_all_bug_matches_features_bug():
    """get_feature_set('bug','all') FEATURES_BUG ile birebir tutarli olmali.

    Feature selection sonrasi sayi degisebildiginden sabit 42 yerine
    FEATURES_BUG'a dinamik baglanir (icerik + uzunluk esitligi)."""
    fs = mu.get_feature_set("bug", "all")
    assert len(fs) == len(FEATURES_BUG)
    assert set(fs) == set(FEATURES_BUG)


def test_get_feature_set_all_smell_matches_features_smell():
    """get_feature_set('smell','all') FEATURES_SMELL ile birebir tutarli olmali.

    Feature selection sonrasi sayi degisebildiginden sabit 48 yerine
    FEATURES_SMELL'e dinamik baglanir (icerik + uzunluk esitligi)."""
    from pipeline.config import FEATURES_SMELL
    fs = mu.get_feature_set("smell", "all")
    assert len(fs) == len(FEATURES_SMELL)
    assert set(fs) == set(FEATURES_SMELL)


def test_get_feature_set_all_commit_has_35_cols():
    assert len(mu.get_feature_set("commit", "all")) == len(FEATURES_COMMIT) == 35


def test_get_feature_set_invalid_task_raises():
    with pytest.raises(ValueError):
        mu.get_feature_set("invalid", "all")


def test_get_feature_set_invalid_variant_raises():
    with pytest.raises(ValueError):
        mu.get_feature_set("commit", "unknown")


# ── extract_xy ────────────────────────────────────────────────────

def test_extract_xy_basic_shape():
    df = pd.DataFrame({
        "loc":          [10, 20, 30],
        "cc_mean":      [1.0, 2.0, 3.0],
        "label_commit": [0, 1, 1],
    })
    X, y = mu.extract_xy(df, ["loc", "cc_mean"], "label_commit")
    assert X.shape == (3, 2)
    assert y.tolist() == [0, 1, 1]


def test_extract_xy_missing_feature_raises():
    df = pd.DataFrame({"loc": [1, 2], "label_commit": [0, 1]})
    with pytest.raises(KeyError):
        mu.extract_xy(df, ["loc", "cc_mean"], "label_commit")


def test_extract_xy_missing_label_raises():
    df = pd.DataFrame({"loc": [1, 2]})
    with pytest.raises(KeyError):
        mu.extract_xy(df, ["loc"], "label_commit")


def test_extract_xy_fills_nan_with_zero():
    df = pd.DataFrame({
        "loc":          [1.0, np.nan, 3.0],
        "label_commit": [0, 1, 0],
    })
    X, _ = mu.extract_xy(df, ["loc"], "label_commit")
    assert X[1, 0] == 0.0


# ── fit_scaler ────────────────────────────────────────────────────

def test_fit_scaler_learns_params():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    scaler = mu.fit_scaler(X)
    # StandardScaler mean_ ve scale_ ogrendi
    assert scaler.mean_.shape == (2,)
    assert scaler.scale_.shape == (2,)
    Xt = scaler.transform(X)
    # Transform ettigi ciktinin kolon ortalamasi ~0
    assert np.allclose(Xt.mean(axis=0), 0.0, atol=1e-9)


# ── classification_metrics ───────────────────────────────────────

def test_classification_metrics_perfect_score():
    y = np.array([0, 1, 1, 0, 1])
    out = mu.classification_metrics(y, y, y.astype(float))
    assert out["f1"] == pytest.approx(1.0)
    assert out["f1_weighted"] == pytest.approx(1.0)
    assert out["f1_macro"] == pytest.approx(1.0)
    assert out["accuracy"] == pytest.approx(1.0)
    assert out["mcc"] == pytest.approx(1.0)
    assert out["pr_auc"] == pytest.approx(1.0)


def test_classification_metrics_pr_auc_nan_without_proba():
    y = np.array([0, 1, 0])
    pred = np.array([0, 1, 0])
    out = mu.classification_metrics(y, pred)
    assert np.isnan(out["pr_auc"])
    assert out["f1"] == pytest.approx(1.0)
    assert out["f1_weighted"] == pytest.approx(1.0)


def test_classification_metrics_imbalanced_distinguishes_averages():
    """
    Class imbalance'da binary, weighted ve macro F1 farkli yorumlar verir.
    "Predict-majority" baseline icin binary=0, weighted yuksek (paper'a not).
    """
    # 90 negatif + 10 pozitif; her zaman 0 tahmin et
    y_true = np.array([0]*90 + [1]*10)
    y_pred = np.zeros_like(y_true)
    out = mu.classification_metrics(y_true, y_pred)
    assert out["f1"] == pytest.approx(0.0)             # pozitif sinif sifir
    assert out["f1_weighted"] > 0.8                    # majority class sisirir
    assert out["f1_macro"] < 0.6                       # sinif eslik dengeli


# ── confusion_quadrants ──────────────────────────────────────────

def test_confusion_quadrants_all_four_cells():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    # tn=2 (indx 0,4), fp=1 (indx 1), fn=1 (indx 3), tp=2 (indx 2,5)
    q = mu.confusion_quadrants(y_true, y_pred)
    assert q == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}


# ── pr_curve ──────────────────────────────────────────────────────

def test_pr_curve_returns_three_arrays():
    y_true  = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.35, 0.8])
    p, r, t = mu.pr_curve(y_true, y_proba)
    assert p.shape[0] == r.shape[0]
    assert t.shape[0] == p.shape[0] - 1


# ── apply_smote_train_only ───────────────────────────────────────

def test_apply_smote_train_only_balances_classes():
    pytest.importorskip("imblearn")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 4))
    y = np.array([0] * 40 + [1] * 10)
    X_res, y_res = mu.apply_smote_train_only(X, y, random_state=42)
    # SMOTE siniflari dengelemelidir
    _, counts = np.unique(y_res, return_counts=True)
    assert counts.min() == counts.max()
    # Ornek sayisi artmis
    assert len(y_res) >= len(y)


def test_apply_smote_train_only_raises_import_error_when_module_missing(monkeypatch):
    """imblearn olmayan bir ortamda ImportError firlatmali."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("imblearn"):
            raise ImportError("no module imblearn (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        mu.apply_smote_train_only(np.zeros((4, 2)), np.array([0, 0, 1, 1]))


# ── F4 — two_stage_split ─────────────────────────────────────────

def _large_df(n_projects: int = 20, rows_per: int = 5) -> pd.DataFrame:
    """project_name + created_at + numeric cols."""
    rng = np.random.default_rng(0)
    rows = []
    base_ts = pd.Timestamp("2024-01-01", tz="UTC")
    for pi in range(n_projects):
        for ri in range(rows_per):
            rows.append({
                "project_name": f"u/p{pi:03d}",
                "created_at":   (base_ts + pd.Timedelta(days=pi * 5 + ri)).isoformat(),
                "loc":          int(rng.integers(10, 500)),
                "label_commit": int(rng.integers(0, 2)),
            })
    return pd.DataFrame(rows)


def test_two_stage_split_proportions_correct():
    """100 proje ile train ~70%, val ~15%, test ~15%."""
    df = _large_df(n_projects=100, rows_per=3)
    s = mu.two_stage_split(df, seed=42)
    n = 100
    # Project sayilari
    assert abs(len(s.test_pids)  - int(0.15 * n)) <= 1
    assert abs(len(s.val_pids)   - int(0.15 * n)) <= 1
    tr_frac = len(s.train_pids) / n
    assert 0.68 <= tr_frac <= 0.72


def test_two_stage_split_no_project_overlap():
    """Hicbir proje iki bolumde birden yer alamaz."""
    df = _large_df(n_projects=30, rows_per=4)
    s = mu.two_stage_split(df)
    tr = set(s.train_pids)
    va = set(s.val_pids)
    te = set(s.test_pids)
    assert tr.isdisjoint(va), f"train ∩ val = {tr & va}"
    assert tr.isdisjoint(te), f"train ∩ test = {tr & te}"
    assert va.isdisjoint(te), f"val ∩ test = {va & te}"
    # Satirlar da sizmasin
    tr_names = set(s.train_dev["project_name"])
    va_names = set(s.val["project_name"])
    te_names = set(s.test["project_name"])
    assert tr_names.isdisjoint(va_names)
    assert tr_names.isdisjoint(te_names)


def test_two_stage_split_deterministic():
    """Ayni seed → ayni proje bolunum."""
    df = _large_df(n_projects=40, rows_per=3)
    s1 = mu.two_stage_split(df, seed=7)
    s2 = mu.two_stage_split(df, seed=7)
    assert set(s1.train_pids) == set(s2.train_pids)
    assert set(s1.val_pids)   == set(s2.val_pids)
    assert set(s1.test_pids)  == set(s2.test_pids)


def test_two_stage_split_different_seeds_differ():
    """Farkli seed → farkli bolunum (olasilikla)."""
    df = _large_df(n_projects=50, rows_per=2)
    s1 = mu.two_stage_split(df, seed=1)
    s2 = mu.two_stage_split(df, seed=999)
    assert set(s1.train_pids) != set(s2.train_pids)


def test_two_stage_split_all_rows_covered():
    """train + val + test toplami orijinal satir sayisina esit olmali."""
    df = _large_df(n_projects=20, rows_per=6)
    s = mu.two_stage_split(df)
    total = len(s.train_dev) + len(s.val) + len(s.test)
    assert total == len(df)


def test_two_stage_split_missing_col_raises():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError):
        mu.two_stage_split(df)


def test_two_stage_split_returns_named_tuple():
    """Donus degeri TwoStageSplit NamedTuple olmali."""
    df = _large_df(n_projects=10, rows_per=3)
    s = mu.two_stage_split(df)
    assert isinstance(s, mu.TwoStageSplit)
    # Alan isimleri
    assert hasattr(s, "train_dev")
    assert hasattr(s, "train_pids")
    assert isinstance(s.train_pids, np.ndarray)
