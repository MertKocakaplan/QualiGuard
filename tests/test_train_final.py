"""
test_train_final.py — scripts.train_final CLI + T1/T3 egitim uc-uca.

- --dry-run: args'lari dogru okur, hicbir sey yazmaz.
- T1 (commit) + T3 (smell) gorevleri mini sentetik dataset uzerinde egitilir
  (AutoGluon agir bagimlilik, T2 bug burada skip).
- Cikti artifact'lari gercekten yaziliyor mu, feature_names.json guncelleniyor
  mu, project_stats.json olusuyor mu kontrol edilir.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.config import FEATURES_BUG
from scripts import train_final


# ── CLI / dry-run ─────────────────────────────────────────────────

def test_cli_dry_run(capsys):
    rc = train_final.main(["--dry-run", "--tasks", "commit,smell"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "train_final" in out
    assert "commit" in out and "smell" in out


def test_cli_invalid_task_exits_2():
    with pytest.raises(SystemExit) as err:
        train_final.main(["--tasks", "bogus_task", "--dry-run"])
    # SystemExit ya int ya str — invalid task mesajiyla 2 donmeli
    assert err.value.code in (2, "HATA: bilinmeyen task(lar): ['bogus_task'] (exit 2)") \
        or "exit 2" in str(err.value)


def test_cli_invalid_time_limit_exits():
    with pytest.raises(SystemExit):
        train_final.main(["--autogluon-time-limit", "0", "--dry-run"])


# ── Sentetik veri uretici ─────────────────────────────────────────

def _synth_dataset(n_projects: int = 6, rows_per: int = 30,
                   seed: int = 0) -> pd.DataFrame:
    """FEATURES_BUG tum sutunlari + etiketler iceren mini dataset."""
    rng = np.random.default_rng(seed)
    rows = []
    base_ts = pd.Timestamp("2024-01-01", tz="UTC")
    for pi in range(n_projects):
        for ri in range(rows_per):
            rec: dict = {col: float(rng.uniform(0, 100)) for col in FEATURES_BUG}
            rec.update({
                "project_name":         f"u/p{pi}",
                "category_primary":     "Web" if pi % 2 == 0 else "AI/ML",
                "categories_all":       "Web",
                "created_at":           (base_ts + pd.Timedelta(days=pi*5+ri)).isoformat(),
                "file_path":            f"mod_{pi}_{ri}.py",
                "bug_keyword":          int(rng.integers(0, 2)),
                "bug_szz":              int(rng.integers(0, 2)),
                "smell_count":          int(rng.integers(0, 10)),
                "smell_binary":         int(rng.integers(0, 2)),
                "label_commit":         int(rng.integers(0, 2)),
                "commits_to_first_bug": int(rng.integers(-1, 20)),
            })
            rows.append(rec)
    return pd.DataFrame(rows)


# ── End-to-end T1 + T3 (AutoGluon atlanir) ──────────────────────

def test_train_commit_and_smell_e2e(tmp_path: Path, monkeypatch):
    """T1 ve T3 mini sentetik veride calisip artifact'lari yaziyor mu."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    dataset = tmp_path / "ds.parquet"
    _synth_dataset().to_parquet(dataset, index=False)

    # Logs dir ve output'un test sirasinda gercek depolari kirletmemesi icin
    monkeypatch.setattr(train_final, "LOGS_DIR", tmp_path / "logs")

    rc = train_final.main([
        "--dataset", str(dataset),
        "--tasks", "commit,smell",
        "--models-dir", str(models_dir),
        "--no-smote",
        "--log-level", "WARNING",
    ])
    assert rc == 0

    # T1 artifact'lari
    assert (models_dir / "commit_rf.joblib").exists()
    assert (models_dir / "scaler_commit.joblib").exists()
    # T3 artifact'lari
    assert (models_dir / "smell_rf.joblib").exists()
    assert (models_dir / "scaler_smell.joblib").exists()

    # feature_names.json iki gorev icin de yazilmis
    fn = json.loads((models_dir / "feature_names.json").read_text(encoding="utf-8"))
    assert "commit" in fn and "smell" in fn
    assert len(fn["commit"]) == 35
    assert len(fn["smell"]) == 48

    # project_stats.json olusmus
    ps = json.loads((models_dir / "project_stats.json").read_text(encoding="utf-8"))
    assert ps["global"]["n_files"] > 0
    assert "by_category" in ps


def test_bug_task_without_autogluon_exits_clean(tmp_path: Path, monkeypatch):
    """
    AutoGluon yoksa T2 bug 'atlandi' mesajiyla warn eder, T1 yine de calisir.
    Burada AutoGluon gercekten kurulu olsa bile, ImportError'i simule ederek
    davranisi test ediyoruz.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    dataset = tmp_path / "ds.parquet"
    _synth_dataset().to_parquet(dataset, index=False)

    # autogluon.tabular import'unu engelle
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("autogluon"):
            raise ImportError("simulated missing autogluon")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(train_final, "LOGS_DIR", tmp_path / "logs")

    rc = train_final.main([
        "--dataset", str(dataset),
        "--tasks", "commit,bug",
        "--models-dir", str(models_dir),
        "--no-smote",
        "--log-level", "WARNING",
    ])
    # T1 basarili oldugu icin cikis 0
    assert rc == 0
    # T1 yazildi
    assert (models_dir / "commit_rf.joblib").exists()
    # T2 stacking artifact'lari yazilmamali (autogluon yok)
    assert not (models_dir / "bug_rf_base.joblib").exists()
    assert not (models_dir / "bug_ag_base").exists()
