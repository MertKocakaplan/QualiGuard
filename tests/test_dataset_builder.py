"""
test_dataset_builder.py — birlestirme + label kolonlari.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipeline import dataset_builder


def _write_project_parquet(path: Path, project: str, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df["project_name"] = project
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


@pytest.fixture
def project_files(tmp_path, monkeypatch):
    pdir = tmp_path / "projects"
    monkeypatch.setattr("pipeline.dataset_builder.PROJECTS_DIR", pdir)
    monkeypatch.setattr("pipeline.dataset_builder.OUTPUT_DIR", tmp_path)
    return pdir


def test_load_project_parquets_empty(project_files):
    df = dataset_builder.load_project_parquets()
    assert df.empty


def test_load_and_concat_parquets(project_files):
    _write_project_parquet(project_files / "a.parquet", "u/a",
                           [{"file_path": "x.py", "commit_count": 3, "smell_count": 5}])
    _write_project_parquet(project_files / "b.parquet", "u/b",
                           [{"file_path": "y.py", "commit_count": 8, "smell_count": 12}])
    df = dataset_builder.load_project_parquets()
    assert len(df) == 2
    assert set(df["project_name"]) == {"u/a", "u/b"}


def test_add_dynamic_smell_binary_per_project_p80(project_files):
    # Proje u/a: smell=[1,3,5,7,9] P80=7.4 -> {9} binary=1
    # Proje u/b: smell=[10,20,30,40,50] P80=42 -> {50} binary=1
    rows_a = [{"file_path": f"a{i}.py", "commit_count": 1, "smell_count": c}
              for i, c in enumerate([1, 3, 5, 7, 9])]
    rows_b = [{"file_path": f"b{i}.py", "commit_count": 1, "smell_count": c}
              for i, c in enumerate([10, 20, 30, 40, 50])]
    _write_project_parquet(project_files / "a.parquet", "u/a", rows_a)
    _write_project_parquet(project_files / "b.parquet", "u/b", rows_b)

    df = dataset_builder.load_project_parquets()
    df = dataset_builder.add_dynamic_smell_binary(df, percentile=80)

    # Bekleme: her proje icin tepe %20 (bu ornekte 5 dosyadan 1 tanesi = max)
    pos_a = df[(df["project_name"] == "u/a") & (df["smell_binary"] == 1)]
    pos_b = df[(df["project_name"] == "u/b") & (df["smell_binary"] == 1)]
    assert pos_a["smell_count"].max() == 9
    assert pos_b["smell_count"].max() == 50


def test_add_dynamic_smell_binary_all_na_skip_prospector(project_files):
    """--skip-prospector durumu: tum smell_count NA -> smell_binary=0, crash yok."""
    import pandas as pd
    rows_a = [{"file_path": f"a{i}.py", "commit_count": 1, "smell_count": pd.NA}
              for i in range(3)]
    rows_b = [{"file_path": f"b{i}.py", "commit_count": 1, "smell_count": pd.NA}
              for i in range(3)]
    _write_project_parquet(project_files / "a.parquet", "u/a", rows_a)
    _write_project_parquet(project_files / "b.parquet", "u/b", rows_b)

    df = dataset_builder.load_project_parquets()
    df = dataset_builder.add_dynamic_smell_binary(df)

    assert (df["smell_binary"] == 0).all()
    assert df["smell_binary"].dtype.name == "int8"


def test_add_commit_label_uses_global_median(project_files):
    rows = [{"commit_count": c, "smell_count": 0} for c in [1, 2, 3, 10, 100]]
    # median = 3; strict ">3" → 1 (median esiti negatif, "high-activity files")
    _write_project_parquet(project_files / "x.parquet", "u/x", rows)
    df = dataset_builder.load_project_parquets()
    df = dataset_builder.add_commit_label(df)
    assert df["label_commit"].tolist() == [0, 0, 0, 1, 1]


def test_add_commit_label_skewed_data_not_trivial(project_files):
    """
    Veri commit_count uzerinde cok skewed olsa bile (median=1, P75=1) strict
    ">" sayesinde label_commit %0'a ya da %100'e patlamaz. Regresyon testi:
    eski "≥ median" davranisi bu durumda %100 pos veriyordu.
    """
    rows = [{"commit_count": c, "smell_count": 0}
            for c in [1, 1, 1, 1, 1, 1, 2, 5, 20]]  # median=1
    _write_project_parquet(project_files / "y.parquet", "u/y", rows)
    df = dataset_builder.load_project_parquets()
    df = dataset_builder.add_commit_label(df)
    pos = int(df["label_commit"].sum())
    # 2,5,20 pozitif; medyan-esiti 1'ler negatif kalir
    assert pos == 3
    assert pos < len(df), "tum satirlar pozitif olmamali (trivial label)"
    assert pos > 0, "hicbir satir pozitif degil — esik mantigi bozuk"


def test_build_full_dataset_writes_parquet(project_files, tmp_path):
    rows = [{"file_path": "x.py", "commit_count": 5, "smell_count": 3}]
    _write_project_parquet(project_files / "p.parquet", "u/p", rows)

    out = dataset_builder.build_full_dataset(output_dir=tmp_path,
                                              timestamp="TEST")
    assert out is not None
    assert out.exists()
    df = pd.read_parquet(out)
    assert "label_commit" in df.columns
    assert "smell_binary" in df.columns


def test_build_full_dataset_returns_none_when_no_input(project_files, tmp_path):
    out = dataset_builder.build_full_dataset(output_dir=tmp_path)
    assert out is None


# ── F4: apply_commit_filter + add_project_categories + sensitivity_summary ──

def test_apply_commit_filter_inclusive_range():
    df = pd.DataFrame({
        "file_path":    ["a", "b", "c", "d", "e"],
        "project_name": ["p"] * 5,
        "commit_count": [1, 10, 25, 80, 100],
    })
    out = dataset_builder.apply_commit_filter(df, min_commits=10, max_commits=80)
    assert sorted(out["file_path"].tolist()) == ["b", "c", "d"]


def test_apply_commit_filter_none_bounds_passthrough():
    df = pd.DataFrame({"file_path": ["a", "b"], "commit_count": [0, 9999]})
    out = dataset_builder.apply_commit_filter(df, min_commits=None, max_commits=None)
    assert len(out) == 2
    # Kopya dondurmeli (caller'in df'ine dokunmaz)
    out["commit_count"] = -1
    assert df["commit_count"].tolist() == [0, 9999]


def test_apply_commit_filter_empty_df_safe():
    out = dataset_builder.apply_commit_filter(pd.DataFrame(), 1, 10)
    assert out.empty


def test_add_project_categories_uses_name_when_meta_missing():
    df = pd.DataFrame({
        "file_path":    ["a.py", "b.py"],
        "project_name": ["pallets/flask", "pytorch/pytorch"],
    })
    out = dataset_builder.add_project_categories(df)
    mapping = dict(zip(out["project_name"], out["category_primary"]))
    assert mapping["pallets/flask"]    == "Web"
    assert mapping["pytorch/pytorch"]  == "AI/ML"
    # categories_all virgulle birlesik string
    assert "Web" in str(out.loc[out["project_name"] == "pallets/flask",
                                "categories_all"].iloc[0])


def test_add_project_categories_uses_topics_and_description():
    df = pd.DataFrame({
        "file_path":    ["x.py"],
        "project_name": ["alice/cryptic-name"],
    })
    meta = {
        "alice/cryptic-name": {
            "topics":      ["security", "vulnerability"],
            "description": "Network scanner",
        },
    }
    out = dataset_builder.add_project_categories(df, project_meta=meta)
    assert out["category_primary"].iloc[0] == "Security"


def test_add_project_categories_other_when_no_match():
    df = pd.DataFrame({
        "file_path":    ["x.py"],
        "project_name": ["zzz/xyz123abcdef"],
    })
    out = dataset_builder.add_project_categories(df)
    assert out["category_primary"].iloc[0] == "Diger"


def test_add_project_categories_empty_df_safe():
    out = dataset_builder.add_project_categories(pd.DataFrame())
    assert "category_primary" in out.columns
    assert "categories_all" in out.columns


def test_sensitivity_summary_three_filters():
    df = pd.DataFrame({
        "file_path":     [f"f{i}" for i in range(6)],
        "project_name":  ["u/p"] * 6,
        "commit_count":  [1, 10, 25, 80, 100, 500],
        "label_commit":  [0, 0, 1, 1, 1, 1],
        "smell_binary":  [0, 0, 0, 1, 1, 1],
    })
    summary = dataset_builder.sensitivity_summary(df)
    # 3 satir: (None,None), (10,100), (25,80)
    assert len(summary) == 3
    assert summary.iloc[0]["files"] == 6
    assert summary.iloc[1]["files"] == 4   # 10..100 -> [10,25,80,100]
    assert summary.iloc[2]["files"] == 2   # 25..80  -> [25,80]
    # Pozitif sinif sutunlari mevcut
    for col in ("pct_label_pos", "pct_smell_pos"):
        assert col in summary.columns


def test_sensitivity_summary_handles_empty_df():
    df = pd.DataFrame(columns=["commit_count", "label_commit", "smell_binary"])
    summary = dataset_builder.sensitivity_summary(df)
    assert len(summary) == 3
    assert (summary["files"] == 0).all()
