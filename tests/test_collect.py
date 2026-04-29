"""
test_collect.py — F3 integration: scripts.collect orkestrasyonu.

Amac: `--phase process` ve `--phase build` uc fake proje ile uctan uca
calisip (1) parquet dosyalarini yazdigini, (2) processed_projects.json'a
dogru flush ettigini, (3) --resume'un onceden islenmisleri atladigini,
(4) build ciktisinin etiket sutunlarini eklediğini dogrular.

process_project modul-seviyesinde mock'lanir (integration odagi collect.py
orkestrasyonudur; process_project'in kendi detay davranislari
test_project_processor.py'de dogrulanir).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


# ── Yardimcilar ──────────────────────────────────────────────────

def _make_project(name: str, idx: int) -> dict:
    """discovery.json icindeki bir kaydi temsil eder."""
    return {
        "full_name":         name,
        "clone_url":         f"https://github.com/{name}.git",
        "stars":             10 + idx,
        "created_at":        "2025-06-01T00:00:00Z",
        "project_age_days":  300,
        "contributor_count": 3,
        "default_branch":    "main",
    }


def _write_discovery(checkpoint_dir: Path, projects: list[dict]) -> None:
    """discovery.json dosyasini fake icerikle yaz."""
    payload = {
        "started_at":   "2026-04-24T00:00:00Z",
        "completed_at": "2026-04-24T00:00:01Z",
        "criteria":     {},
        "target_count": len(projects),
        "found_count":  len(projects),
        "found":        projects,
    }
    (checkpoint_dir / "discovery.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )


def _write_minimal_project_parquet(
    path: Path, project_name: str, rows: list[dict],
) -> None:
    """Per-project parquet'i manuel olusturmak icin (build testi)."""
    df = pd.DataFrame(rows)
    df["project_name"] = project_name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


# ── Fixture: tum config path'lerini tmp_path'e yonlendir ────────

@pytest.fixture
def collect_env(tmp_path, monkeypatch):
    """
    scripts.collect'in dokundugu tum output dizinlerini tmp_path'e yonlendir.

    Patch edilen isimler modullerde **import edildigi haliyle** degistirilir;
    aksi halde stale referanslar devreye girer.
    """
    out       = tmp_path / "output"
    checkpnt  = out / "checkpoints"
    projects  = out / "projects"
    logs      = out / "logs"
    for d in (out, checkpnt, projects, logs):
        d.mkdir(parents=True, exist_ok=True)

    # Modul-level isimleri override et
    monkeypatch.setattr("pipeline.config.CHECKPOINT_DIR", checkpnt, raising=False)
    monkeypatch.setattr("pipeline.config.PROJECTS_DIR",   projects, raising=False)
    monkeypatch.setattr("pipeline.config.OUTPUT_DIR",     out,      raising=False)
    monkeypatch.setattr("pipeline.config.LOGS_DIR",       logs,     raising=False)
    monkeypatch.setattr("pipeline.checkpoint.CHECKPOINT_DIR",      checkpnt, raising=False)
    monkeypatch.setattr("pipeline.dataset_builder.PROJECTS_DIR",   projects, raising=False)
    monkeypatch.setattr("pipeline.dataset_builder.OUTPUT_DIR",     out,      raising=False)
    monkeypatch.setattr("scripts.collect.CHECKPOINT_DIR",          checkpnt, raising=False)
    monkeypatch.setattr("scripts.collect.OUTPUT_DIR",              out,      raising=False)
    monkeypatch.setattr("scripts.collect.LOGS_DIR",                logs,     raising=False)

    return {"out": out, "checkpoint": checkpnt, "projects": projects, "logs": logs}


def _fake_process_ok(projects_dir: Path):
    """
    process_project mock factory — verilen dizine minimal parquet yaz.

    Gercek process_project'in semasina yakin (schema testleri buna guvenmemeli,
    integration testleri sadece "parquet yazildi mi" kontrolu yapar).
    """
    def _inner(project, **kwargs):
        safe = project["full_name"].replace("/", "_")
        parquet_path = projects_dir / f"{safe}.parquet"
        pd.DataFrame([{
            "file_path":    "x.py",
            "project_name": project["full_name"],
            "loc":          120,
            "commit_count": 4,
            "smell_count":  3,
            "bug_keyword":  1,
        }]).to_parquet(parquet_path, index=False)
        return {
            "status":         "ok",
            "files":          1,
            "total_loc":      120,
            "bugs_keyword":   1,
            "bugs_szz":       0,
            "smells_total":   3,
            "smells_missing": 0,
            "parquet":        str(parquet_path),
            "duration_secs":  0.05,
            "completed_at":   "2026-04-24T00:00:02Z",
        }
    return _inner


# ── Tests ──────────────────────────────────────────────────────────

def test_process_phase_end_to_end_three_projects(collect_env):
    """
    3 fake proje -> collect.main(["--phase", "process"])
    -> 3 parquet + processed_projects.json'da 3 ok kaydi + rc=0.
    """
    from scripts import collect as collect_mod

    projects = [_make_project(f"u/p{i}", i) for i in range(3)]
    _write_discovery(collect_env["checkpoint"], projects)

    with patch(
        "scripts.collect.project_processor.process_project",
        side_effect=_fake_process_ok(collect_env["projects"]),
    ):
        rc = collect_mod.main(["--phase", "process", "--log-level", "WARNING"])

    assert rc == 0

    # 3 parquet yazildi
    parquet_files = sorted(collect_env["projects"].glob("*.parquet"))
    assert len(parquet_files) == 3

    # processed_projects.json: 3 ok kaydi
    processed = json.loads(
        (collect_env["checkpoint"] / "processed_projects.json").read_text(encoding="utf-8")
    )
    assert set(processed["processed"].keys()) == {"u/p0", "u/p1", "u/p2"}
    assert all(v["status"] == "ok" for v in processed["processed"].values())


def test_resume_skips_already_processed_projects(collect_env):
    """
    discovery.json: 3 proje; processed_projects.json'da 2'si onceden ok ise
    --resume ile sadece 3. proje islenir.
    """
    from scripts import collect as collect_mod

    projects = [_make_project(f"u/p{i}", i) for i in range(3)]
    _write_discovery(collect_env["checkpoint"], projects)

    # 2 projeyi onceden ok isaretle
    (collect_env["checkpoint"] / "processed_projects.json").write_text(
        json.dumps({"processed": {
            "u/p0": {"status": "ok", "files": 5},
            "u/p1": {"status": "ok", "files": 7},
        }}),
        encoding="utf-8",
    )

    fake = _fake_process_ok(collect_env["projects"])
    with patch(
        "scripts.collect.project_processor.process_project",
        side_effect=fake,
    ) as m:
        rc = collect_mod.main(["--phase", "process", "--resume", "--log-level", "WARNING"])

    assert rc == 0
    # Yalnizca bir kez cagrildi (p2)
    assert m.call_count == 1
    called_with = m.call_args_list[0].args[0]
    assert called_with["full_name"] == "u/p2"

    # p0, p1 icin parquet yok (mock skip), p2 icin var
    created = sorted(p.name for p in collect_env["projects"].glob("*.parquet"))
    assert created == ["u_p2.parquet"]

    # processed_projects.json'da artik 3 kayit
    processed = json.loads(
        (collect_env["checkpoint"] / "processed_projects.json").read_text(encoding="utf-8")
    )
    assert set(processed["processed"].keys()) == {"u/p0", "u/p1", "u/p2"}


def test_build_phase_produces_dataset_full_with_labels(collect_env):
    """
    build fazi: manuel per-project parquet'lerden dataset_full_*.parquet uret
    ve smell_binary / label_commit sutunlarini ekle.
    """
    from scripts import collect as collect_mod

    # 2 per-project parquet yaz
    _write_minimal_project_parquet(
        collect_env["projects"] / "u_a.parquet",
        "u/a",
        [{"file_path": f"a{i}.py", "commit_count": c, "smell_count": s}
         for i, (c, s) in enumerate([(1, 2), (5, 8), (20, 30)])],
    )
    _write_minimal_project_parquet(
        collect_env["projects"] / "u_b.parquet",
        "u/b",
        [{"file_path": f"b{i}.py", "commit_count": c, "smell_count": s}
         for i, (c, s) in enumerate([(3, 4), (10, 15), (50, 60)])],
    )

    rc = collect_mod.main(["--phase", "build", "--log-level", "WARNING"])
    assert rc == 0

    outputs = sorted(collect_env["out"].glob("dataset_full_*.parquet"))
    assert len(outputs) == 1

    df = pd.read_parquet(outputs[0])
    assert "label_commit" in df.columns
    assert "smell_binary" in df.columns
    assert len(df) == 6  # 3 + 3


def test_process_phase_missing_discovery_returns_error(collect_env):
    """discovery.json yoksa process fazi 1 donmeli."""
    from scripts import collect as collect_mod

    rc = collect_mod.main(["--phase", "process", "--log-level", "ERROR"])
    assert rc == 1


def test_build_phase_empty_input_returns_error(collect_env):
    """Hicbir per-project parquet yoksa build fazi 1 donmeli."""
    from scripts import collect as collect_mod

    rc = collect_mod.main(["--phase", "build", "--log-level", "ERROR"])
    assert rc == 1


def test_dry_run_writes_nothing(collect_env, capsys):
    """--dry-run hicbir dosya/checkpoint yazmaz, config raporu bastirir."""
    from scripts import collect as collect_mod

    rc = collect_mod.main(["--dry-run", "--target", "42"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "MetricHunter V2" in captured.out
    assert "target" in captured.out
    # Hicbir cikti dosyasi yok
    assert not list(collect_env["projects"].glob("*"))
    assert not list(collect_env["checkpoint"].glob("*"))


def test_discovery_merge_preserves_old_projects(collect_env):
    """
    5 onceki proje varken 3 yeni (1 overlapping) gelindiginde 7 proje olur;
    overlapping proje yeni meta'ya sahip olur, eski projeler korunur.
    """
    from scripts import collect as collect_mod

    old_projects = [
        {**_make_project(f"u/old{i}", i), "description": f"old_desc{i}"}
        for i in range(5)
    ]
    _write_discovery(collect_env["checkpoint"], old_projects)

    new_discovered = [
        {**_make_project("u/old0", 99), "description": "fresh_desc"},  # overlap
        _make_project("u/new1", 10),
        _make_project("u/new2", 11),
    ]

    with patch("pipeline.discovery.search_projects", return_value=new_discovered), \
         patch("scripts.collect.refresh_quota"), \
         patch("scripts.collect.current_quota", return_value={}):
        rc = collect_mod.main(["--phase", "discovery", "--log-level", "WARNING"])

    assert rc == 0

    data = json.loads(
        (collect_env["checkpoint"] / "discovery.json").read_text(encoding="utf-8")
    )
    found = {p["full_name"]: p for p in data["found"]}

    assert len(found) == 7
    assert found["u/old0"]["description"] == "fresh_desc"
    for i in range(1, 5):
        assert f"u/old{i}" in found
        assert found[f"u/old{i}"]["description"] == f"old_desc{i}"
