"""
test_checkpoint.py — checkpoint.py unit testleri.
"""
from __future__ import annotations

import json

import pytest

from pipeline import checkpoint


def test_load_missing_returns_none(isolated_checkpoint_dir):
    assert checkpoint.load_checkpoint("yok") is None


def test_save_and_load_roundtrip(isolated_checkpoint_dir):
    data = {"found": [{"full_name": "a/b"}], "found_count": 1}
    checkpoint.save_checkpoint("discovery", data)
    loaded = checkpoint.load_checkpoint("discovery")
    assert loaded == data


def test_atomic_write_leaves_no_temp(isolated_checkpoint_dir):
    checkpoint.save_checkpoint("discovery", {"x": 1})
    # temp isimleri ".<name>.*.tmp" sekline sahip — hedef yazildiktan sonra kalmamali
    leftovers = [p.name for p in isolated_checkpoint_dir.iterdir()
                 if p.name.startswith(".") and p.name.endswith(".tmp")]
    assert leftovers == []


def test_mark_project_done_and_is_done(isolated_checkpoint_dir):
    checkpoint.mark_project_done("user/repo-ok", {"status": "ok", "files": 10})
    checkpoint.mark_project_done("user/repo-fail", {"status": "failed", "error": "timeout"})

    assert checkpoint.is_project_done("user/repo-ok") is True
    # failed olan tekrar denenebilsin — False donmeli
    assert checkpoint.is_project_done("user/repo-fail") is False
    assert checkpoint.is_project_done("user/unknown") is False


def test_get_processed_set_only_ok(isolated_checkpoint_dir):
    checkpoint.mark_project_done("a/b", {"status": "ok"})
    checkpoint.mark_project_done("c/d", {"status": "failed"})
    checkpoint.mark_project_done("e/f", {"status": "ok"})
    assert checkpoint.get_processed_set() == {"a/b", "e/f"}


def test_corrupt_json_returns_none(isolated_checkpoint_dir):
    path = isolated_checkpoint_dir / "broken.json"
    path.write_text("not a json", encoding="utf-8")
    assert checkpoint.load_checkpoint("broken") is None


def test_phase_validation_rejects_empty(isolated_checkpoint_dir):
    with pytest.raises(ValueError):
        checkpoint.save_checkpoint("", {"x": 1})


def test_mark_project_done_requires_name(isolated_checkpoint_dir):
    with pytest.raises(ValueError):
        checkpoint.mark_project_done("", {"status": "ok"})
