"""
test_prospector_runner.py — Prospector runner'inin subprocess + parse
davranislari.

Gercek prospector cagrilarini mock'luyoruz. Paralel yol icin worker=1
zorlanir (test hizli ve deterministik kalsin).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline import prospector_runner as pr


def _proc(stdout: str, returncode: int = 0, stderr: str = "") -> MagicMock:
    p = MagicMock(spec=subprocess.CompletedProcess)
    p.stdout      = stdout
    p.stderr      = stderr
    p.returncode  = returncode
    return p


def _json_payload(messages: list[dict]) -> str:
    return json.dumps({
        "summary":  {"message_count": len(messages), "tools": []},
        "messages": messages,
    })


# ── run_prospector — mutlu/mutsuz yollar ─────────────────────────

def test_success_counts_messages_and_groups_categories():
    payload = _json_payload([
        {"source": "pylint",     "message": "x"},
        {"source": "pylint",     "message": "y"},
        {"source": "pycodestyle", "message": "z"},
    ])
    with patch.object(pr.subprocess, "run", return_value=_proc(payload, 1)):
        out = pr.run_prospector(Path("a.py"))
    assert out["smell_count"] == 3
    assert out["categories"]  == {"pylint": 2, "pycodestyle": 1}
    assert len(out["messages"]) == 3


def test_clean_file_returns_zero_count_and_empty_categories():
    with patch.object(pr.subprocess, "run", return_value=_proc(_json_payload([]), 0)):
        out = pr.run_prospector(Path("clean.py"))
    assert out["smell_count"] == 0
    assert out["categories"]  == {}
    assert out["messages"]    == []


def test_timeout_returns_none_count():
    with patch.object(pr.subprocess, "run",
                      side_effect=subprocess.TimeoutExpired(cmd="prospector", timeout=30)):
        out = pr.run_prospector(Path("slow.py"), timeout_seconds=30)
    assert out["smell_count"] is None
    assert out["categories"]  == {}
    assert out["messages"]    == []


def test_file_not_found_returns_none_count():
    with patch.object(pr.subprocess, "run", side_effect=FileNotFoundError("prospector yok")):
        out = pr.run_prospector(Path("a.py"))
    assert out["smell_count"] is None


def test_garbage_stdout_returns_none_count():
    with patch.object(pr.subprocess, "run",
                      return_value=_proc("not json at all", 2, stderr="boom")):
        out = pr.run_prospector(Path("a.py"))
    assert out["smell_count"] is None


def test_leading_warnings_then_json_parsed():
    """prospector bazen JSON'dan once kendi uyarisini basiyor."""
    stdout = (
        "WARNING: some config warning\n"
        + _json_payload([{"source": "pylint", "message": "m"}])
    )
    with patch.object(pr.subprocess, "run", return_value=_proc(stdout, 1)):
        out = pr.run_prospector(Path("a.py"))
    assert out["smell_count"] == 1
    assert out["categories"]  == {"pylint": 1}


def test_message_count_missing_falls_back_to_len_messages():
    payload = json.dumps({"summary": {}, "messages": [{"source": "x"}, {"source": "y"}]})
    with patch.object(pr.subprocess, "run", return_value=_proc(payload, 1)):
        out = pr.run_prospector(Path("a.py"))
    assert out["smell_count"] == 2


# ── run_prospector_batch ─────────────────────────────────────────

def test_batch_empty_input_returns_empty():
    assert pr.run_prospector_batch([]) == {}


def test_batch_single_worker_is_sequential(monkeypatch):
    """workers=1'de Pool acilmamali — test ortamini yormasin."""
    payload = _json_payload([{"source": "pylint"}])
    with patch.object(pr.subprocess, "run", return_value=_proc(payload, 1)):
        out = pr.run_prospector_batch(
            [Path("a.py"), Path("b.py")], workers=1,
        )
    assert set(out.keys()) == {Path("a.py"), Path("b.py")}
    for r in out.values():
        assert r["smell_count"] == 1


def test_batch_clamps_workers_to_file_count(monkeypatch):
    """2 dosya + workers=8 -> Pool 2 worker'la acilmali."""
    captured = {}

    class _FakePool:
        def __init__(self, processes):
            captured["processes"] = processes
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def map(self, fn, items):
            return [fn(x) for x in items]

    payload = _json_payload([])
    with patch.object(pr, "Pool", _FakePool), \
         patch.object(pr.subprocess, "run", return_value=_proc(payload, 0)):
        pr.run_prospector_batch(
            [Path("a.py"), Path("b.py")], workers=8,
        )
    assert captured["processes"] == 2
