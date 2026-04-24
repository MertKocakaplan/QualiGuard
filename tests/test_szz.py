"""
test_szz.py — SZZ sarmalayicisinin temel davranislari.

Gercek pydriller cagrilarini mock'luyoruz: Git sinifi ile
get_commits_last_modified_lines ve get_commit sahte veri doner.
Gercek repo ile integration test F2 pilot asamasinda elle yapilir.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# pydriller runtime bagimliligi; opsiyonel dev extra. Kurulu degilse
# bu modul butunuyle skip olur — collection error'a neden olmaz.
pytest.importorskip("pydriller")

from pipeline import szz  # noqa: E402 — importorskip'ten sonra import


class _FakeCommit:
    def __init__(self, h: str):
        self.hash = h


def _fake_git(intro_map: dict[str, set[str]]):
    """Git sinifinin minimal sahtesi."""
    git = MagicMock()
    git.get_commit.side_effect = lambda h: _FakeCommit(h)
    git.get_commits_last_modified_lines.return_value = intro_map
    git.clear.return_value = None
    return git


# ── Temel davranislar ────────────────────────────────────────────

def test_empty_head_files_returns_empty():
    out = szz.compute_szz_labels(Path("/x"), [], ["abc"])
    assert out == {}


def test_no_bug_fix_commits_all_zero(monkeypatch):
    with patch("pipeline.szz.Git", return_value=_fake_git({})):
        out = szz.compute_szz_labels(Path("/x"), ["a.py", "b.py"], [])
    assert out == {"a.py": 0, "b.py": 0}


def test_labels_head_file_when_intro_map_matches():
    intro = {"src/a.py": {"aaa111"}, "unrelated/x.py": {"bbb222"}}
    head_files = ["src/a.py", "src/b.py"]
    with patch("pipeline.szz.Git", return_value=_fake_git(intro)):
        out = szz.compute_szz_labels(Path("/x"), head_files, ["fix1"])
    assert out["src/a.py"]   == 1
    assert out["src/b.py"]   == 0


def test_non_head_files_ignored():
    """intro_map HEAD disinda bir yol donerse etiket degismemeli."""
    intro = {"removed/gone.py": {"ccc333"}}
    head_files = ["kept.py"]
    with patch("pipeline.szz.Git", return_value=_fake_git(intro)):
        out = szz.compute_szz_labels(Path("/x"), head_files, ["fix1"])
    assert out == {"kept.py": 0}


def test_git_init_failure_returns_empty_for_fallback():
    with patch("pipeline.szz.Git", side_effect=RuntimeError("bozuk repo")):
        out = szz.compute_szz_labels(Path("/x"), ["a.py"], ["fix1"])
    assert out == {}


def test_get_commit_error_skipped_and_others_processed():
    git = MagicMock()
    def _get_commit(h):
        if h == "bad":
            raise ValueError("missing object")
        return _FakeCommit(h)
    git.get_commit.side_effect = _get_commit
    git.get_commits_last_modified_lines.return_value = {"a.py": {"intro"}}
    git.clear.return_value = None

    with patch("pipeline.szz.Git", return_value=git):
        out = szz.compute_szz_labels(Path("/x"), ["a.py"], ["bad", "good"])
    assert out == {"a.py": 1}


def test_blame_error_does_not_crash():
    git = MagicMock()
    git.get_commit.side_effect = lambda h: _FakeCommit(h)
    git.get_commits_last_modified_lines.side_effect = RuntimeError("blame patlattik")
    git.clear.return_value = None

    with patch("pipeline.szz.Git", return_value=git):
        out = szz.compute_szz_labels(Path("/x"), ["a.py"], ["fix1"])
    assert out == {"a.py": 0}


def test_timeout_returns_empty_for_fallback(monkeypatch):
    """time.monotonic'i once 0, sonra timeout otesi dondur."""
    fake_times = iter([0.0, 0.0, 9999.0, 9999.0, 9999.0])

    def _mono():
        try:
            return next(fake_times)
        except StopIteration:
            return 9999.0

    git = _fake_git({"a.py": {"x"}})
    with patch("pipeline.szz.Git", return_value=git), \
         patch("pipeline.szz.time.monotonic", side_effect=_mono):
        out = szz.compute_szz_labels(
            Path("/x"), ["a.py", "b.py"], ["f1", "f2", "f3"], timeout_seconds=1,
        )
    assert out == {}


def test_backslash_paths_normalized_in_intro_map():
    """pydriller Windows'ta ters slash dondurebilir."""
    intro = {"src\\a.py": {"intro1"}}
    head_files = ["src/a.py"]
    with patch("pipeline.szz.Git", return_value=_fake_git(intro)):
        out = szz.compute_szz_labels(Path("/x"), head_files, ["fix1"])
    assert out == {"src/a.py": 1}
