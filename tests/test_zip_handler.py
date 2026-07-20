"""
test_zip_handler.py — pipeline.zip_handler guvenli ZIP extract testleri.

Coverage:
  - validate_zip_metadata: happy, path traversal, absolute path
  - find_repo_root: dogrudan kok / tek-altdir saran / .git/ eksik
  - safe_extract: happy (extract + root detect) + .git/ olmadan error
  - extract_local_meta: git yoksa default degerler
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from pipeline.zip_handler import (
    MAX_FILE_COUNT,
    ZipValidationError,
    extract_local_meta,
    find_repo_root,
    safe_extract,
    validate_zip_metadata,
)


# ── Yardimcilar ──────────────────────────────────────────────────

def _make_zip(zip_path: Path, files: dict[str, str]) -> None:
    """files: {arc_name: content_str} -> ZIP olustur."""
    with zipfile.ZipFile(zip_path, "w") as zf:
        for arc, content in files.items():
            zf.writestr(arc, content)


# ── validate_zip_metadata ────────────────────────────────────────

def test_validate_zip_metadata_happy(tmp_path: Path) -> None:
    zip_p = tmp_path / "ok.zip"
    _make_zip(zip_p, {
        "myrepo/main.py":     "x = 1\n",
        "myrepo/utils.py":    "def f(): pass\n",
        "myrepo/.git/HEAD":   "ref: refs/heads/main\n",
    })
    total_size, file_count = validate_zip_metadata(zip_p)
    assert file_count == 3
    assert total_size > 0


def test_validate_zip_metadata_path_traversal_double_dot(tmp_path: Path) -> None:
    zip_p = tmp_path / "trav.zip"
    with zipfile.ZipFile(zip_p, "w") as zf:
        zf.writestr("../etc/passwd", "evil\n")
    with pytest.raises(ZipValidationError, match="path traversal"):
        validate_zip_metadata(zip_p)


def test_validate_zip_metadata_absolute_path(tmp_path: Path) -> None:
    zip_p = tmp_path / "abs.zip"
    with zipfile.ZipFile(zip_p, "w") as zf:
        zf.writestr("/etc/passwd", "evil\n")
    with pytest.raises(ZipValidationError, match="path traversal"):
        validate_zip_metadata(zip_p)


def test_validate_zip_metadata_windows_drive_path(tmp_path: Path) -> None:
    zip_p = tmp_path / "win.zip"
    with zipfile.ZipFile(zip_p, "w") as zf:
        zf.writestr("C:/Windows/System32/cmd.exe", "evil\n")
    with pytest.raises(ZipValidationError, match="path traversal"):
        validate_zip_metadata(zip_p)


# ── find_repo_root ───────────────────────────────────────────────

def test_find_repo_root_at_root(tmp_path: Path) -> None:
    extract = tmp_path / "ext"
    (extract / ".git").mkdir(parents=True)
    (extract / "main.py").write_text("x = 1\n", encoding="utf-8")
    assert find_repo_root(extract) == extract


def test_find_repo_root_in_single_subdir(tmp_path: Path) -> None:
    """GitHub'in default ZIP yapisi: flask-main/.git/, flask-main/main.py"""
    extract = tmp_path / "ext"
    (extract / "flask-main" / ".git").mkdir(parents=True)
    (extract / "flask-main" / "main.py").write_text("x = 1\n", encoding="utf-8")
    assert find_repo_root(extract) == extract / "flask-main"


def test_find_repo_root_missing_git_raises(tmp_path: Path) -> None:
    extract = tmp_path / "ext"
    extract.mkdir()
    (extract / "main.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ZipValidationError, match=r"\.git"):
        find_repo_root(extract)


def test_find_repo_root_subdir_without_git_raises(tmp_path: Path) -> None:
    """Tek altdir olsa bile .git/ yoksa hata."""
    extract = tmp_path / "ext"
    (extract / "proj" / "src").mkdir(parents=True)
    (extract / "proj" / "main.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ZipValidationError, match=r"\.git"):
        find_repo_root(extract)


# ── safe_extract — entegre uctan uca ─────────────────────────────

def test_safe_extract_happy_with_subdir(tmp_path: Path) -> None:
    zip_p = tmp_path / "repo.zip"
    _make_zip(zip_p, {
        "myrepo/main.py":   "x = 1\n",
        "myrepo/.git/HEAD": "ref: refs/heads/main\n",
    })
    dest = tmp_path / "out"
    root = safe_extract(zip_p, dest)
    assert root.name == "myrepo"
    assert (root / "main.py").is_file()
    assert (root / ".git" / "HEAD").is_file()


def test_safe_extract_happy_at_root(tmp_path: Path) -> None:
    zip_p = tmp_path / "repo.zip"
    _make_zip(zip_p, {
        "main.py":   "x = 1\n",
        ".git/HEAD": "ref: refs/heads/main\n",
    })
    dest = tmp_path / "out"
    root = safe_extract(zip_p, dest)
    assert root == dest
    assert (root / "main.py").is_file()


def test_safe_extract_no_git_raises(tmp_path: Path) -> None:
    zip_p = tmp_path / "no_git.zip"
    _make_zip(zip_p, {
        "myrepo/main.py": "x = 1\n",
        "myrepo/README":  "no git here\n",
    })
    dest = tmp_path / "out"
    with pytest.raises(ZipValidationError, match=r"\.git"):
        safe_extract(zip_p, dest)


def test_safe_extract_path_traversal_aborts_before_write(tmp_path: Path) -> None:
    """validate_zip_metadata fail edince extractall hic cagrilmaz."""
    zip_p = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_p, "w") as zf:
        zf.writestr("../oops.py", "evil\n")
    dest = tmp_path / "out"
    with pytest.raises(ZipValidationError):
        safe_extract(zip_p, dest)
    # extractall cagrilmadan once raise olur -> dest bos ya da yok
    assert not (dest / ".." / "oops.py").exists()


# ── extract_local_meta ───────────────────────────────────────────

def test_extract_local_meta_no_git_returns_defaults(tmp_path: Path) -> None:
    """Git komutu yoksa veya repo gecersizse default'lar dondurulur."""
    repo = tmp_path / "norepo"
    repo.mkdir()
    info = extract_local_meta(repo, "myproject")
    assert info["full_name"] == "myproject"
    assert info["stars"] == 0
    assert info["contributor_count"] >= 1
    assert info["project_age_days"] >= 1
    assert info["topics"] == []
    assert "github" not in info.get("clone_url", "")


def test_extract_local_meta_uses_fallback_name(tmp_path: Path) -> None:
    """fallback_name parametresi full_name'e dusurulur."""
    repo = tmp_path / "x"
    repo.mkdir()
    info = extract_local_meta(repo, "explicit_name_123")
    assert info["full_name"] == "explicit_name_123"
