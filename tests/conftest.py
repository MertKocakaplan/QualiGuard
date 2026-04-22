"""
conftest.py — pytest ortak fixture'lar.

`CHECKPOINT_DIR`'i her testte izole bir tmp_path'e yonlendir;
testler birbirinin dosyalarina dokunmasin.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Proje kokunu sys.path'e ekle — `from pipeline import ...` calismasi icin
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def isolated_checkpoint_dir(tmp_path, monkeypatch):
    """checkpoint.py'nin diske yazdigi yolu tmp_path'e yonlendir."""
    from pipeline import checkpoint, config  # noqa: WPS433

    tmp_cp = tmp_path / "checkpoints"
    tmp_cp.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_cp, raising=False)
    monkeypatch.setattr(checkpoint, "CHECKPOINT_DIR", tmp_cp, raising=False)
    return tmp_cp
