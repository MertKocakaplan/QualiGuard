"""
checkpoint.py — Faz/proje checkpoint yonetimi.

API contract (PLAN §13.1):

    save_checkpoint(phase, data)      -> None   (atomic write)
    load_checkpoint(phase)            -> dict | None
    mark_project_done(name, result)   -> None
    is_project_done(name)             -> bool
    get_processed_set()               -> set[str]   (sadece status=ok)

Tum yazimlar temp dosya + os.replace ile atomic'tir, boylece kesinti
anlarinda checkpoint dosyalari corruption yasamaz.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from pipeline.config import CHECKPOINT_DIR

logger = logging.getLogger(__name__)

# processed_projects checkpoint'i icin ayri kilit; save/load paralel
# cagrildiginda yaris olmasin.
_processed_lock = threading.Lock()

_PROCESSED_FILE = "processed_projects"


# ── Temel I/O ─────────────────────────────────────────────────────

def _checkpoint_path(phase: str) -> Path:
    """Verilen faz icin checkpoint dosya yolu."""
    safe = phase.strip().replace("/", "_").replace("\\", "_")
    if not safe:
        raise ValueError("phase bos olamaz.")
    return CHECKPOINT_DIR / f"{safe}.json"


def _atomic_write_json(path: Path, data: Any) -> None:
    """
    Temp dosyaya yaz, fsync, sonra os.replace ile hedefe tasi.
    Boylece yazim ortasinda kesinti olursa hedef dosya bozulmaz.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # tempfile ayni dizinde olmali — os.replace cross-device atomic degil.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        # Basarisiz olursa temp dosyayi temizle
        tmp_path.unlink(missing_ok=True)
        raise


def save_checkpoint(phase: str, data: dict) -> None:
    """
    Faz checkpoint'ini atomic olarak diske yaz.

    Args:
        phase: checkpoint adi (ornegin "discovery", "phase_status")
        data:  JSON-serializable dict
    """
    path = _checkpoint_path(phase)
    _atomic_write_json(path, data)
    logger.debug("checkpoint yazildi: %s", path.name)


def load_checkpoint(phase: str) -> dict | None:
    """
    Faz checkpoint'ini oku. Dosya yoksa None.

    Bozuk JSON'da logger.warning + None donulur (pipeline temiz
    baslangic yapabilsin).
    """
    path = _checkpoint_path(phase)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("checkpoint bozuk/okunamaz: %s (%s)", path.name, exc)
        return None


# ── processed_projects yonetimi ──────────────────────────────────

def _load_processed_raw() -> dict:
    """processed_projects checkpoint'ini sozluk olarak al."""
    data = load_checkpoint(_PROCESSED_FILE)
    if data is None or "processed" not in data:
        return {"processed": {}}
    return data


def mark_project_done(project_name: str, result: dict) -> None:
    """
    Bir proje islendiginde processed_projects'e ekler.

    Args:
        project_name: "user/repo" formatinda
        result: {"status": "ok"|"failed", ...}  — §14.4 semasi
    """
    if not project_name:
        raise ValueError("project_name bos olamaz.")
    with _processed_lock:
        data = _load_processed_raw()
        data["processed"][project_name] = result
        save_checkpoint(_PROCESSED_FILE, data)


def is_project_done(project_name: str) -> bool:
    """
    Proje daha once basariyla islenmis mi?

    Sadece status=='ok' olanlari true sayar — failed projelere tekrar
    denenebilsin. Eger stricter davranis istenirse caller
    `get_processed_set()` uzerinde istedigi gibi filtreleyebilir.

    Lock altinda okunur; concurrent mark_project_done ile yarismaz.
    """
    with _processed_lock:
        data = _load_processed_raw()
        entry = data["processed"].get(project_name)
    return bool(entry and entry.get("status") == "ok")


def get_processed_set() -> set[str]:
    """
    Status='ok' olan proje isimlerinin kumesini dondurur.

    Lock altinda okunur; concurrent mark_project_done ile yarismaz.
    """
    with _processed_lock:
        data = _load_processed_raw()
    return {
        name
        for name, entry in data["processed"].items()
        if isinstance(entry, dict) and entry.get("status") == "ok"
    }
