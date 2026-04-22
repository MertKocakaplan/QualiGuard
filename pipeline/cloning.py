"""
cloning.py — Git repo klonlama sarmalayicisi.

V1'deki app/git_utils.clone_repo() kismi buraya tasindi (PLAN §3.3).

SZZ kullanmak icin full history gerekir; yine de --single-branch
bandwidth kazanir. Timeout asimi veya diger hatalar None,mesaj seklinde
caller'a doner (her proje bagimsiz, pipeline devam eder).
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from pipeline.config import CLONE_SINGLE_BRANCH, CLONE_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


def safe_repo_name(url: str) -> str:
    """
    URL'den dosya sistemi icin guvenli klasor adi uret.

    Edge-case'ler: trailing slash, .git uzantisi, aciklamasiz bos uc.
    """
    clean = url.rstrip("/")
    if clean.endswith(".git"):
        clean = clean[:-4]
    raw  = clean.split("/")[-1]
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", raw).strip("_-")
    return name if name else "repo"


def clone_repo(url: str, target_dir: Path) -> tuple[Optional[Path], str]:
    """
    Git repo'yu target_dir icine klonla.

    Args:
        url:        "https://github.com/user/repo[.git]" gibi bir URL
        target_dir: Klonun icine yapilacagi ebeveyn dizin

    Returns:
        (repo_path, status) — hata durumunda (None, mesaj).
        status degerleri:
            "basarili"  — yeni klon
            "zaten_var" — hedefde gecerli klon bulundu
    """
    name      = safe_repo_name(url)
    repo_path = target_dir / name

    # Hedefte gecerli klon varsa atla
    if repo_path.exists() and (repo_path / ".git").exists():
        return repo_path, "zaten_var"

    # Yarim kalan klasor kalintisi varsa temizle
    if repo_path.exists():
        shutil.rmtree(repo_path, ignore_errors=True)

    target_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["git", "clone"]
    if CLONE_SINGLE_BRANCH:
        cmd.append("--single-branch")
    cmd.extend([url, str(repo_path)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=CLONE_TIMEOUT_SECONDS,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(repo_path, ignore_errors=True)
        return None, f"Klonlama zaman asimina ugradi ({CLONE_TIMEOUT_SECONDS}s)."
    except FileNotFoundError:
        return None, "git komutu bulunamadi. Git kurulu ve PATH'te mi?"
    except OSError as exc:
        shutil.rmtree(repo_path, ignore_errors=True)
        return None, f"Klonlama hatasi: {str(exc)[:200]}"

    if result.returncode != 0:
        err = result.stderr.strip()[:300]
        shutil.rmtree(repo_path, ignore_errors=True)
        return None, f"Git clone hatasi: {err}"

    if not (repo_path / ".git").exists():
        return None, "Klonlama tamamlandi ancak .git bulunamadi."

    return repo_path, "basarili"
