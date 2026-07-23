"""
zip_handler.py — Guvenli ZIP extract + .git/ zorunlu dogrulamasi.

Flask UI'da kullanici github URL yerine yerel bir repo'yu ZIP olarak
yukleyebilir (F7). Bu modul:

  1. ZIP metadata'sini ACMADAN dogrular:
     - Toplam decompressed boyut <= 500 MB
     - Dosya sayisi <= 50,000
     - Compression ratio <= 100x (zip bomb koruması)
     - Path traversal yok (..., mutlak yol)

  2. Validate edilince extract eder, repo kokunu (tek-altdir saran ya da
     dogrudan extract_dir) tespit eder.

  3. `.git/` dizini ZORUNLU — yoksa hata. Bug/commit tahmini git history'e
     dayanir, statik-only akis V2'de desteklenmiyor.

  4. `extract_local_meta()` — git log'dan minimum project_info uretir
     (GitHub API'sini taklit etmek icin).
"""
from __future__ import annotations

import logging
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)


# ── Limitler (F7 tasarim kararı) ──────────────────────────────────
MAX_DECOMPRESSED_SIZE: Final[int] = 500 * 1024 * 1024   # 500 MB
MAX_FILE_COUNT:        Final[int] = 50_000
MAX_COMPRESSION_RATIO: Final[int] = 100                  # zip bomb koruması
EXTRACT_TIMEOUT_SECS:  Final[int] = 30


class ZipValidationError(Exception):
    """ZIP icerigi limitleri asti veya yapisal sorun var."""


# ── ZIP metadata dogrulama ────────────────────────────────────────

def validate_zip_metadata(zip_path: Path) -> tuple[int, int]:
    """
    ZIP icerigini ACMADAN dogrula.

    Returns:
        (total_decompressed_bytes, file_count)
    Raises:
        ZipValidationError:
            * Dosya sayisi limiti
            * Decompressed boyut limiti
            * Compression ratio limiti (zip bomb)
            * Path traversal sehirli yol
        zipfile.BadZipFile: ZIP bozuk veya gecerli ZIP degil.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = zf.infolist()

        # 1. Dosya sayisi
        n = len(infos)
        if n > MAX_FILE_COUNT:
            raise ZipValidationError(
                f"Too many files in the ZIP: {n:,} (limit: {MAX_FILE_COUNT:,})."
            )

        # 2. Decompressed toplam boyut
        total_size = sum(info.file_size for info in infos)
        if total_size > MAX_DECOMPRESSED_SIZE:
            raise ZipValidationError(
                f"Decompressed size {total_size / 1024 / 1024:.1f} MB "
                f"exceeds the limit ({MAX_DECOMPRESSED_SIZE / 1024 / 1024:.0f} MB)."
            )

        # 3. Compression ratio (zip bomb koruması)
        compressed_size = sum(info.compress_size for info in infos)
        if compressed_size > 0:
            ratio = total_size / compressed_size
            if ratio > MAX_COMPRESSION_RATIO:
                raise ZipValidationError(
                    f"Compression ratio {ratio:.0f}x is suspicious (zip bomb?). "
                    f"Limit: {MAX_COMPRESSION_RATIO}x."
                )

        # 4. Path traversal — mutlak yol veya ".." iceren entry'ler
        for info in infos:
            name = info.filename
            if not name:
                continue
            # Mutlak yol (Windows ya da Unix)
            if name.startswith("/") or name.startswith("\\") or (len(name) > 1 and name[1] == ":"):
                raise ZipValidationError(f"Absolute path entry (path traversal): {name!r}")
            # ".." parcasi
            parts = Path(name).parts
            if any(p == ".." for p in parts):
                raise ZipValidationError(f"'..' entry (path traversal): {name!r}")

    return total_size, n


# ── Extract + repo koku tespiti ───────────────────────────────────

def find_repo_root(extract_dir: Path) -> Path:
    """
    Extract sonrasi repo'nun gercek kokunu bul.

    ZIP'ler genelde top-level bir dizinin altinda paketlenir
    (orn. `flask-main/`). Bu durumda repo koku o alt dizindir.

    Returns:
        `.git/` iceren Path.
    Raises:
        ZipValidationError: `.git/` hicbir yerde bulunmuyorsa.
    """
    # Once dogrudan extract_dir
    if (extract_dir / ".git").is_dir():
        return extract_dir

    # Top-level icindeki tek alt dizine bak (GitHub ZIP'leri boyle)
    subdirs = [p for p in extract_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1 and (subdirs[0] / ".git").is_dir():
        return subdirs[0]

    raise ZipValidationError(
        "No '.git/' directory was found in the ZIP. Defect and commit prediction "
        "rely on git history. Please zip the repository including '.git/': "
        "`cd repo && zip -r ../repo.zip .`"
    )


def safe_extract(zip_path: Path, dest_dir: Path) -> Path:
    """
    Validate edip ZIP'i `dest_dir`'e ac, repo kokunu dondur.

    Args:
        zip_path: Yuklenmis ZIP dosyasi.
        dest_dir: Bos hedef dizin (varsa olusur).

    Returns:
        Repo koku (`.git/` iceren Path).

    Raises:
        ZipValidationError, zipfile.BadZipFile, OSError
    """
    validate_zip_metadata(zip_path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    return find_repo_root(dest_dir)


# ── Yerel repo'dan project_info (git log'dan tureyen) ─────────────

def extract_local_meta(repo_path: Path, fallback_name: str) -> dict:
    """
    Yerel repo'dan GitHub API'sini taklit eden project_info dictionary uret.

    `get_project_info(github_url)` ile ayni anahtarlar — analyzer.py
    fark gormez. Bilinmeyen alanlar mantikli default'larla doldurulur:
      stars=0, default_branch='HEAD', description='', topics=[]

    contributor_count + project_age_days `git log`'dan hesaplanir.
    """
    info: dict = {
        "full_name":         fallback_name,
        "stars":             0,
        "contributor_count": 1,
        "project_age_days":  1,
        "default_branch":    "HEAD",
        "description":       "",
        "topics":            [],
        "created_at":        "",
        "clone_url":         "",  # yerel repo'da yok
    }

    # Contributor sayisi (unique author)
    try:
        r = subprocess.run(
            ["git", "log", "--format=%aN"],
            cwd=str(repo_path),
            capture_output=True, text=True, timeout=15,
            encoding="utf-8", errors="replace",
        )
        if r.returncode == 0:
            authors = {a.strip() for a in r.stdout.splitlines() if a.strip()}
            if authors:
                info["contributor_count"] = len(authors)
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("git log --format=%%aN failed: %s", exc)

    # Project age + created_at (ilk commit zamani)
    try:
        r = subprocess.run(
            ["git", "log", "--reverse", "--format=%at"],
            cwd=str(repo_path),
            capture_output=True, text=True, timeout=15,
            encoding="utf-8", errors="replace",
        )
        if r.returncode == 0:
            lines = r.stdout.strip().splitlines()
            if lines:
                first_ts = int(lines[0])
                first_dt = datetime.fromtimestamp(first_ts)
                info["project_age_days"] = max((datetime.now() - first_dt).days, 1)
                info["created_at"] = first_dt.isoformat()
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        logger.debug("git log --format=%%at failed: %s", exc)

    return info
