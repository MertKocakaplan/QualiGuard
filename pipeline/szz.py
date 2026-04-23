"""
szz.py — SZZ (Sliwerski-Zimmermann-Zeller) bug-introducing commit tespiti.

PLAN §3.6 ve §13.4'e gore pydriller ile bug-fix commit'lerinden geri donerek
hangi HEAD dosyalarinin tarihsel olarak "buggy" oldugunu etiketler.

Algoritma (pydriller `get_commits_last_modified_lines` sarmalayicisi):

    1. Her bug-fix commit icin pydriller SZZ cagrilir.
    2. Donen `{fix_path: {intro_commit_hashes}}` dict'inin anahtarlari,
       fix commit'inde silinen satirlari iceren dosyalardir — yani
       bug'in yasadigi dosyalar.
    3. Bu dosya yollari HEAD dosya listesiyle kesistirilerek etiketlenir.
    4. Timeout aldiginde bos dict doner; caller `bug_keyword`'e fallback.

Timeout davranisi:

    Python-seviye sinyal yok (Windows uyumlu kalmak icin). Her fix commit
    bittiginde `time.monotonic()` kontrolu yapilir; budget asilmissa islem
    biter ve BOS dict donulur — kismi etiket donmek istemiyoruz, caller
    baseline'a dusmeli.

Rename/move edilmis dosyalar:

    Fix commit'teki yol HEAD'de ayni olmayabilir. Basit cozum: sadece
    HEAD setinde bulunan yollari etiketle. Renamelar kacirilir; bu sade
    ve literaturde yaygin bir yaklasim.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from pydriller import Git

from pipeline.config import SZZ_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


def _normalize_path(p: str) -> str:
    """pydriller bazen ters slash donebilir; forward slash'a normalize et."""
    return p.replace("\\", "/")


def compute_szz_labels(
    repo_path: Path,
    head_files: list[str],
    bug_fix_commits: list[str],
    timeout_seconds: int = SZZ_TIMEOUT_SECONDS,
) -> dict[str, int]:
    """
    SZZ ile HEAD dosyalari icin bug etiketi hesapla.

    Args:
        repo_path:        Klonlanmis git repo yolu (full history gerekli).
        head_files:       HEAD'deki .py dosyalarinin repo-relative yollari.
        bug_fix_commits:  Bug-fix olarak isaretlenmis commit hash'leri.
        timeout_seconds:  SZZ zaman butcesi (saniye).

    Returns:
        {file_path: 0|1} — head_files'in her elemani icin etiket.
        Timeout veya kurtarilamaz hata durumunda bos dict; caller
        bu durumda `bug_keyword` etiketine fallback yapmalidir.
    """
    if not head_files:
        return {}

    head_set = {_normalize_path(f) for f in head_files}
    labels: dict[str, int] = {f: 0 for f in head_set}

    if not bug_fix_commits:
        return labels

    start = time.monotonic()

    try:
        git = Git(str(repo_path))
    except Exception as exc:  # pydriller yolu acamaz
        logger.warning("pydriller Git init basarisiz (%s): %s", repo_path, exc)
        return {}

    processed = 0
    try:
        for fix_hash in bug_fix_commits:
            if time.monotonic() - start > timeout_seconds:
                logger.warning(
                    "SZZ timeout (%ds asildi, %d/%d commit islendi) — fallback",
                    timeout_seconds, processed, len(bug_fix_commits),
                )
                return {}

            try:
                fix_commit = git.get_commit(fix_hash)
            except Exception as exc:  # hash gecersiz / silinmis
                logger.debug("SZZ get_commit(%s) hatali: %s", fix_hash, exc)
                processed += 1
                continue

            try:
                intro_map = git.get_commits_last_modified_lines(fix_commit)
            except Exception as exc:  # blame hatasi, binary dosya vs.
                logger.debug("SZZ blame hatali (%s): %s", fix_hash, exc)
                processed += 1
                continue

            for fix_path in intro_map.keys():
                normed = _normalize_path(fix_path)
                if normed in head_set:
                    labels[normed] = 1

            processed += 1
    finally:
        try:
            git.clear()
        except Exception:
            pass

    return labels
