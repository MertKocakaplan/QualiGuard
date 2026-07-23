"""
discovery.py — GitHub search + enrichment + contributor filter.

API contract (PLAN §13.3):

    search_projects(target_count, min_age_days, max_age_days,
                    max_contributors, min_stars) -> list[dict]

Yaklasim:
- GitHub search API tek sorguda 1000 sonuc dondurur.
- Yas araligini sliding window ile daraltarak hedef sayiya ulas.
- Her aday icin /repos/{full_name}/contributors ile enrichment —
  contributor_count <= max_contributors filtresi uygulanir.
- Her DISCOVERY_FLUSH_EVERY projede discovery.json'a flush.

Flask analyzer.py'nin tek-proje cagrisi icin `get_project_info()` de
burada tutulur — hem kullanici URL'si analizi hem discovery ayni
enrichment yolunu kullanir.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from pipeline import checkpoint
from pipeline.config import (
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_AGE_DAYS,
    DEFAULT_MAX_CONTRIBUTORS,
    DEFAULT_MIN_AGE_DAYS,
    DEFAULT_MIN_STARS,
    DEFAULT_TARGET_COUNT,
    DISCOVERY_FLUSH_EVERY,
    GITHUB_REPO_URL,
    GITHUB_SEARCH_MAX_PER_QUERY,
    GITHUB_SEARCH_PER_PAGE,
    GITHUB_SEARCH_URL,
)
from pipeline.rate_limit import guarded_get

logger = logging.getLogger(__name__)


# ── Yardimcilar ──────────────────────────────────────────────────

def _iso_date(days_ago: int) -> str:
    """Bugunden N gun once icin YYYY-MM-DD formatinda tarih."""
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.strftime("%Y-%m-%d")


def _calc_age_days(created_at_iso: str) -> int:
    """ISO8601 created_at -> bugune kadar gecen gun."""
    try:
        created = datetime.fromisoformat(created_at_iso.replace("Z", "+00:00"))
        return max((datetime.now(timezone.utc) - created).days, 1)
    except (ValueError, AttributeError):
        return 365


def _contributor_count(full_name: str) -> int:
    """
    Hizli contributor sayisi tahmini.

    Link header'daki rel=last sayfa numarasindan okunur — N istek yerine 1.
    Fallback: liste uzunlugu (cok fazla contributor olsa bile hizli).
    """
    resp = guarded_get(
        f"{GITHUB_REPO_URL}/{full_name}/contributors",
        params={"per_page": 1, "anon": "true"},
    )
    if resp.status_code != 200:
        return 1
    link = resp.headers.get("Link", "")
    if 'rel="last"' in link:
        m = re.search(r"[&?]page=(\d+)>; rel=\"last\"", link)
        if m:
            return int(m.group(1))
    try:
        body = resp.json()
    except ValueError:
        return 1
    return max(len(body) if isinstance(body, list) else 1, 1)


def _extract_full_name(github_url: str) -> Optional[str]:
    """https://github.com/user/repo(.git)? -> "user/repo" veya None."""
    m = re.search(r"github\.com/([^/]+/[^/]+?)(?:\.git)?/?$", github_url)
    return m.group(1).rstrip("/") if m else None


# ── Flask tek-proje API'si ──────────────────────────────────────

def get_project_info(github_url: str) -> dict:
    """
    Kullanicinin girdigi URL icin proje bilgilerini cek (Flask analyzer
    yolu). V1 git_utils.get_project_info() ile uyumlu donus semasi.

    Returns:
        {"stars": int, "contributor_count": int, "project_age_days": int}

    Raises:
        RuntimeError: 403 rate limit veya 404 not found.
    """
    full_name = _extract_full_name(github_url)
    if not full_name:
        return {"stars": 0, "contributor_count": 1, "project_age_days": 365}

    resp = guarded_get(f"{GITHUB_REPO_URL}/{full_name}")
    if resp.status_code == 403:
        remaining = resp.headers.get("X-RateLimit-Remaining", "?")
        raise RuntimeError(
            f"GitHub API rate limit exceeded (remaining: {remaining}). "
            "Set a GITHUB_TOKEN."
        )
    if resp.status_code == 404:
        raise RuntimeError(f"Repository not found: {full_name}")
    if resp.status_code != 200:
        raise RuntimeError(f"GitHub API error: HTTP {resp.status_code}")

    try:
        data = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"Could not parse the GitHub response: {exc}") from exc

    return {
        "stars":             int(data.get("stargazers_count", 0)),
        "contributor_count": _contributor_count(full_name),
        "project_age_days":  _calc_age_days(data.get("created_at", "")),
    }


# ── Discovery ────────────────────────────────────────────────────

def _build_query(
    min_stars: int,
    min_age_days: int,
    max_age_days: int,
    language: str,
) -> str:
    """GitHub search qualifier query string'i olustur."""
    max_date = _iso_date(min_age_days)   # created_at bu tarih-veya-oncesi
    min_date = _iso_date(max_age_days)   # created_at bu tarih-veya-sonrasi
    return (
        f"language:{language} "
        f"stars:>={min_stars} "
        f"created:{min_date}..{max_date} "
        f"is:public archived:false fork:false"
    )


def _search_page(query: str, page: int) -> list[dict]:
    """Tek search sayfasi (max 100 sonuc)."""
    resp = guarded_get(
        GITHUB_SEARCH_URL,
        params={
            "q":        query,
            "sort":     "stars",
            "order":    "desc",
            "per_page": GITHUB_SEARCH_PER_PAGE,
            "page":     page,
        },
    )
    if resp.status_code != 200:
        logger.warning("search HTTP %d (q=%s, page=%d)", resp.status_code, query, page)
        return []
    try:
        payload = resp.json()
    except ValueError:
        return []
    return list(payload.get("items", []))


def _iter_search_results(query: str, per_query_limit: int = GITHUB_SEARCH_MAX_PER_QUERY):
    """Bir query icin max `per_query_limit` sonuca kadar yield."""
    pages = (per_query_limit + GITHUB_SEARCH_PER_PAGE - 1) // GITHUB_SEARCH_PER_PAGE
    yielded = 0
    for page in range(1, pages + 1):
        items = _search_page(query, page)
        if not items:
            return
        for item in items:
            yield item
            yielded += 1
            if yielded >= per_query_limit:
                return


def search_projects(
    target_count: int = DEFAULT_TARGET_COUNT,
    min_age_days: int = DEFAULT_MIN_AGE_DAYS,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    max_contributors: int = DEFAULT_MAX_CONTRIBUTORS,
    min_stars: int = DEFAULT_MIN_STARS,
    language: str = DEFAULT_LANGUAGE,
) -> list[dict]:
    """
    Kriterlere uyan projeleri GitHub search API ile topla.

    Donus her bir item (PLAN §13.3):
        {
            'full_name': 'user/repo',
            'clone_url': 'https://github.com/user/repo.git',
            'stars': int,
            'created_at': 'iso8601',
            'project_age_days': int,
            'contributor_count': int,
            'default_branch': str,
        }

    Strateji:
        - Yas araligini adim adim kaydirarak max 1000 kisitini as.
        - Her aday icin contributor enrichment.
        - DISCOVERY_FLUSH_EVERY'de checkpoint flush.
    """
    if target_count <= 0:
        return []

    logger.info(
        "discovery basladi: target=%d, age=[%d,%d], max_contrib=%d, min_stars=%d",
        target_count, min_age_days, max_age_days, max_contributors, min_stars,
    )

    found: list[dict] = []
    seen: set[str] = set()

    # Onceki discovery varsa kaldigimiz yerden devam et
    prior = checkpoint.load_checkpoint("discovery")
    if prior and isinstance(prior.get("found"), list):
        for item in prior["found"]:
            name = item.get("full_name")
            if name and name not in seen:
                seen.add(name)
                found.append(item)
        logger.info("discovery checkpoint: %d proje onceden bulunmus", len(found))
        if len(found) >= target_count:
            return found[:target_count]

    criteria = {
        "min_age_days":     min_age_days,
        "max_age_days":     max_age_days,
        "max_contributors": max_contributors,
        "min_stars":        min_stars,
        "language":         language,
    }

    def _flush() -> None:
        checkpoint.save_checkpoint("discovery", {
            "started_at":  datetime.now(timezone.utc).isoformat(),
            "criteria":    criteria,
            "target_count": target_count,
            "found_count": len(found),
            "found":       found,
        })

    # Yas penceresini birer haftada kaydirarak gezeriz (max 1000/query limitini as)
    window_days = 7
    cur_max = max_age_days
    cur_min = max(min_age_days, cur_max - window_days)

    while len(found) < target_count and cur_max > 0 and cur_min >= min_age_days:
        query = _build_query(min_stars, cur_min, cur_max, language)
        logger.info("query: %s (found=%d)", query, len(found))

        for item in _iter_search_results(query):
            full_name = item.get("full_name")
            if not full_name or full_name in seen:
                continue
            seen.add(full_name)

            # Hizli contributor filtresi
            try:
                contributors = _contributor_count(full_name)
            except Exception as exc:
                logger.warning("contributor enrichment failed (%s): %s", full_name, exc)
                continue
            if contributors > max_contributors:
                continue

            created_at = item.get("created_at", "")
            # F4 (filter_categorize) topics + description bekliyor; GitHub search
            # API bu alanlari zaten donuyor — sadece extract ediyoruz.
            # Olmadigi takdirde kategorizasyon project_name'a duser ve cogu
            # proje "Diger" olur (V1 ile karsilastirildiginda gorulen sorun).
            topics_raw = item.get("topics") or []
            found.append({
                "full_name":         full_name,
                "clone_url":         item.get("clone_url") or f"https://github.com/{full_name}.git",
                "stars":             int(item.get("stargazers_count", 0)),
                "created_at":        created_at,
                "project_age_days":  _calc_age_days(created_at),
                "contributor_count": contributors,
                "default_branch":    item.get("default_branch") or "main",
                "topics":            [str(t) for t in topics_raw if t],
                "description":       (item.get("description") or "").strip(),
            })

            if len(found) % DISCOVERY_FLUSH_EVERY == 0:
                _flush()
                logger.info("checkpoint flush: found=%d", len(found))

            if len(found) >= target_count:
                break

        if cur_min == min_age_days:
            break
        cur_max = cur_min - 1
        cur_min = max(min_age_days, cur_max - window_days)

    _flush()
    # Son durum: completed_at ekle
    data = checkpoint.load_checkpoint("discovery") or {}
    data["completed_at"] = datetime.now(timezone.utc).isoformat()
    checkpoint.save_checkpoint("discovery", data)

    logger.info("discovery completed: %d projects", len(found))
    return found[:target_count]
