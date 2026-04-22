"""
rate_limit.py — GitHub API rate limit guard + token yardimcilari.

API contract (PLAN §13.2):

    guarded_get(url, **kwargs) -> requests.Response
    current_quota()            -> dict

Davranis:
- Her cagri oncesi X-RateLimit-Remaining takibi; esik altindaysa
  reset_at'a kadar sleep.
- 403 rate-limit cevabinda exponential backoff (5/15/45/135 sn).
- 429 throttle'da 60sn sleep.
- 5xx'de linear backoff (2/5/10 sn).
- Diger HTTP durumlarinda response ham olarak caller'a doner.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import requests

from pipeline.config import (
    GITHUB_ACCEPT,
    GITHUB_API_VERSION,
    GITHUB_RATELIMIT_URL,
    HTTP_TIMEOUT_SECONDS,
    RATE_LIMIT_BACKOFF,
    RATE_LIMIT_MAX_RETRIES,
    RATE_LIMIT_MIN_REMAINING,
    RATE_LIMIT_THROTTLE_SLEEP,
    SERVER_ERROR_BACKOFF,
    github_token,
)

logger = logging.getLogger(__name__)


# ── Token yardimcilari ────────────────────────────────────────────

def github_headers() -> dict[str, str]:
    """
    Guncel token ile GitHub API header'larini hazirla. Token yoksa
    sadece Accept header'i donulur (cagrilar daha yavas rate limit'e tabi).
    """
    headers: dict[str, str] = {
        "Accept": GITHUB_ACCEPT,
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    token = github_token()
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def github_token_configured() -> bool:
    """Token ayarli mi? UI ve log'lar icin kullanilir."""
    return bool(github_token())


# ── Quota takibi ──────────────────────────────────────────────────

# Son gorulen rate-limit bilgisi (header'lardan guncellenir).
_last_quota: dict[str, Any] = {
    "remaining": None,
    "limit":     None,
    "reset_at":  None,   # datetime | None
}


def _update_quota_from_headers(resp: requests.Response) -> None:
    """Response header'larindan _last_quota'yi gunceller."""
    remaining = resp.headers.get("X-RateLimit-Remaining")
    limit     = resp.headers.get("X-RateLimit-Limit")
    reset_ts  = resp.headers.get("X-RateLimit-Reset")

    if remaining is not None and remaining.isdigit():
        _last_quota["remaining"] = int(remaining)
    if limit is not None and limit.isdigit():
        _last_quota["limit"] = int(limit)
    if reset_ts is not None and reset_ts.isdigit():
        _last_quota["reset_at"] = datetime.fromtimestamp(int(reset_ts), tz=timezone.utc)


def current_quota() -> dict[str, Any]:
    """Son gorulen rate limit bilgisini dondur."""
    return dict(_last_quota)


def _sleep_until_reset(reset_at: Optional[datetime]) -> None:
    """Verilen reset zamanina kadar sleep; +1 sn toleransla."""
    if reset_at is None:
        time.sleep(RATE_LIMIT_THROTTLE_SLEEP)
        return
    now = datetime.now(timezone.utc)
    wait = max((reset_at - now).total_seconds(), 0) + 1
    logger.warning("GitHub rate limit dustu, %.0fsn sleep (reset=%s)", wait, reset_at.isoformat())
    time.sleep(wait)


# ── Ana API ───────────────────────────────────────────────────────

def guarded_get(url: str, **kwargs: Any) -> requests.Response:
    """
    requests.get wrapper. Rate limit, 429 throttle, 5xx gibi durumlari
    sessizce halleder; diger HTTP durumlarinda response caller'a doner.

    kwargs:
        - headers: mevcut header'lara merge edilir (GitHub default'lari korunur)
        - timeout: HTTP_TIMEOUT_SECONDS varsayilani
        - diger tum requests.get parametreleri

    Raises:
        requests.RequestException: Network hatasi 3 denemeyi asarsa
    """
    merged_headers = github_headers()
    user_headers = kwargs.pop("headers", None)
    if user_headers:
        merged_headers.update(user_headers)
    timeout = kwargs.pop("timeout", HTTP_TIMEOUT_SECONDS)

    # Ilk kontrol: proaktif olarak remaining<esik ise reset'e kadar bekle.
    _proactive_wait_if_low()

    forbidden_attempts = 0
    throttle_attempts = 0
    server_attempts = 0

    while True:
        try:
            resp = requests.get(url, headers=merged_headers, timeout=timeout, **kwargs)
        except requests.RequestException as exc:
            # Network hatasinda da linear backoff + retry
            if server_attempts >= RATE_LIMIT_MAX_RETRIES:
                logger.error("Network hatasi %d deneme sonrasi basarisiz: %s", server_attempts, exc)
                raise
            wait = SERVER_ERROR_BACKOFF[min(server_attempts, len(SERVER_ERROR_BACKOFF) - 1)]
            logger.warning("Network hatasi, %dsn sonra retry: %s", wait, exc)
            time.sleep(wait)
            server_attempts += 1
            continue

        _update_quota_from_headers(resp)

        # 403 — Rate limit mi yoksa baska bir forbidden mi?
        if resp.status_code == 403 and _looks_like_rate_limit(resp):
            if forbidden_attempts >= RATE_LIMIT_MAX_RETRIES:
                logger.error("403 rate limit, %d deneme sonrasi hala limit; response donuyor", forbidden_attempts)
                return resp
            # Reset zamanini goruyorsak oraya kadar uyu; degilse ussal backoff
            reset_at = _last_quota.get("reset_at")
            if reset_at is not None:
                _sleep_until_reset(reset_at)
            else:
                wait = RATE_LIMIT_BACKOFF[min(forbidden_attempts, len(RATE_LIMIT_BACKOFF) - 1)]
                logger.warning("403 rate limit, %dsn backoff", wait)
                time.sleep(wait)
            forbidden_attempts += 1
            continue

        if resp.status_code == 429:
            if throttle_attempts >= RATE_LIMIT_MAX_RETRIES:
                return resp
            logger.warning("429 throttle, %dsn sleep", RATE_LIMIT_THROTTLE_SLEEP)
            time.sleep(RATE_LIMIT_THROTTLE_SLEEP)
            throttle_attempts += 1
            continue

        if 500 <= resp.status_code < 600:
            if server_attempts >= RATE_LIMIT_MAX_RETRIES:
                return resp
            wait = SERVER_ERROR_BACKOFF[min(server_attempts, len(SERVER_ERROR_BACKOFF) - 1)]
            logger.warning("HTTP %d, %dsn sonra retry", resp.status_code, wait)
            time.sleep(wait)
            server_attempts += 1
            continue

        # 2xx, 4xx (rate-limit disi): caller'a birak
        return resp


def _looks_like_rate_limit(resp: requests.Response) -> bool:
    """403 response'u rate limit kaynakli mi degerlendirir."""
    # Header'da remaining=0 varsa kesin rate limit
    remaining = resp.headers.get("X-RateLimit-Remaining")
    if remaining is not None and remaining.isdigit() and int(remaining) == 0:
        return True
    # Body'de 'rate limit' gecer mi?
    try:
        body = resp.json()
    except ValueError:
        return False
    if not isinstance(body, dict):
        return False
    message = str(body.get("message", "")).lower()
    return "rate limit" in message or "abuse" in message


def _proactive_wait_if_low() -> None:
    """remaining esigin altindaysa reset'e kadar proaktif bekle."""
    remaining = _last_quota.get("remaining")
    if remaining is None:
        return
    if remaining >= RATE_LIMIT_MIN_REMAINING:
        return
    _sleep_until_reset(_last_quota.get("reset_at"))


def refresh_quota() -> dict[str, Any]:
    """
    /rate_limit endpoint'ine tek cagri yap — header bilgilerini gunceller,
    current_quota()'yu doldurur. Token durumunu sinamak icin kullanislidir.
    """
    try:
        resp = requests.get(
            GITHUB_RATELIMIT_URL,
            headers=github_headers(),
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        _update_quota_from_headers(resp)
        if resp.status_code == 200:
            payload = resp.json()
            core = payload.get("resources", {}).get("core", {})
            if "remaining" in core:
                _last_quota["remaining"] = int(core["remaining"])
            if "limit" in core:
                _last_quota["limit"] = int(core["limit"])
            if "reset" in core:
                _last_quota["reset_at"] = datetime.fromtimestamp(
                    int(core["reset"]), tz=timezone.utc
                )
    except requests.RequestException as exc:
        logger.warning("quota refresh basarisiz: %s", exc)
    return current_quota()
