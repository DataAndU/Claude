"""Zerodha Kite Connect session manager with TOTP auto-login.

Handles:
    1. Generating the login URL
    2. Auto-login via headless requests + TOTP when secrets are configured
    3. Exchanging request_token → access_token
    4. Storing and refreshing the access_token at runtime
    5. Providing a ready-to-use KiteConnect instance
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

import httpx
import pyotp
from kiteconnect import KiteConnect

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_kite: Optional[KiteConnect] = None
_access_token: Optional[str] = None


def get_kite() -> KiteConnect:
    """Return the singleton KiteConnect instance."""
    global _kite
    if _kite is None:
        _kite = KiteConnect(api_key=settings.kite_api_key)
        if _access_token:
            _kite.set_access_token(_access_token)
    return _kite


def get_login_url() -> str:
    return get_kite().login_url()


async def auto_login() -> str:
    """Fully automated login via TOTP — no browser needed.

    Flow:
        1. POST to Kite login with user_id + password → get request_id
        2. POST TOTP to 2FA endpoint → get session cookies
        3. GET Kite Connect login URL with session cookies → redirect with request_token
        4. Exchange request_token for access_token

    Returns the access_token string.
    """
    if not all([settings.kite_user_id, settings.kite_password, settings.kite_totp_secret]):
        raise RuntimeError(
            "Auto-login requires KITE_USER_ID, KITE_PASSWORD, and KITE_TOTP_SECRET in .env"
        )

    import urllib.parse

    kite = get_kite()
    login_url = "https://kite.zerodha.com/api/login"
    twofa_url = "https://kite.zerodha.com/api/twofa"
    connect_url = f"https://kite.trade/connect/login?v=3&api_key={settings.kite_api_key}"

    totp = pyotp.TOTP(settings.kite_totp_secret)
    request_token = ""

    # Use a cookie jar that sends cookies cross-domain (kite.zerodha.com → kite.trade)
    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
    ) as client:
        # ── Step 1: Login with user_id + password ──
        logger.info("Kite auto-login step 1: credentials...")
        resp1 = await client.post(
            login_url,
            data={
                "user_id": settings.kite_user_id,
                "password": settings.kite_password,
            },
        )
        resp1.raise_for_status()
        login_data = resp1.json().get("data", {})
        request_id = login_data.get("request_id", "")
        if not request_id:
            raise RuntimeError(f"Login step 1 failed — no request_id: {resp1.text}")

        logger.info("Kite auto-login step 1 OK (request_id=%s)", request_id[:8])

        # ── Step 2: Submit TOTP ──
        logger.info("Kite auto-login step 2: TOTP...")
        resp2 = await client.post(
            twofa_url,
            data={
                "user_id": settings.kite_user_id,
                "request_id": request_id,
                "twofa_value": totp.now(),
                "twofa_type": "totp",
            },
        )
        resp2.raise_for_status()

        # Check if twofa response directly contains request_token
        twofa_data = resp2.json().get("data", {})
        request_token = twofa_data.get("request_token", "")
        logger.info("Kite auto-login step 2 OK")

        # ── Step 3: Get request_token from Kite Connect redirect ──
        if not request_token:
            logger.info("Kite auto-login step 3: extracting request_token...")

            # Collect all cookies from login + twofa responses and pass to kite.trade
            # Cookies are on kite.zerodha.com domain; we must forward them manually
            all_cookies = {}
            for r in (resp1, resp2):
                for name, value in r.cookies.items():
                    all_cookies[name] = value

            resp3 = await client.get(connect_url, cookies=all_cookies)
            location = ""

            if resp3.status_code in (301, 302, 303, 307, 308):
                location = resp3.headers.get("location", "")
            else:
                location = str(resp3.url)

            # Follow up to 5 redirects manually, forwarding cookies
            for _ in range(5):
                if not location or "request_token" in location:
                    break
                # Merge any new cookies
                for name, value in resp3.cookies.items():
                    all_cookies[name] = value
                resp3 = await client.get(location, cookies=all_cookies)
                if resp3.status_code in (301, 302, 303, 307, 308):
                    location = resp3.headers.get("location", "")
                else:
                    location = str(resp3.url)

            # Parse request_token from the final redirect URL
            if location:
                parsed = urllib.parse.urlparse(location)
                params = urllib.parse.parse_qs(parsed.query)
                request_token = params.get("request_token", [""])[0]

            # Fallback: check response body for request_token
            if not request_token and resp3.status_code == 200:
                body = resp3.text
                if "request_token" in body:
                    import re
                    m = re.search(r'request_token["\s:=]+([a-zA-Z0-9]+)', body)
                    if m:
                        request_token = m.group(1)

        if not request_token:
            raise RuntimeError(
                "Could not extract request_token from Kite login flow. "
                "Check credentials or try manual login via /api/v1/kite/login-url"
            )

        logger.info("Kite auto-login: got request_token=%s...", request_token[:8])

    return await set_request_token(request_token)


async def set_request_token(request_token: str) -> str:
    """Exchange request_token for access_token and store it."""
    global _access_token
    kite = get_kite()
    data = kite.generate_session(request_token, api_secret=settings.kite_api_secret)
    _access_token = data["access_token"]
    kite.set_access_token(_access_token)
    logger.info("Kite session active for user %s", settings.kite_user_id)
    return _access_token


def is_logged_in() -> bool:
    global _access_token
    if not _access_token:
        return False
    try:
        get_kite().profile()
        return True
    except Exception:
        _access_token = None
        return False


def get_access_token() -> Optional[str]:
    return _access_token
