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
        1. POST to Kite login endpoint with user_id + password
        2. POST TOTP to the 2FA endpoint
        3. Extract request_token from redirect
        4. Exchange for access_token via Kite API

    Returns the access_token string.
    """
    if not all([settings.kite_user_id, settings.kite_password, settings.kite_totp_secret]):
        raise RuntimeError(
            "Auto-login requires KITE_USER_ID, KITE_PASSWORD, and KITE_TOTP_SECRET in .env"
        )

    kite = get_kite()
    login_url = "https://kite.zerodha.com/api/login"
    twofa_url = "https://kite.zerodha.com/api/twofa"

    totp = pyotp.TOTP(settings.kite_totp_secret)

    async with httpx.AsyncClient(follow_redirects=False, timeout=30) as client:
        # Step 1: Login with user_id + password
        resp = await client.post(
            login_url,
            data={
                "user_id": settings.kite_user_id,
                "password": settings.kite_password,
            },
        )
        resp.raise_for_status()
        login_data = resp.json().get("data", {})
        request_id = login_data.get("request_id", "")
        if not request_id:
            raise RuntimeError(f"Login step 1 failed: {resp.text}")

        # Step 2: Submit TOTP
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

        # Step 3: Kite redirects to redirect_url?request_token=...&action=login
        # We follow the redirect manually from login URL
        redirect_url = f"https://kite.trade/connect/login?v=3&api_key={settings.kite_api_key}"
        resp3 = await client.get(redirect_url)
        # The final redirect URL contains the request_token
        if resp3.status_code in (301, 302, 303, 307, 308):
            location = resp3.headers.get("location", "")
        else:
            # Sometimes the token is in the URL directly
            location = str(resp3.url)

        # Extract request_token from redirect URL
        import urllib.parse
        parsed = urllib.parse.urlparse(location)
        params = urllib.parse.parse_qs(parsed.query)
        request_token = params.get("request_token", [""])[0]

        if not request_token:
            # Fallback: try from the 2FA response itself
            twofa_data = resp2.json().get("data", {})
            request_token = twofa_data.get("request_token", "")

        if not request_token:
            raise RuntimeError(
                "Could not extract request_token from Kite login flow. "
                "Try manual login via /api/v1/kite/login-url"
            )

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
