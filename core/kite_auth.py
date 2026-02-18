"""Zerodha Kite Connect session manager.

Supports three login methods:
    1. auto_login()   — fully automated via TOTP (headless, no browser)
    2. browser_login() — opens Kite login URL, captures callback
    3. token_login()   — user pastes request_token directly

All methods exchange the request_token for an access_token and
configure the singleton KiteConnect instance.
"""

from __future__ import annotations

import logging
import re
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import httpx
import pyotp
from kiteconnect import KiteConnect

from core.config import load_config

logger = logging.getLogger(__name__)

_kite: Optional[KiteConnect] = None
_access_token: Optional[str] = None

# Callback server state
_callback_token: Optional[str] = None
_callback_event = threading.Event()
CALLBACK_PORT = 5678


# ─────────────────────────────────────────────────────────────────────────────
# Singleton KiteConnect instance
# ─────────────────────────────────────────────────────────────────────────────

def get_kite() -> KiteConnect:
    """Return the singleton KiteConnect instance."""
    global _kite
    cfg = load_config()
    if _kite is None:
        _kite = KiteConnect(api_key=cfg.kite_api_key)
        if _access_token:
            _kite.set_access_token(_access_token)
    return _kite


def is_logged_in() -> bool:
    """Check if we have a valid Kite session."""
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


def get_login_url() -> str:
    cfg = load_config()
    return f"https://kite.trade/connect/login?v=3&api_key={cfg.kite_api_key}"


# ─────────────────────────────────────────────────────────────────────────────
# Login Method 1: Fully automatic TOTP (headless)
# ─────────────────────────────────────────────────────────────────────────────

async def auto_login() -> str:
    """Fully automated login via TOTP — no browser needed.

    Flow:
        1. Load Kite login page for cookies
        2. POST credentials to get request_id
        3. POST TOTP for 2FA
        4. Follow redirects to get request_token
        5. Exchange for access_token
    """
    cfg = load_config()
    if not all([cfg.kite_user_id, cfg.kite_password, cfg.kite_totp_secret]):
        raise RuntimeError(
            "Auto-login requires KITE_USER_ID, KITE_PASSWORD, and "
            "KITE_TOTP_SECRET in .env"
        )

    kite = get_kite()
    base_url = "https://kite.zerodha.com"
    login_url = f"{base_url}/api/login"
    twofa_url = f"{base_url}/api/twofa"
    connect_url = f"https://kite.trade/connect/login?v=3&api_key={cfg.kite_api_key}"

    totp = pyotp.TOTP(cfg.kite_totp_secret)

    browser_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Linux; Android 13; Pixel 7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Mobile Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": base_url,
        "Referer": f"{base_url}/connect/login",
        "X-Kite-Version": "3",
    }

    async with httpx.AsyncClient(
        follow_redirects=False, timeout=30, headers=browser_headers,
    ) as client:
        # Step 0: Load login page for session cookies
        logger.info("Kite auto-login: loading login page...")
        try:
            page_resp = await client.get(connect_url)
        except Exception:
            page_resp = await client.get(base_url)

        # Step 1: Credentials
        logger.info("Kite auto-login: submitting credentials...")
        resp1 = await client.post(login_url, data={
            "user_id": cfg.kite_user_id,
            "password": cfg.kite_password,
        })
        resp1.raise_for_status()
        login_data = resp1.json().get("data", {})
        request_id = login_data.get("request_id", "")
        if not request_id:
            raise RuntimeError(f"Login failed — no request_id: {resp1.text}")

        # Step 2: TOTP 2FA
        logger.info("Kite auto-login: submitting TOTP...")
        resp2 = await client.post(twofa_url, data={
            "user_id": cfg.kite_user_id,
            "request_id": request_id,
            "twofa_value": totp.now(),
            "twofa_type": "totp",
        })

        # Retry on 403 (timing issue with TOTP)
        if resp2.status_code == 403:
            logger.warning("TOTP 403, retrying with fresh code...")
            import time
            time.sleep(1)
            resp2 = await client.post(twofa_url, data={
                "user_id": cfg.kite_user_id,
                "request_id": request_id,
                "twofa_value": totp.now(),
                "twofa_type": "totp",
            })
        resp2.raise_for_status()

        # Check for request_token in 2FA response
        twofa_data = resp2.json().get("data", {})
        request_token = twofa_data.get("request_token", "")

        # Step 3: Follow redirects to extract request_token
        if not request_token:
            logger.info("Kite auto-login: following redirects...")
            all_cookies = {}
            for r in (page_resp, resp1, resp2):
                for name, value in r.cookies.items():
                    all_cookies[name] = value

            resp3 = await client.get(connect_url, cookies=all_cookies)
            location = ""

            if resp3.status_code in (301, 302, 303, 307, 308):
                location = resp3.headers.get("location", "")
            else:
                location = str(resp3.url)

            for _ in range(5):
                if not location or "request_token" in location:
                    break
                for name, value in resp3.cookies.items():
                    all_cookies[name] = value
                resp3 = await client.get(location, cookies=all_cookies)
                if resp3.status_code in (301, 302, 303, 307, 308):
                    location = resp3.headers.get("location", "")
                else:
                    location = str(resp3.url)

            if location:
                parsed = urllib.parse.urlparse(location)
                params = urllib.parse.parse_qs(parsed.query)
                request_token = params.get("request_token", [""])[0]

            # Fallback: parse response body
            if not request_token and resp3.status_code == 200:
                m = re.search(r'request_token["\s:=]+([a-zA-Z0-9]+)', resp3.text)
                if m:
                    request_token = m.group(1)

        if not request_token:
            raise RuntimeError(
                "Could not extract request_token. Use browser login instead."
            )

        logger.info("Kite auto-login: got request_token=%s...", request_token[:8])

    return await _set_access_token(request_token)


# ─────────────────────────────────────────────────────────────────────────────
# Login Method 2: Browser callback
# ─────────────────────────────────────────────────────────────────────────────

class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global _callback_token
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        token = params.get("request_token", [""])[0]

        if token:
            _callback_token = token
            _callback_event.set()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body style='font-family:sans-serif;text-align:center;padding:60px'>"
                b"<h1 style='color:green'>Login Successful!</h1>"
                b"<p>You can close this tab.</p></body></html>"
            )
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing request_token")

    def log_message(self, format, *args):
        pass


def start_callback_server() -> Optional[HTTPServer]:
    global _callback_token
    _callback_token = None
    _callback_event.clear()
    try:
        server = HTTPServer(("0.0.0.0", CALLBACK_PORT), _CallbackHandler)
        server.timeout = 1

        def _serve(srv, max_wait=180):
            import time
            deadline = time.time() + max_wait
            while not _callback_event.is_set() and time.time() < deadline:
                srv.handle_request()
            try:
                srv.server_close()
            except Exception:
                pass

        threading.Thread(target=_serve, args=(server,), daemon=True).start()
        return server
    except OSError as e:
        logger.warning("Could not start callback server: %s", e)
        return None


def wait_for_callback(timeout: int = 180) -> Optional[str]:
    _callback_event.wait(timeout=timeout)
    return _callback_token


async def browser_login(request_token: str) -> str:
    return await _set_access_token(request_token)


# ─────────────────────────────────────────────────────────────────────────────
# Login Method 3: Manual token
# ─────────────────────────────────────────────────────────────────────────────

def extract_token_from_url(url: str) -> str:
    url = url.strip()
    if re.match(r'^[a-zA-Z0-9]{10,}$', url):
        return url
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)
    return params.get("request_token", [""])[0]


async def token_login(url_or_token: str) -> str:
    token = extract_token_from_url(url_or_token)
    if not token:
        raise RuntimeError("Could not find request_token in the provided input")
    return await _set_access_token(token)


# ─────────────────────────────────────────────────────────────────────────────
# Common
# ─────────────────────────────────────────────────────────────────────────────

async def _set_access_token(request_token: str) -> str:
    global _access_token
    cfg = load_config()
    kite = get_kite()
    data = kite.generate_session(request_token, api_secret=cfg.kite_api_secret)
    _access_token = data["access_token"]
    kite.set_access_token(_access_token)
    logger.info("Kite session active for user %s", cfg.kite_user_id)
    return _access_token
