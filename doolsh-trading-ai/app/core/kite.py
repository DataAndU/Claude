"""Zerodha Kite Connect session manager.

Three login methods:
    1. auto_login() — fully automated TOTP-based (headless, no browser)
    2. semi_auto_login() — opens Kite in phone browser, captures callback
    3. manual_token_login() — user pastes the redirect URL with request_token

Also:
    - Exchanging request_token → access_token
    - Storing and refreshing the access_token at runtime
    - Providing a ready-to-use KiteConnect instance
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import socket
import threading
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

import httpx
import pyotp
from kiteconnect import KiteConnect

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_kite: Optional[KiteConnect] = None
_access_token: Optional[str] = None

# Shared between callback server thread and main thread
_callback_token: Optional[str] = None
_callback_event = threading.Event()

CALLBACK_PORT = 5678


def get_kite() -> KiteConnect:
    """Return the singleton KiteConnect instance."""
    global _kite
    if _kite is None:
        _kite = KiteConnect(api_key=settings.kite_api_key)
        if _access_token:
            _kite.set_access_token(_access_token)
    return _kite


def get_login_url(redirect_port: int = CALLBACK_PORT) -> str:
    """Get Kite login URL with local callback as redirect."""
    return (
        f"https://kite.trade/connect/login?v=3"
        f"&api_key={settings.kite_api_key}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Callback HTTP Server — catches request_token from Kite redirect
# ─────────────────────────────────────────────────────────────────────────────

class _CallbackHandler(BaseHTTPRequestHandler):
    """Tiny HTTP handler that captures request_token from Kite's redirect."""

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
                b"<h1 style='color:green'>&#10004; Login Successful!</h1>"
                b"<p>You can close this tab and go back to Userland terminal.</p>"
                b"</body></html>"
            )
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing request_token")

    def log_message(self, format, *args):
        # Suppress default stderr output
        pass


def _start_callback_server(port: int = CALLBACK_PORT) -> Optional[HTTPServer]:
    """Start a one-shot HTTP server to capture the Kite callback."""
    global _callback_token
    _callback_token = None
    _callback_event.clear()

    try:
        server = HTTPServer(("0.0.0.0", port), _CallbackHandler)
        server.timeout = 1  # handle_request timeout
        t = threading.Thread(target=_serve_until_token, args=(server,), daemon=True)
        t.start()
        return server
    except OSError as e:
        logger.warning("Could not start callback server on port %d: %s", port, e)
        return None


def _serve_until_token(server: HTTPServer, max_wait: int = 180):
    """Serve requests until we get the token or timeout."""
    import time
    deadline = time.time() + max_wait
    while not _callback_event.is_set() and time.time() < deadline:
        server.handle_request()
    try:
        server.server_close()
    except Exception:
        pass


def _get_device_ip() -> str:
    """Get the device's LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ─────────────────────────────────────────────────────────────────────────────
# Login Method 1: Fully automatic (TOTP headless)
# ─────────────────────────────────────────────────────────────────────────────

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

    kite = get_kite()
    login_url = "https://kite.zerodha.com/api/login"
    twofa_url = "https://kite.zerodha.com/api/twofa"
    connect_url = f"https://kite.trade/connect/login?v=3&api_key={settings.kite_api_key}"

    totp = pyotp.TOTP(settings.kite_totp_secret)
    request_token = ""

    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
    ) as client:
        # Step 1: Login with user_id + password
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

        # Step 2: Submit TOTP
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

        # Step 3: Get request_token from Kite Connect redirect
        if not request_token:
            logger.info("Kite auto-login step 3: extracting request_token...")

            # Collect all cookies and forward them cross-domain
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
                    m = re.search(r'request_token["\s:=]+([a-zA-Z0-9]+)', body)
                    if m:
                        request_token = m.group(1)

        if not request_token:
            raise RuntimeError(
                "Could not extract request_token from Kite login flow. "
                "Use semi-auto login (option 'k' → '2') to login via browser."
            )

        logger.info("Kite auto-login: got request_token=%s...", request_token[:8])

    return await set_request_token(request_token)


# ─────────────────────────────────────────────────────────────────────────────
# Login Method 2: Semi-automatic (browser login + local callback server)
# ─────────────────────────────────────────────────────────────────────────────

def get_semi_auto_login_url() -> str:
    """Get the Kite login URL for semi-auto flow."""
    return (
        f"https://kite.trade/connect/login?v=3"
        f"&api_key={settings.kite_api_key}"
    )


def start_callback_listener() -> Optional[HTTPServer]:
    """Start the local callback server and return it."""
    return _start_callback_server(CALLBACK_PORT)


def wait_for_callback(timeout: int = 180) -> Optional[str]:
    """Block until callback is received or timeout. Returns request_token or None."""
    _callback_event.wait(timeout=timeout)
    return _callback_token


async def semi_auto_login(request_token: str) -> str:
    """Complete the semi-auto login by exchanging request_token."""
    return await set_request_token(request_token)


# ─────────────────────────────────────────────────────────────────────────────
# Login Method 3: Manual (user pastes redirect URL)
# ─────────────────────────────────────────────────────────────────────────────

def extract_token_from_url(url: str) -> str:
    """Extract request_token from a Kite redirect URL or raw token string."""
    url = url.strip()
    # If it looks like a raw token (no URL structure), return as-is
    if re.match(r'^[a-zA-Z0-9]{10,}$', url):
        return url
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)
    return params.get("request_token", [""])[0]


async def manual_token_login(url_or_token: str) -> str:
    """Login using a manually provided request_token or redirect URL."""
    token = extract_token_from_url(url_or_token)
    if not token:
        raise RuntimeError("Could not find request_token in the provided URL/token")
    return await set_request_token(token)


# ─────────────────────────────────────────────────────────────────────────────
# Common
# ─────────────────────────────────────────────────────────────────────────────

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
