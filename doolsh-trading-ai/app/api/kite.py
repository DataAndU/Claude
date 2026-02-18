"""Kite Connect authentication and session management endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.deps import get_current_user, require_role
from app.core.kite import auto_login, get_kite, get_login_url, is_logged_in, set_request_token
from app.models.user import User, UserRole

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kite", tags=["kite"])


@router.get("/login-url")
async def kite_login_url(user: User = Depends(get_current_user)):
    """Get the Kite login URL to open in a browser."""
    return {"login_url": get_login_url()}


@router.get("/callback")
async def kite_callback(request_token: str = Query(...)):
    """OAuth callback — Kite redirects here with request_token.

    Exchange for access_token and store it.
    """
    try:
        access_token = await set_request_token(request_token)
        return {"status": "ok", "message": "Kite session active"}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/auto-login")
async def kite_auto_login(user: User = Depends(require_role(UserRole.ADMIN))):
    """Automated TOTP-based login — no browser needed.

    Requires KITE_USER_ID, KITE_PASSWORD, KITE_TOTP_SECRET in .env.
    """
    try:
        access_token = await auto_login()
        return {"status": "ok", "message": "Auto-login successful"}
    except Exception as exc:
        logger.exception("Auto-login failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/status")
async def kite_status(user: User = Depends(get_current_user)):
    """Check if Kite session is active."""
    logged_in = is_logged_in()
    result = {"logged_in": logged_in}
    if logged_in:
        try:
            profile = get_kite().profile()
            result["user_id"] = profile.get("user_id")
            result["user_name"] = profile.get("user_name")
            result["email"] = profile.get("email")
            result["broker"] = profile.get("broker", "ZERODHA")
        except Exception:
            result["logged_in"] = False
    return result


@router.get("/margins")
async def kite_margins(user: User = Depends(get_current_user)):
    """Get account margins / available funds."""
    if not is_logged_in():
        raise HTTPException(status_code=401, detail="Kite not logged in")
    try:
        margins = get_kite().margins()
        return margins
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
