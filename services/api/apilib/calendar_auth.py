"""Google Calendar OAuth 2.0 flow, secure token management, and 7-day event retrieval."""

from __future__ import annotations

import datetime
import logging
import os
import secrets
import time
from typing import Any
import urllib.parse

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
import requests

from apilib.auth import current_uid
from apilib.stores import Store, get_store
from fitcore.schemas import CalendarEvent

logger = logging.getLogger("api.calendar_auth")

router = APIRouter(prefix="/auth/calendar", tags=["calendar_auth"])

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_CALENDAR_EVENTS_URL = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar.readonly"
STATE_TTL_SECONDS = 600  # 10 minutes


class ConnectResponse(BaseModel):
    auth_url: str
    state: str


class ConnectionStatusResponse(BaseModel):
    connected: bool
    scope: str | None = None


class DisconnectResponse(BaseModel):
    status: str


def _get_oauth_config() -> tuple[str, str, str]:
    """Retrieve Google OAuth client credentials."""
    client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")
    redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/calendar/callback")
    return client_id, client_secret, redirect_uri


@router.get("/connect", response_model=ConnectResponse)
def get_connect_url(
    uid: str = Depends(current_uid),
    store: Store = Depends(get_store),
) -> ConnectResponse:
    """Generate a Google OAuth 2.0 authorization URL with cryptographically secure single-use state."""
    client_id, _, redirect_uri = _get_oauth_config()

    # Generate cryptographically secure state bound to authenticated user
    state = secrets.token_urlsafe(32)
    state_doc = {
        "uid": uid,
        "created_at": time.time(),
        "used": False,
    }
    store.set("oauth_states", state, state_doc)

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": CALENDAR_SCOPE,
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    auth_url = f"{GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"
    return ConnectResponse(auth_url=auth_url, state=state)


@router.get("/callback")
def oauth_callback(
    code: str = Query(...),
    state: str = Query(...),
    uid: str = Depends(current_uid),
    store: Store = Depends(get_store),
    http_session: Any | None = None,
) -> dict[str, str]:
    """Validate single-use state, exchange code for tokens, and securely persist credentials."""
    state_doc = store.get("oauth_states", state)
    if not state_doc:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state parameter.")

    if state_doc.get("used", False):
        raise HTTPException(status_code=400, detail="OAuth state has already been consumed.")

    if state_doc.get("uid") != uid:
        raise HTTPException(status_code=403, detail="OAuth state user mismatch.")

    # Check 10-minute expiry
    created_at = state_doc.get("created_at", 0)
    if time.time() - created_at > STATE_TTL_SECONDS:
        raise HTTPException(status_code=400, detail="OAuth state has expired.")

    # Immediately consume state (single-use)
    store.update("oauth_states", state, {"used": True})

    client_id, client_secret, redirect_uri = _get_oauth_config()
    token_payload = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }

    try:
        requester = http_session or requests
        resp = requester.post(GOOGLE_TOKEN_URL, data=token_payload, timeout=15)
        if hasattr(resp, "raise_for_status"):
            resp.raise_for_status()
        token_data = resp.json() if hasattr(resp, "json") else resp
    except Exception as exc:
        logger.error("Token exchange failed: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to exchange authorization code with Google.")

    expires_in = token_data.get("expires_in", 3600)
    token_record = {
        "access_token": token_data.get("access_token"),
        "refresh_token": token_data.get("refresh_token"),
        "expires_at": time.time() + expires_in,
        "scope": token_data.get("scope", CALENDAR_SCOPE),
        "user_id": uid,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    store.set("calendar_tokens", uid, token_record)

    return {"status": "connected", "scope": "calendar.readonly"}


@router.get("/status", response_model=ConnectionStatusResponse)
def get_connection_status(
    uid: str = Depends(current_uid),
    store: Store = Depends(get_store),
) -> ConnectionStatusResponse:
    """Return whether the user has connected Google Calendar without exposing secrets."""
    token_doc = store.get("calendar_tokens", uid)
    if token_doc and token_doc.get("access_token"):
        return ConnectionStatusResponse(connected=True, scope=token_doc.get("scope", CALENDAR_SCOPE))
    return ConnectionStatusResponse(connected=False, scope=None)


@router.post("/disconnect", response_model=DisconnectResponse)
def disconnect_calendar(
    uid: str = Depends(current_uid),
    store: Store = Depends(get_store),
) -> DisconnectResponse:
    """Safely disconnect Google Calendar by clearing stored tokens."""
    store.set("calendar_tokens", uid, {"access_token": None, "refresh_token": None, "user_id": uid})
    return DisconnectResponse(status="disconnected")


def fetch_google_calendar_events(
    uid: str,
    store: Store,
    start_iso: str | None = None,
    http_client: Any | None = None,
) -> list[CalendarEvent] | None:
    """Fetch next 7 days of events from Google Calendar API, or return None if not connected / on error."""
    token_doc = store.get("calendar_tokens", uid)
    if not token_doc or not token_doc.get("access_token"):
        return None

    access_token = token_doc.get("access_token")
    expires_at = token_doc.get("expires_at", 0)

    # Refresh token if expired and refresh_token is present
    if time.time() >= expires_at and token_doc.get("refresh_token"):
        client_id, client_secret, _ = _get_oauth_config()
        refresh_payload = {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": token_doc.get("refresh_token"),
            "grant_type": "refresh_token",
        }
        try:
            requester = http_client or requests
            ref_resp = requester.post(GOOGLE_TOKEN_URL, data=refresh_payload, timeout=10)
            data = ref_resp.json()
            access_token = data.get("access_token")
            token_doc["access_token"] = access_token
            token_doc["expires_at"] = time.time() + data.get("expires_in", 3600)
            store.set("calendar_tokens", uid, token_doc)
        except Exception as exc:
            logger.warning("Failed to refresh Google Calendar token (%s); falling back to demo events.", exc)
            return None

    # Calculate 7-day range
    now = datetime.datetime.now(datetime.timezone.utc)
    if start_iso:
        try:
            start_dt = datetime.datetime.fromisoformat(start_iso)
        except Exception:
            start_dt = now
    else:
        start_dt = now

    end_dt = start_dt + datetime.timedelta(days=7)
    time_min = start_dt.isoformat()
    time_max = end_dt.isoformat()

    params = {
        "timeMin": time_min,
        "timeMax": time_max,
        "singleEvents": "true",
        "orderBy": "startTime",
    }
    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        requester = http_client or requests
        res = requester.get(GOOGLE_CALENDAR_EVENTS_URL, headers=headers, params=params, timeout=10)
        items = res.json().get("items", [])
    except Exception as exc:
        logger.warning("Google Calendar API call failed (%s); falling back to demo events.", exc)
        return None

    from apilib.planner import propose_circle

    mapped_events: list[CalendarEvent] = []
    for item in items:
        start_raw = item.get("start", {})
        end_raw = item.get("end", {})
        start_time = start_raw.get("dateTime") or f"{start_raw.get('date', '2026-10-18')}T09:00:00"
        end_time = end_raw.get("dateTime") or f"{end_raw.get('date', '2026-10-18')}T10:00:00"
        title = item.get("summary", "Calendar Event")
        attendees = [a.get("email") for a in item.get("attendees", []) if a.get("email")]
        description = item.get("description")
        circle = propose_circle(title, attendees, description)

        ev = CalendarEvent(
            id=item.get("id", secrets.token_hex(8)),
            title=title,
            start_time=start_time,
            end_time=end_time,
            location=item.get("location"),
            attendees=attendees,
            circle=circle,
            description=description,
        )
        mapped_events.append(ev)

    return mapped_events
