"""Who is calling: signed-in users (Firebase ID tokens), consent tokens (JWT), and rate limits."""

import logging
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from fastapi import Header, HTTPException

logger = logging.getLogger("api.auth")

ISSUER = "measured-fit"
CONSENT_DAYS = 30
SCOPE = "fit_verdict"


def err(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


# ----------------------------------------------------------------------------- startup security config

MIN_CONSENT_SECRET_LEN = 32


def is_production() -> bool:
    """True when this process is a real deployed Cloud Run revision (CLAUDE.md: Cloud Run is the
    only hosting target for this service). Cloud Run sets `K_SERVICE` on every revision
    automatically; nothing in local dev, `pytest`, or `services/api/Dockerfile` sets it, so this
    needs no new env var a deploy script could forget to set, unlike `AUTH_MODE` itself."""
    return bool(os.environ.get("K_SERVICE"))


def validate_security_config() -> None:
    """Fail at import time, not on the first request, when the auth configuration is unsafe.

    Two independent checks, called once from `main.py` right before the app is built:

    1. `AUTH_MODE=dev` (the `Bearer dev:<uid>` shortcut in `current_uid`) must never be reachable
       on a real Cloud Run deployment. Gated on `is_production()` specifically -- not merely
       "AUTH_MODE is set to something" -- so it cannot fire for a local developer who sets
       AUTH_MODE=dev, which is its entire purpose.
    2. `CONSENT_JWT_SECRET` must be set and long enough whenever `AUTH_MODE` is not the literal
       string "dev" -- the same "is this dev mode" signal `_secret()` already uses, so a developer
       running the real Firebase-token path locally (AUTH_MODE unset) is held to the same bar as
       production, matching `_secret()`'s pre-existing fail-closed behavior for that case; only the
       strength threshold and the startup-time (vs. first-request-time) check are new.

    Never logs the secret itself, only its length.
    """
    if is_production() and os.environ.get("AUTH_MODE") == "dev":
        raise RuntimeError(
            "AUTH_MODE=dev is set but this process is running on Cloud Run (K_SERVICE is set). "
            "AUTH_MODE=dev accepts `Bearer dev:<any-uid>` as a signed-in user and must never be "
            "reachable in a real deployment. Remove AUTH_MODE from this service's deployed "
            "environment."
        )
    if os.environ.get("AUTH_MODE") != "dev":
        secret = os.environ.get("CONSENT_JWT_SECRET", "")
        if not secret:
            raise RuntimeError(
                "CONSENT_JWT_SECRET is not set. Set it to a random secret of at least "
                f"{MIN_CONSENT_SECRET_LEN} characters (AUTH_MODE=dev is the only exemption, for "
                "local development)."
            )
        if len(secret) < MIN_CONSENT_SECRET_LEN:
            raise RuntimeError(
                f"CONSENT_JWT_SECRET is too short ({len(secret)} characters); it must be at "
                f"least {MIN_CONSENT_SECRET_LEN} characters."
            )


# ----------------------------------------------------------------------------- users


def current_uid(authorization: str | None = Header(default=None)) -> str:
    """UID of the signed-in user from `Authorization: Bearer <Firebase ID token>`.

    AUTH_MODE=dev also accepts `Bearer dev:<uid>` for local work and tests. Never set it in production.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise err(401, "auth_required", "Sign in first.")
    token = authorization.split(" ", 1)[1].strip()
    if os.environ.get("AUTH_MODE") == "dev" and token.startswith("dev:") and len(token) > 4:
        return token[4:]
    try:
        from google.auth.transport import requests as grequests
        from google.oauth2 import id_token

        claims = id_token.verify_firebase_token(token, grequests.Request(), audience=os.environ.get("GOOGLE_CLOUD_PROJECT"))
        return claims["sub"]
    except Exception as exc:  # bad signature, expired, wrong project, no network for certs
        logger.info("ID token rejected: %s", type(exc).__name__)
        raise err(401, "invalid_auth", "Your sign-in is not valid. Sign in again.") from exc


# ----------------------------------------------------------------------------- consent tokens


def _secret() -> str:
    secret = os.environ.get("CONSENT_JWT_SECRET", "")
    if secret:
        return secret
    if os.environ.get("AUTH_MODE") == "dev":
        return "dev-only-consent-secret"
    raise RuntimeError("CONSENT_JWT_SECRET is not set")  # fail closed rather than sign with a guessable key


def issue_consent_token(consent_id: str, merchant_id: str, now: datetime | None = None) -> tuple[str, datetime]:
    """Signed 30-day token. It names the consent and the merchant, never the user or any body data."""
    now = now or datetime.now(timezone.utc)
    exp = now + timedelta(days=CONSENT_DAYS)
    claims = {"iss": ISSUER, "cid": consent_id, "mid": merchant_id, "scope": SCOPE, "iat": int(now.timestamp()), "exp": int(exp.timestamp())}
    return jwt.encode(claims, _secret(), algorithm="HS256"), exp


def verify_consent_token(token: str, now: datetime | None = None) -> dict[str, Any]:
    try:
        claims = jwt.decode(token, _secret(), algorithms=["HS256"], issuer=ISSUER,
                            options={"require": ["exp", "iat", "cid", "mid", "scope"], "verify_exp": False})
    except jwt.PyJWTError as exc:
        raise err(401, "invalid_consent_token", "The consent token is not valid.") from exc
    current = now or datetime.now(timezone.utc)
    if claims["exp"] <= int(current.timestamp()):
        raise err(401, "consent_expired", "This consent has expired. Ask the shopper to reconnect.")
    if claims.get("scope") != SCOPE:
        raise err(403, "wrong_scope", "The consent does not cover fit verdicts.")
    return claims


# ----------------------------------------------------------------------------- rate limit


class RateLimiter:
    """Sliding window per key, kept in memory. With several Cloud Run instances each enforces its own window."""

    def __init__(self, window_s: float = 60.0, clock=time.monotonic) -> None:
        self.window, self.clock = window_s, clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int) -> float | None:
        """None if allowed (and counted); otherwise seconds until a slot frees up."""
        now = self.clock()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= limit:
                return max(1.0, self.window - (now - q[0]))
            q.append(now)
            return None


_limiter = RateLimiter()


def get_limiter() -> RateLimiter:
    return _limiter
