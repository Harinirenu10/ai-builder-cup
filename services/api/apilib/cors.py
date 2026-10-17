"""CORS that follows the merchant registry instead of allowing every origin.

An origin may call the API from a browser if it is the customer app itself (API_CORS_ORIGINS) or an origin
registered by some merchant. The fit query then checks that the origin belongs to the merchant whose key it carries.
"""

import os
import time
from typing import Any

from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

from apilib.stores import get_store

ALLOW_HEADERS = "authorization, content-type, x-merchant-key"
ALLOW_METHODS = "GET, POST, OPTIONS"


def _app_origins() -> set[str]:
    raw = os.environ.get("API_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    return {o.strip().rstrip("/").lower() for o in raw.split(",") if o.strip()}


class DynamicCORS:
    def __init__(self, app: ASGIApp, ttl_s: float = 30.0) -> None:
        self.app, self.ttl, self._cache = app, ttl_s, (0.0, set())

    def _merchant_origins(self) -> set[str]:
        stamp, origins = self._cache
        if time.monotonic() - stamp > self.ttl:
            try:
                origins = {o.rstrip("/").lower() for m in get_store().find("merchants") for o in m.get("allowed_origins", [])}
            except Exception:
                origins = set()
            self._cache = (time.monotonic(), origins)
        return origins

    def allowed(self, origin: str) -> bool:
        o = origin.rstrip("/").lower()
        return o in _app_origins() or o in self._merchant_origins()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
        origin = headers.get("origin")
        ok = bool(origin) and self.allowed(origin)  # type: ignore[arg-type]
        if scope["method"] == "OPTIONS" and "access-control-request-method" in headers:
            resp = Response(status_code=204 if ok else 403)
            if ok:
                self._add(resp.headers, origin)  # type: ignore[arg-type]
                resp.headers["Access-Control-Max-Age"] = "600"
            return await resp(scope, receive, send)

        async def send_wrapper(message: Any) -> None:
            if ok and message["type"] == "http.response.start":
                from starlette.datastructures import MutableHeaders
                self._add(MutableHeaders(scope=message), origin)  # type: ignore[arg-type]
            await send(message)

        await self.app(scope, receive, send_wrapper)

    @staticmethod
    def _add(h: Any, origin: str) -> None:
        h["Access-Control-Allow-Origin"] = origin
        h["Access-Control-Allow-Headers"] = ALLOW_HEADERS
        h["Access-Control-Allow-Methods"] = ALLOW_METHODS
        h["Vary"] = "Origin"
