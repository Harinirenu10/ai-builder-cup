# API service

FastAPI. Run locally: `uvicorn main:app --port 8000` from `services/api` (needs `pip install -e packages/fitcore`). Deploy: `make deploy-api`. Tests: `python -m pytest services/api/tests`.

| Endpoint | Audience | Returns |
|---|---|---|
| `POST /v1/fit?explain_text=false&language=en` | customer (owner) | full `FitResult`, review notes, optional explanation |
| `POST /v1/biz/verdict` | business | `BizVerdict`: size, per-zone verdicts, alteration actions. No measurements, ease or cm values |
| `POST /v1/tailor-card?languages=en,ta` | customer | `TailorCard` JSON (409 when no size fits) |

Body for all three: `{product, twin, review_signals?, preferences?, pose_deltas?}`; see `samples/product_kurta.json` and `samples/twin.json` (simulated data). The explanation and translation use the LLM backend from `.env` (see `packages/fitcore/README.md`); `explain_text` defaults to false to spare quota.

Not yet: auth and Firestore lookup of the twin by id (today the twin is posted in the request), PNG download, the ADK planner.

## Business fit query, consent and gifts

| Endpoint | Caller | Purpose |
|---|---|---|
| `POST /v1/fit-query` | store (publishable key in `X-Merchant-Key` + consent token) | `{recommended_size, confidence, zone_verdicts[{zone, verdict}], alteration_hint, guarantee_eligible}`. No measurement or shape fields; a test fails if one is added |
| `GET /v1/merchants/lookup?key=&origin=` | consent popup | store name, and whether the opener origin is registered |
| `POST /v1/consents` | popup, signed-in user | creates (or reuses) the consent and returns a 30-day signed token, only if `origin` is in the merchant's `allowed_origins` |
| `GET /v1/consents`, `POST /v1/consents/{id}/revoke` | customer app (My twin) | list a user's store consents; revoke one. Later queries with its token get 403 |
| `POST /v1/gifts`, `GET /v1/gifts/{id}`, `POST /v1/gifts/{id}/approve`, `.../decline` | giver / recipient | the recipient approves once; the giver's view is only `recommended_size` and `confidence` |

Routes also exist under `/api/v1/...` (the Firebase Hosting rewrite), and the gift routes also without a prefix (`POST /gifts`, `GET /gifts/{id}`).

Checks on a fit query, in order: merchant key (401); `Origin`, if sent, must be registered (403); per-merchant rate limit (429; default 60 a minute, `rate_limit_per_min` on the merchant overrides it); token signature and expiry (401); token belongs to this merchant (403); consent not revoked (403); the shopper has a twin (409). Each answered query logs `{merchant_id, product_ref, recommended_size, confidence, ts}`: no body data and no user id.

Data: `merchants/{id} {name, publishable_key, allowed_origins}`, `consents/{id} {uid, merchant_id, scope: "fit_verdict", created_at, revoked_at}`, `fit_queries`, `gifts`, `twins/{uid} {owner_id, measurements}`.

| Setting | Meaning |
|---|---|
| `CONSENT_JWT_SECRET` | HS256 key for consent tokens. Required: the service fails closed without it. Keep it in a Cloud Run secret, not in the repo |
| `AUTH_MODE=dev` | accepts `Authorization: Bearer dev:<uid>` and a built-in dev secret. Local work and tests only, never production |
| `API_STORE` | `firestore` (default when `GOOGLE_CLOUD_PROJECT` is set), `file` (default locally, `services/api/.data`), `memory` |
| `API_CORS_ORIGINS` | origins of the customer app. Registered merchant origins are allowed automatically; nothing else is |
| `RATE_LIMIT_PER_MIN` | default per-merchant limit |

Signed-in users send a Firebase ID token (`Authorization: Bearer ...`), verified with `google-auth` against `GOOGLE_CLOUD_PROJECT`. The rate limit is kept in memory, so with several Cloud Run instances each enforces its own window.

Local demo: `python services/api/seed.py`, then see `web/store/README.md`. Browser tests (`tests/test_widget_e2e.py`) need `playwright install chromium webkit` and `npm install` in `web/`; they skip what is missing.
