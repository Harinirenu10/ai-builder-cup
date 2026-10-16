# API service

FastAPI. Run locally: `uvicorn main:app --port 8000` from `services/api` (needs `pip install -e packages/fitcore`). Deploy: `make deploy-api`. Tests: `python -m pytest services/api/tests`.

| Endpoint | Audience | Returns |
|---|---|---|
| `POST /v1/fit?explain_text=false&language=en` | customer (owner) | full `FitResult`, review notes, optional explanation |
| `POST /v1/biz/verdict` | business | `BizVerdict`: size, per-zone verdicts, alteration actions. No measurements, ease or cm values |
| `POST /v1/tailor-card?languages=en,ta` | customer | `TailorCard` JSON (409 when no size fits) |

Body for all three: `{product, twin, review_signals?, preferences?, pose_deltas?}`; see `samples/product_kurta.json` and `samples/twin.json` (simulated data). The explanation and translation use the LLM backend from `.env` (see `packages/fitcore/README.md`); `explain_text` defaults to false to spare quota.

Not yet: auth and Firestore lookup of the twin by id (today the twin is posted in the request), PNG download, the ADK planner.
