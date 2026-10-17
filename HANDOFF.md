# Handoff: state of the build (written 2026-10-17)

Prototype deadline: Oct 18. Everything below is about what is **not done** or **not verified**. What works is in each folder's README. All tests are offline and pass at commit `5a93cf9`: `packages/fitcore` 190, `services/api` 78, `services/reader` 55.

## Run the tests
```
pip install -e packages/fitcore && pip install fastapi httpx pytest playwright pillow pyjwt google-auth uvicorn
python -m playwright install chromium webkit      # browser tests skip what is missing
cd web && npm install && cd ..                    # needed by the widget e2e test (it builds the consent popup)
python -m pytest packages/fitcore/tests services/api/tests services/reader/tests
```
The reader and widget tests drive real browsers and take about 3 minutes in total.

## What exists
- `packages/fitcore`: fit engine, reviews, tailor card, LLM wrapper, reader normalisation, calibration CLI.
- `services/api`: `/v1/fit`, `/v1/biz/verdict`, `/v1/tailor-card`, business fit query, consent, gifts. See its README.
- `services/reader`: `/read`, `/read-image`, CLI, five sample pages. See its README.
- `web`: consent popup `/app/connect`, widget `web/widget`, static store `web/store`. The rest of the React app (`/app`, `/biz`) is still the original scaffold with hard-coded sample data.
- Simulated data is labelled as such: `samples/*.json`, the Demo Threads Co. pages. Public pages saved in `samples/products/` (Taylor Stitch, Vistaprint) are third-party content, test use only: delete them if that is a problem.

## Must do before a public deploy
- [ ] **Firebase Hosting rewrite points at the wrong service name.** `firebase.json` sends `/api/**` to Cloud Run service `api`, but `make deploy-api` deploys `fit-api`. Make them match.
- [ ] **Set `CONSENT_JWT_SECRET`** on the API service (random, 32+ bytes). Without it the API refuses to issue tokens. Storing it in Secret Manager adds a cloud service: get approval first (CLAUDE.md).
- [ ] **Never set `AUTH_MODE=dev`** in the cloud. It accepts `Bearer dev:<anything>` as a signed-in user. Also leave `GEMINI_API_KEY`, `OLLAMA_*` and `LLM_BACKEND` unset there, so Gemini goes through Vertex.
- [ ] **Lock down the older endpoints.** `/v1/fit`, `/v1/biz/verdict` and `/v1/tailor-card` take the twin in the request body with no authentication, and `explain_text=true` calls an LLM. Add auth or remove them before they are public. `/v1/biz/verdict` is also meant to be business-facing and should not receive measurements at all: it should look the twin up server-side like `/v1/fit-query` does.
- [ ] **Protect the reader.** `/read` and `/read-image` are open: anyone can trigger Chromium and LLM calls. Require auth (Cloud Run IAM or a key) and a rate limit.
- [ ] **Reader memory.** `make deploy-reader` uses the default memory; Chromium needs about 2 GiB. Add `--memory=2Gi`.
- [ ] **Merchant onboarding does not exist.** Merchants (`merchants/{id}`: name, `publishable_key`, `allowed_origins`) can only be created by hand in Firestore or with `services/api/seed.py` locally. Register the demo store's real origin, otherwise the popup and the API refuse it.
- [ ] **Twins.** The fit query reads `twins/{uid}` = `{owner_id, measurements: {zones: {...}}}`. Nothing creates real twins yet (geometry service not built). For the demo, write a simulated twin (see `seed.py`), and say it is simulated.
- [ ] **Check Gemini model IDs on Vertex.** `.env` holds Developer API IDs (`gemini-3.8-flash`, `gemini-3.1-pro-preview`, ...). Confirm they exist in the Vertex region/location `global`. Never use gemini-1.x or gemini-2.x.
- [ ] **Rotate the Gemini API key.** It appeared in a chat transcript. `.env` is gitignored and was never committed.
- [ ] Add the deployed app and store origins to `API_CORS_ORIGINS` (app) and to the merchant's `allowed_origins` (store).

## Not tested (written, but never run for real)
- Vertex path of `llm.py` (`genai.Client(vertexai=True, ...)`). All live runs used the Developer API key. The Ollama path ran live against the relay.
- Firebase Auth: the popup's "Sign in with Google" and server-side ID-token verification (`google.oauth2.id_token.verify_firebase_token`). Tests use the dev stand-in only.
- Firestore: `FirestoreStore` (API) and `FirestoreStore` (reader cache) are untested. `firestore.rules` changes (store consents are server-written, `fit_queries` and `gifts` server-only) were never run in the emulator.
- Docker builds and Cloud Run for api and reader. The reader image is `playwright/python:v1.49.0-jammy` and `requirements.txt` pins `playwright==1.49.0`, but tests ran on Playwright 1.60 locally.
- Real Safari (Mac, iPhone). The widget was tested in Chromium and WebKit only. Test pop-up blocking on iOS especially.
- The reader against live sites, apart from one visit each to Taylor Stitch, Vistaprint, Meesho (403, reported `blocked`) and Orvis (404). `robots.txt` is **not** checked. Decide whether that is acceptable.
- Cloud Tasks, the ADK planner agent, `make deploy-web`, and the hosted `dist/store` + `dist/widget` copy (`npm run build` runs it locally; hosting itself is untested).
- Consent revocation through a real "My twin" screen (the endpoints are tested; the screen does not exist).

## Not built
- "My twin" screen (list and revoke store consents; endpoints `GET /v1/consents`, `POST /v1/consents/{id}/revoke` are ready). Gift screens (`/app/gift/{id}`).
- Garment photo measurement (flat-lay + A4), the geometry service (MediaPipe, SMPL fitting), twin capture, 18+ check in the capture flow.
- Try-and-keep checkout (the store bag already allows two sizes of one item).
- Business console data (`/biz` is scaffold).
- Tuning: every `# TUNE` number in `packages/fitcore/fitcore/config/*.yaml` is a starting guess. `python -m fitcore.fit.calibrate` proposes values from size charts that list body and garment numbers; none were run on real data.

## Known limits
- Rate limit is in memory per instance (several Cloud Run instances each count separately). CORS merchant origins are cached 30 s.
- Reader: reviews behind a "Reviews" tab click are not opened. The reader treats blocked results as cached for 7 days, so a site is not retried for a week.
- The consent token sits in the store page's `localStorage`. Revocation works server-side, but the token itself stays readable by that page's scripts.
- `alteration_hint` in the fit query contains a garment change in cm (for example "sleeves may need 2 cm shortening").
- Extraction quirks seen: Gemini called the "Utility Overshirt" a shirt (not a jacket) and the polo a shirt; fixtures record this.

## Env vars
`GOOGLE_CLOUD_PROJECT`, `GEMINI_PRO_MODEL`, `GEMINI_FLASH_MODEL`, `GEMINI_IMAGE_MODEL`, `VTO_MODEL`; dev only: `LLM_BACKEND`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `GEMINI_API_KEY`, `GEMINI_API_MAX_CALLS`. API: `CONSENT_JWT_SECRET`, `AUTH_MODE`, `API_STORE`, `API_CORS_ORIGINS`, `RATE_LIMIT_PER_MIN`. Reader: `READER_CACHE`, `READER_CACHE_DIR`, `READER_ALLOW_PRIVATE` and `READER_OFFLINE` (tests only). Web build: `VITE_API_URL` (default `/api`), `VITE_AUTH_MODE` (`dev` only locally), `VITE_FIREBASE_*`. Template: `.env.example`.
