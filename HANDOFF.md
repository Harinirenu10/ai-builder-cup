# Handoff: state of the build (written 2026-10-18)

## Demo virtual try-on: a tee and fitted trousers on the 3D twin (added 2026-10-18)

Built for the demo video. `services/geometry/tryon.py` + `services/api/apilib/tryon.py` +
a **Body Twin / T-Shirt / Trousers** picker on the customer app's 3D twin panel.

Each garment is real 3D geometry, not an image overlay: a closed offset shell (outer face,
inner face, stitched rim at every opening) built from the twin's own SMPL surface, exported
alongside the body into one GLB scene (`body` + `demo_tee` or `demo_trousers`) under a single
grounding transform. Which vertices a garment covers is read off the SMPL model's own skinning
weights, so it works for every gender and every set of betas rather than from a hand-painted
vertex list. The body mesh is handed back unchanged.

- **Tee** (`TopGarmentSpec`): body panel plus two sleeves. Four openings -- neckline, hem, two
  cuffs.
- **Trousers** (`BottomGarmentSpec`): a waistband plus two separate legs with a real inseam, not
  a skirt-like tube. Three openings -- the waistband and one hem per leg. The waistband is cut
  along the twin's own pelvis->neck axis and each leg along its own hip->knee->ankle chain, so
  both survive a pose change. `leg_length_fraction` and `ease_cm` are the length and fit dials
  (shorts/cropped/full-length are the same code with one number changed), which is how jeans or
  shorts get added later: a spec in `tryon.GARMENTS` plus a blurb in `apilib/tryon.py`.

- `GET /api/tryon/garments` -- the catalog, each entry tagged `kind: top|bottom`.
- `GET /api/twin/{twin_id}/tryon/{garment_id}/glb?pose=standing` -- signed-in, owner-only, same
  ownership rule as `GET /api/twin/{twin_id}/glb`, cached per twin AND per garment. Also served
  without the `/api` prefix.
- `python -m services.geometry.tryon --garment <id> --betas ... --out tryon.glb` -- offline CLI,
  prints the shell's measured diagnostics (watertight, boundary loops and ring sizes, clearance
  band, folded-face ratio, and for a bottom the gap between the leg shells below the knee).

**This is a visualization, not a fit answer.** No cloth simulation: no drape, gravity, collision
solve or fabric stretch. It computes no verdict and no measurement; sizing stays with the Fit
Engine and the body-mesh fit heatmap (`POST /fit/mesh`), both untouched.

Measured on the mean neutral body -- tee: 1,838 source vertices, 7,360 shell faces, 4 boundary
loops, clearance 0.89-1.34 cm, 1.0% folded faces. Trousers: 1,125 source vertices, 4,504 shell
faces, 3 boundary loops (40/15/15), clearance 0.45-1.27 cm, 1.2% folded faces, leg shells 5.4 cm
apart below the knee. Both watertight. See "Known limits".

## Product URL Fit Check: why it timed out, and what is still missing (2026-10-18)

The browser gave up after 15s on `POST /v1/me/fit-from-url` even though a direct Reader call
worked. Four separate causes, all now fixed except the last:

1. **The Reader launched Playwright's driver and a fresh Chromium per `/read`** -- ~6.7s of the
   ~9s a cache-missing capture took, against ~2.2s of actual page work.
   `readerlib/capture.py` now keeps one browser (`_BrowserHost`, its own thread and loop, since
   `capture_page` is synchronous) and gives each visit its own fresh context, so visits stay
   isolated. `services/reader/main.py` warms it at startup via a lifespan hook. Capture on the
   demo page: ~8.9s -> ~4s, measured repeatedly.
2. **The Reader cache fell back to RAM, not disk.** `GOOGLE_CLOUD_PROJECT` is a placeholder, so
   `readerlib/cache.py` tried Firestore, failed on credentials, and used `MemoryStore` -- which
   dies with the process. With `--reload` restarting the Reader on every edit, *every* request
   was a cold read. It now falls back to the on-disk `FileStore` this module's own docstring
   documents, so a read survives a restart. This is what made a direct Reader call look fine
   (it had warmed that one process) while the browser always paid full price.
3. **The client budget was below the server's.** apiClient's 15s default sat under the API's own
   `READER_READ_TIMEOUT_S` (35s), so a read still in progress -- or already finished -- was
   thrown away by the browser. That one call now uses 40s, just above the server's, so the
   server's specific 504/422 surfaces instead of a generic client timeout. The extraction model
   call, previously the only unbounded step, is now bounded by `READER_EXTRACT_TIMEOUT_S` (25s).
4. **The size chart depended on a model that kept dropping it.** The chart is published as a
   real HTML `<table>` (inside a `hidden` modal the capture already opens), yet gemma4:12b
   returned it on roughly one attempt in three, so the API answered 422 `no_size_chart`.
   `fitcore/read/tables.py` now reads a structured chart straight out of the page's HTML --
   deterministically, with the page's own `cm`/garment evidence rather than normalize.py's
   inference -- and the Reader fills it in only when the model returned none, flagging
   `size_chart_from_html`. Cold reads went from 1/7 to 4/5 producing a valid "size M @ 0.47".

Also fixed along the way: `fitcore/read/extract.py` treats a structurally valid but **empty**
extraction as a failure and retries once on Gemini (`generate_json` took a new `backend`
override; `llm.gemini_backend_for_role` reports whether there is a Gemini to retry on). And a
read whose extraction returned no category is no longer cached as `unsupported` -- with the
cache now on disk, one transient model failure would otherwise have pinned "not a garment" to
the demo URL for 7 days.

**What is still missing, and it is the demo blocker:** `category` still comes only from the
extraction model, and `GEMINI_PRO_MODEL`/`GEMINI_FLASH_MODEL`/`GEMINI_IMAGE_MODEL`/`VTO_MODEL`
are all empty in `.env`, so there is no Gemini to fall back to and gemma4:12b is the only
option. Late in the session it stopped returning a category for the demo shirt at all (it had
been returning `shirt` reliably earlier the same day), so the endpoint now answers a truthful,
retryable 502 instead of a recommendation. Nothing deterministic was added for category on
purpose: guessing it from the title would silently change which zones and tolerances the fit
engine applies. Fill in the `GEMINI_*_MODEL` IDs from Agent Platform Model Garden and the
existing fallback path covers this; the chart, capture, cache and timeout work above does not
depend on it.

## Save My 3D Twin + restart-surviving twin persistence (added 2026-10-18)

The twin pipeline's dev-fallback clients used to be pure in-RAM dicts, so every twin, job and
uploaded GLB vanished when the API process restarted (the Twin Viewer then showed "No 3D Twin
Yet" even though the capture had succeeded). `services/geometry/local_persistence.py` adds
`FileBackedFirestoreClient`/`FileBackedStorageClient` — the same JSON-file pattern
`services/api/apilib/stores.py::FileStore` and `services/reader/readerlib/cache.py` already
use, no new service — and `cloud_client.py`'s `get_firestore_client`/`get_storage_client` now
fall back to them (documents in `services/geometry/.data/firestore.json`, blobs as real files
under `services/geometry/.data/blobs/`, both gitignored). A configured GCP project still gets
the real `GoogleCloudFirestoreClient`/`GoogleCloudStorageClient` — that path is unchanged.
`GEOMETRY_STORE=memory` restores the old pure-RAM behavior; both test conftests set it so the
suites never write to the repo's dev store.

`POST /v1/me/twin/{twin_id}/save` records the owner's explicit keep intent (`saved`/`saved_at`
on `TwinDocument`, surfaced by `GET /v1/me/twin`); the customer app shows a "Save My 3D Twin"
button and a "Saved" badge. It is idempotent and owner-only.

Verified: twin doc + measurements + exact GLB bytes recovered by a second, separate process
through the real `get_*_client()` selection path; the live reloaded server served a twin
written by a different process; and the browser flow (button → POST 200 → "Saved" badge →
survives page refresh) ran against the real dev servers. Real captures have since run through
it end to end: two real twins (276 KB GLBs) persisted and still serve after many server
reloads.

Cross-process consistency was fixed on 2026-10-18 after it caused a real failure. The client
originally read the JSON file only at construction and rewrote it whole on every write, so a
long-running server silently undid anything another process wrote or deleted. In practice: a
stale probe twin deleted out-of-process reappeared on disk when the server next saved a
capture, and because its `created_at` was later than the real twins', it shadowed them in
`get_latest_twin_for_user` while pointing at a blob that no longer existed — the viewer then
failed with "Storage object not found". `FileBackedFirestoreClient` now re-reads the file
before every read and every write (the same thing `apilib/stores.py::FileStore` does), so
writes merge instead of clobbering. Still **not** verified: more than one process writing
concurrently (last-writer-wins within a single operation; there is no file locking), and the
real Firestore/Cloud Storage path, which remains untested as noted below.

A twin document whose GLB blob is missing now returns 404 from `GET /api/twin/{id}/glb`
(not 500), and `/v1/me/twin` still returns that twin's measurements, so the viewer shows
"3D Model Unavailable" rather than looking like a server fault. Note that such a twin still
wins `get_latest_twin_for_user` if its `created_at` is newest — the lookup deliberately does
not check blob existence (that would couple Firestore to Storage and cost a round-trip per
lookup), so a mesh-less newest twin will shadow older working ones until it is removed.

Prototype deadline: Oct 18. Everything below is about what is **not done** or **not verified**. What works is in each folder's README. All tests are offline and pass (at the commit after `683f032`, with the returns work): `packages/fitcore` 375, `services/api` 150, `services/reader` 55.

## Run the tests
```
pip install -e packages/fitcore && pip install fastapi httpx pytest playwright pillow pyjwt google-auth uvicorn python-multipart
python -m playwright install chromium webkit      # browser tests skip what is missing
cd web && npm install && cd ..                    # needed by the widget e2e test (it builds the consent popup)
python -m pytest packages/fitcore/tests services/api/tests services/reader/tests
```
The reader and widget tests drive real browsers and take about 3 minutes in total.

## Fixed this session: `services/api` would not start

`main.py` mixed two import conventions (`apilib.*`, which needs `services/api` itself on
`sys.path`, and `services.api.*`/`services.geometry.*`, which needs the repo root on
`sys.path`), so the app crashed on import under both documented launch methods:
`infra/Makefile`'s `dev` target (`ModuleNotFoundError: No module named 'apilib'`) and
`services/api/Dockerfile`'s `uvicorn main:app` (`ModuleNotFoundError: No module named
'services'` — `services/geometry` was never copied into the image). The test suite never
caught this because `python -m pytest` and `conftest.py` put both required paths on
`sys.path` as a side effect of how pytest collects tests, which a real `uvicorn` launch
doesn't get. `services/reader/main.py` had the same problem for `make dev` specifically
(`readerlib` not on `sys.path` from the repo root).

Fixed by adding a small `sys.path` bootstrap to the top of `services/api/main.py` and
`services/reader/main.py` (each inserts its own directory, which is enough for `apilib.*`/
`readerlib.*` under both launch methods), changing `main.py`'s one self-referential
`from services.api.garment_routes import ...` to a plain sibling import, and adding
`services/geometry/__init__.py` + `cloud_client.py` (the one geometry module `main.py`,
`garment_routes.py`, and `apilib/fit_mesh.py` import unconditionally — it has no
torch/smplx/mediapipe dependency, unlike the rest of the geometry service) to the API
Dockerfile's `COPY` step. Verified `/healthz` returns 200 for both launch methods (the
Dockerfile path was verified by mirroring its exact `COPY` layout in a throwaway directory,
since Docker itself isn't available in this environment) and that
`python -m pytest services/api/tests -q` is still 204 passed/9 skipped (unchanged). Added
`services/api/tests/test_app_startup.py` and `services/reader/tests/test_app_startup.py`
as regression tests — they use a subprocess with a controlled environment instead of
relying on pytest's own import machinery, since that's the only way to actually catch this
class of bug.

**Not fixed, pre-existing, separate from this bug:** the heavier geometry modules
(`fitting.py`, `garment_heatmap.py`, `exporter.py`, `poses.py`, `worker.py`) are still not
in the API image — `main.py` and `apilib/fit_mesh.py` already guard those imports and
return 503 if unavailable, by design, since this image does not install torch/smplx/
mediapipe. `google-adk` is listed in `services/api/requirements.txt` but is not actually
usable (see `apilib/planner.py`'s own comment on the `opentelemetry-api` version conflict)
and `google-cloud-storage` is not in `requirements.txt` either, so `get_storage_client()`'s
real (non-in-memory) path would still fail if called in a deployed image — neither is new,
and neither was in scope for this fix.

## What exists
- `packages/fitcore`: fit engine, reviews, tailor card, LLM wrapper, reader normalisation, calibration CLI.
- `services/api`: `/v1/fit`, `/v1/biz/verdict`, `/v1/tailor-card`, business fit query, consent, gifts, orders with a fit guarantee, nudge events, returns routing, returns desk endpoints and KPIs. See its README.
- `packages/fitcore/fitcore/returns`: deterministic return routing (4 rules, decision log) and the neutral case summary with language guardrails.
- `services/reader`: `/read`, `/read-image`, CLI, five sample pages. See its README.
- `web`: consent popup `/app/connect`, widget `web/widget`, static store `web/store` (now with the bracketing nudge in the bag, a simulated checkout and a returns form), returns desk at `/biz/returns`. The rest of the React app (`/app`, `/biz`) is still the original scaffold with hard-coded sample data.
- `services/geometry/tryon.py` + `services/api/apilib/tryon.py` + the customer app's **Try On Demo T-Shirt** button: the demo virtual try-on described at the top of this file. One predefined garment only.
- `seed/returns_demo.py`: 4 simulated returns (one per routing branch) for the desk demo; Prompt 12 reuses them.
- `packages/fitcore/fitcore/garment`: garment photo measurement (A4 sheet to scale and perspective, outline, landmarks from Gemini or a file, cm per zone, overlay image), a synthetic photo generator, and a CLI. `samples/garment/` holds 5 simulated photos.
- `packages/fitcore/fitcore/sim.py`, `population.py`, `sim_memo.py` + `services/api/apilib/simulate.py` + `/biz/simulate`: the pre-production return simulation (size mix, below/ideal/above shares per zone per size, no-good-size share, regrade search, pro-model memo, permanent banner). **Its population is a statistical stand-in with invented numbers, not SMPL** (see below).
- `packages/fitcore/fitcore/chart` + `services/api/apilib/charts.py` + `/biz/charts`: the size chart studio (measure per photo, grading check, reviews beside the chart, publish, export, "Use on store"). The mock store's product pages show a live chart when the seller has put one on the store.
- Simulated data is labelled as such: `samples/*.json`, the Demo Threads Co. pages. Public pages saved in `samples/products/` (Taylor Stitch, Vistaprint) are third-party content, test use only: delete them if that is a problem.

## Must do before a public deploy
- [x] **Firebase Hosting rewrite points at the wrong service name** -- fixed: `firebase.json`'s `/api/**` rewrite now points at `fit-api`, matching `infra/Makefile`'s `API_SERVICE`. (Local config edit only; not deployed.)
- [x] **`CONSENT_JWT_SECRET` is now enforced at startup**, not just at first use: `apilib/auth.py::validate_security_config()` (called from `main.py` before the app is built) raises and refuses to start if it is unset or under 32 characters, whenever `AUTH_MODE` is not the literal string `dev`. Storing the actual secret value in Secret Manager still adds a cloud service: get approval first (CLAUDE.md).
- [x] **`AUTH_MODE=dev` is now blocked at startup when running on Cloud Run.** The same `validate_security_config()` raises if `AUTH_MODE=dev` and `K_SERVICE` (set automatically by every real Cloud Run revision) is present. Still leave `GEMINI_API_KEY`, `OLLAMA_*` and `LLM_BACKEND` unset in the cloud too, so Gemini goes through Vertex -- that part is not enforced in code.
- [x] **`/v1/fit`, `/v1/biz/verdict` and `/v1/tailor-card` no longer take the twin in the request body.** All three now require a signed-in caller (`current_uid`) and always look that caller's own twin up server-side (`apilib/fitquery.py::resolve_twin_measurements_for_uid`, the same Store-then-Firestore lookup `/v1/fit-query` uses) -- a signed-in caller can no longer submit or retrieve another user's measurements. `explain_text=true` on `/v1/fit` still calls an LLM per authenticated request; no rate limit beyond the global per-instance one is applied to it specifically. See "Two business-verdict endpoints" below for `/v1/biz/verdict`'s remaining design question (documented, not changed).
- [ ] **Protect the reader.** `/read` and `/read-image` are open: anyone can trigger Chromium and LLM calls. Require auth (Cloud Run IAM or a key) and a rate limit.
- [ ] **Reader memory.** `make deploy-reader` uses the default memory; Chromium needs about 2 GiB. Add `--memory=2Gi`.
- [ ] **Merchant onboarding does not exist.** Merchants (`merchants/{id}`: name, `publishable_key`, `allowed_origins`) can only be created by hand in Firestore or with `services/api/seed.py` locally. Register the demo store's real origin, otherwise the popup and the API refuse it.
- [ ] **Twins.** The fit query reads `twins/{uid}` = `{owner_id, measurements: {zones: {...}}}`. Nothing creates real twins yet (geometry service not built). For the demo, write a simulated twin (see `seed.py`), and say it is simulated.
- [ ] **Check Gemini model IDs on Vertex.** `.env` holds Developer API IDs (`gemini-3.8-flash`, `gemini-3.1-pro-preview`, ...). Confirm they exist in the Vertex region/location `global`. Never use gemini-1.x or gemini-2.x.
- [ ] **Rotate the Gemini API key.** It appeared in a chat transcript. `.env` is gitignored and was never committed.
- [ ] Add the deployed app and store origins to `API_CORS_ORIGINS` (app) and to the merchant's `allowed_origins` (store).
- [ ] **Returns desk access.** Staff are the uids in `merchants/{id}.staff_uids`; set it (and `return_policy`, `return_window_days`) on the real merchant document. Nothing else grants desk access. There is no UI for these settings.
- [ ] **`POST /v1/orders`, `/v1/returns` and `/v1/store-events` are open to anyone holding the merchant key** (plus the origin check in browsers). A return is identified only by the unguessable order id from checkout. In production the store should call them from its own backend after authenticating the shopper.
- [ ] **The return simulation does not use SMPL yet.** The brief says sample SMPL betas in `services/geometry` and measure them with the Prompt 1 loops; the geometry service is a scaffold, `smplx` and `mediapipe` are not installed, and only `SMPL_FEMALE.pkl` is present (no male model). Instead `population.generate_standin` draws 2,000 bodies per gender from a correlated normal whose means, spreads and correlations I chose by hand (`config/sim.yaml`). **Every simulation result is only as real as those invented numbers.** The page banner says "stand-in, not the SMPL shape space" so it cannot be mistaken for SMPL. To finish: build the SMPL measurement loops, sample and measure the betas, save them as `Population(source="smpl")` parquet at `POPULATION_PATH`, then calibrate against consented, anonymized twins.
- [ ] **Garment measurement has only been tried on synthetic photos.** Accuracy there is 0.1 to 0.3 cm typical, about 2.5 cm worst. The Gemini landmark step was checked on a handful of synthetic photos (4 live calls, 2 recorded: within 1.4 cm). **No real photograph has been measured.** Try real flat-lays before relying on it, and tune `config/grading.yaml` (`jump_tolerance_cm`, step ranges) with real sizes.
- [ ] **API service resources.** Measuring a photo needs a few hundred MB, and the studio measures four at once: give the API Cloud Run service at least 1 GiB (it also now ships OpenCV, `opencv-python-headless`; the Docker image was not rebuilt or run).
- [ ] **Unboxing measurements are simulated.** `unboxing_measurements/{id}` is only written by `seed/returns_demo.py` and tests; nothing produces one from a photo (garment photo measurement is not built). Overlay images are static SVGs in `web/store/img/unboxing/`; real ones need Cloud Storage (ask first) and signed URLs.

### Two business-verdict endpoints -- documented, not reconciled this session

There are two different ways to get a business-safe fit verdict (size + per-zone verdicts, never
measurements/ease/cm), with two different, deliberately un-reconciled authentication models:

- **`POST /v1/fit-query`** (`apilib/fitquery.py`) -- the intended model for a real external
  merchant integration: the caller authenticates with a merchant's `publishable_key` plus a
  signed, expiring consent token that names a specific shopper who connected their twin to that
  merchant. Answers "what verdict would *this specific consenting shopper* get," asked by the
  merchant's own backend.
- **`POST /v1/biz/verdict`** (`main.py`) -- authenticates the caller as the *shopper themselves*
  (`current_uid`: a Firebase ID token, or `Bearer dev:<uid>` under `AUTH_MODE=dev`) and always
  looks up that caller's own twin. Answers "what verdict would *I* get," just in the
  business-safe response shape instead of the full customer `FitResponse`. It is not a merchant
  integration at all -- nothing about it lets a merchant ask about a shopper; the "biz" in the
  name describes the *response shape* (no measurements), not the caller's identity.

Neither is wrong for what it does; they just answer different questions and should not be treated
as interchangeable. A merchant that wants "ask about one of our shoppers" must use
`/v1/fit-query` with a real consent token -- `/v1/biz/verdict` cannot do that, by design, since it
only ever returns the calling uid's own twin. Whether `/v1/biz/verdict` should exist at all
alongside `/v1/fit-query`, or be merged into it, or renamed to make the distinction clearer, is an
open product/design question, not resolved in this session -- its merchant authorization design
was explicitly out of scope for this fix.

## Not tested (written, but never run for real)
- The **trousers** in a real browser. The tee has been confirmed working in the browser; the trousers have NOT -- their geometry, API route and GLB bytes are covered by tests, and the mesh was checked by software-rendering four views plus a crotch close-up offline, but <model-viewer> has never displayed them and nobody has used the three-way picker in the running app. Do this before the video: sign in at /app, load a twin, switch Body Twin -> T-Shirt -> Trousers -> Body Twin.
- Try-on on Cloud Run: the API image does not install torch/smplx/trimesh/scipy, so both garments answer 503 there unless those deps are added (the same gap POST /fit/mesh already has).
- Vertex path of `llm.py` (`genai.Client(vertexai=True, ...)`). All live runs used the Developer API key. The Ollama path ran live against the relay.
- Firebase Auth: the popup's "Sign in with Google" and server-side ID-token verification (`google.oauth2.id_token.verify_firebase_token`). Tests use the dev stand-in only.
- Firestore: `FirestoreStore` (API) and `FirestoreStore` (reader cache) are untested. `firestore.rules` changes (store consents are server-written, `fit_queries` and `gifts` server-only) were never run in the emulator.
- Docker builds and Cloud Run for api and reader. The reader image is `playwright/python:v1.49.0-jammy` and `requirements.txt` pins `playwright==1.49.0`, but tests ran on Playwright 1.60 locally.
- Real Safari (Mac, iPhone). The widget was tested in Chromium and WebKit only. Test pop-up blocking on iOS especially.
- The reader against live sites, apart from one visit each to Taylor Stitch, Vistaprint, Meesho (403, reported `blocked`) and Orvis (404). `robots.txt` is **not** checked. Decide whether that is acceptable.
- Cloud Tasks, the ADK planner agent, `make deploy-web`, and the hosted `dist/store` + `dist/widget` copy (`npm run build` runs it locally; hosting itself is untested).
- Consent revocation through a real "My twin" screen (the endpoints are tested; the screen does not exist).
- Returns desk with Firebase sign-in (tests use the dev stand-in), and the desk's Firestore access: `desk_merchant` reads every merchant document to find the user's merchant, and KPIs read all of a merchant's orders, returns and events. Fine for a demo; add queries and indexes before real volume.
- The simulation population on Cloud Storage: `Population.save/load` with a `gs://` path (needs `google-cloud-storage`, which is not in the API requirements) was never run; local parquet works and is tested. The memo has run live only on Gemini pro through the Developer API key (one recorded call, 37 s); not on Vertex. The simulation page ran in Chromium and WebKit against a stand-in memo writer that waits 2 s.
- The size chart studio against Firebase sign-in and Firestore (`charts`, and reading `products/{sha256(url)}` that the reader writes: the API and the reader must share that collection and its shape; locally they use different stores, so the demo seeds a copy). The studio ran in Chromium and WebKit with a stand-in measurer (landmarks taken from the synthetic generator), and once against live Gemini on four synthetic photos (15 s to measure, 22 s to publish).
- The case summary has run live only on Gemini (Developer API key, one recorded call). It has not run on Vertex or on Gemma. The template fallback is what runs when no model is configured.

## Not built
- "My twin" screen (list and revoke store consents; endpoints `GET /v1/consents`, `POST /v1/consents/{id}/revoke` are ready). Gift screens (`/app/gift/{id}`).
- The geometry service (MediaPipe, SMPL fitting), twin capture, 18+ check in the capture flow. (Garment photo measurement now exists, see above; its output is not yet wired to unboxing measurements for returns.)
- Studio gaps: no way to attach a chart to a real store's catalogue other than the mock store's live-chart hook (a real store would pull `GET /v1/store/charts/{ref}` or the export), no merchant-facing list of products, no photo upload progress for large files, and `back_rise` cannot be measured from a front flat-lay.
- Try-and-keep checkout (the bag allows two sizes of one item and the nudge works; there is no keep/return step after delivery).
- Returns: shopper-facing status page and emails, actually issuing refunds or shipping exchanges (checkout is a simulation with no payment), a merchant settings screen, and anything that consumes the `model_miss` and `seller_label_error` flags beyond the desk's count (no evaluation export, no seller-facing SKU workflow).
- Business console data (`/biz` is scaffold).
- Tuning: every `# TUNE` number in `packages/fitcore/fitcore/config/*.yaml` is a starting guess. `python -m fitcore.fit.calibrate` proposes values from size charts that list body and garment numbers; none were run on real data.

## Known limits
- Demo try-on: two predefined garments (a tee and fitted trousers), no arbitrary retailer garment and no generative garment modelling. The shell is a geometric offset, so garments can only ever read as close-fitting -- drape, bunching, folds at the knee and a hanging hem are not representable, and the clearance is a display parameter, not a measured garment dimension. About 1% of each shell's outer faces fold back on themselves inside tight concave creases (armpit and under the chin on the tee; the crotch and the back of the knee on the trousers); the shells stay watertight and it is not visible at demo camera distance, but it is real and is reported as `folded_face_ratio`. The trousers have no fly, pockets, belt loops or seam allowance, and the waistband is a cut edge rather than a separate band; at the top of the inseam the two legs' surfaces touch (as real trousers do), though below the knee they are 5.4 cm apart. The seated pose builds and stays watertight for both, but the tee's hem cut splits into 5 boundary loops rather than 4, so the endpoint defaults to standing. Each build is an SMPL forward pass plus mesh work (~2-3 s on CPU); results are cached in memory per instance (last 8, keyed per twin and per garment), so a cold Cloud Run instance pays that cost again.
- **`fitcore.llm`'s `load_dotenv()` overrides what the test environment deliberately clears**, so two tests fail locally whenever the repo's `.env` is populated. Both are test-isolation holes, not product defects, and both are unrelated to the persistence work above (confirmed: with `GEOMETRY_STORE=memory`, which both test conftests set, `FileBackedFirestoreClient` is never constructed during a test run).
  - `test_security_config.py::test_real_process_starts_with_a_valid_production_configuration` — the test strips `AUTH_MODE` from the subprocess environment; `load_dotenv()` puts `AUTH_MODE=dev` straight back, so the production-config guard correctly refuses to start. The guard works; the test cannot isolate itself from `.env`. Fails deterministically.
  - `test_commerce.py::test_desk_never_exposes_body_measurements` — order-dependent (passes in isolation and under `-k`, failed once in a full `services/api + services/geometry` run). `conftest.py::_env` deletes `LLM_BACKEND`, but `.env` sets **`LLM_PROVIDER=ollama`**, which nothing deletes, and `load_dotenv()` also restores `OLLAMA_BASE_URL`/`OLLAMA_MODEL`/`GEMINI_API_KEY` at import time. The desk-summary test does not stub `get_summarizer`, so depending on which earlier test first imported `fitcore.llm`, the real summarizer can reach the live Gemma relay and return nondeterministic text — and the test asserts no twin measurement value may appear in any response string. Fixing it properly means either deleting `LLM_PROVIDER` in `_env` too or stubbing `get_summarizer` in that test; neither was done here.
- Rate limit is in memory per instance (several Cloud Run instances each count separately). CORS merchant origins are cached 30 s.
- Reader: reviews behind a "Reviews" tab click are not opened. The reader treats blocked results as cached for 7 days, so a site is not retried for a week.
- The consent token sits in the store page's `localStorage`. Revocation works server-side, but the token itself stays readable by that page's scripts.
- `alteration_hint` in the fit query contains a garment change in cm (for example "sleeves may need 2 cm shortening").
- Simulation definitions: a "good size" is a size where every hard-to-alter zone is ideal both as bought and after the first wash (easy zones can be altered, so they do not count). The below/ideal/above shares per size are among the bodies that size is the best size for. The simulation uses the fit engine's current numbers (`# TUNE` values) and ignores personal ease offsets, review adjustments and poses. The regrade is a coordinate search (up to 2 cm per zone per size, never making a size smaller than the one below), not a global optimum, and its suggestions are for the sample population only. The memo check guards the numbers and the wording; it cannot judge whether a claim in a sentence is sensible. The memo takes 35 to 45 s on the pro model, so it is a separate call from the statistics.
- Garment photos: the garment must contrast with the surface and the A4 sheet must be plain white and fully in view (a white garment on a white sheet, a patterned surface, or a folded garment will fail or give poor values; the studio reports `no_a4`, `no_garment` or low confidence). Sleeves are measured as a straight line from shoulder tip to cuff, so a sleeve laid with a bend reads short. Landmarks off the garment are dropped and the zone is left empty, never guessed.
- The grading check allows `jump_tolerance_cm` (1 cm) outside a step's range before flagging a jump, because measured values carry about 1 cm of noise; a value smaller than the previous size is flagged at any size. With noise, a real 1 cm decrease can show up as "grows only 0.1 cm" (seen in the live run).
- Guarantee threshold is one config value, `guarantee.min_confidence` = 0.7 (lowered from 0.75). The fit query's `guarantee_eligible` also needs every hard zone to fit; an order's guarantee is the simpler rule (size bought equals recommended size and confidence >= 0.7). The widget still words confidence as high from 0.75 (display only).
- Return routing cannot compare a measured garment with a **body-size chart** (the kurta and blouse in the demo store), so a guaranteed sizing return on those goes to a person. Guaranteed sizing returns with no unboxing measurement also go to a person.
- The nudge appears only with consent and when the recommended size is one of the sizes in the bag; "shown" is logged once per cart view per product.
- Extraction quirks seen: Gemini called the "Utility Overshirt" a shirt (not a jacket) and the polo a shirt; fixtures record this.

## Env vars
`GOOGLE_CLOUD_PROJECT`, `GEMINI_PRO_MODEL`, `GEMINI_FLASH_MODEL`, `GEMINI_IMAGE_MODEL`, `VTO_MODEL`; dev only: `LLM_BACKEND`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `GEMINI_API_KEY`, `GEMINI_API_MAX_CALLS`. API: `CONSENT_JWT_SECRET`, `AUTH_MODE`, `API_STORE`, `API_CORS_ORIGINS`, `RATE_LIMIT_PER_MIN`. Geometry/twin dev store: `GEOMETRY_STORE` (`auto` default, disk-backed; `memory` for pure RAM), `GEOMETRY_DATA_DIR` (default `services/geometry/.data`). Reader: `READER_CACHE`, `READER_CACHE_DIR`, `READER_EXTRACT_TIMEOUT_S` (default 25s, bounds the extraction model call), `READER_ALLOW_PRIVATE` and `READER_OFFLINE` (tests only). Web build: `VITE_API_URL` (default `/api`), `VITE_AUTH_MODE` (`dev` only locally), `VITE_FIREBASE_*`. Template: `.env.example`.
