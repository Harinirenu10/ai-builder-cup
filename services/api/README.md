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
| `CONSENT_JWT_SECRET` | HS256 key for consent tokens. Required and must be 32+ characters whenever `AUTH_MODE` is not `dev` -- checked at startup (`apilib/auth.py::validate_security_config`), so a missing or weak secret stops the service before it serves any traffic. Keep it in a Cloud Run secret, not in the repo |
| `AUTH_MODE=dev` | accepts `Authorization: Bearer dev:<uid>` and a built-in dev secret. Local work and tests only, never production -- the same startup check refuses to start if `AUTH_MODE=dev` is set while running on Cloud Run (detected via `K_SERVICE`) |
| `API_STORE` | `firestore` (default when `GOOGLE_CLOUD_PROJECT` is set), `file` (default locally, `services/api/.data`), `memory` |
| `API_CORS_ORIGINS` | origins of the customer app. Registered merchant origins are allowed automatically; nothing else is |
| `RATE_LIMIT_PER_MIN` | default per-merchant limit |

Signed-in users send a Firebase ID token (`Authorization: Bearer ...`), verified with `google-auth` against `GOOGLE_CLOUD_PROJECT`. The rate limit is kept in memory, so with several Cloud Run instances each enforces its own window.

Local demo: `python services/api/seed.py`, then see `web/store/README.md`. Browser tests (`tests/test_widget_e2e.py`) need `playwright install chromium webkit` and `npm install` in `web/`; they skip what is missing.

## Fit guarantee, bracketing nudge and returns desk

Orders, returns and events are store-facing (merchant key + origin, like the fit query); the desk is for signed-in staff.

| Endpoint | Caller | Purpose |
|---|---|---|
| `POST /v1/orders` | store checkout | places a simulated order. With a valid consent the server records `recommended_size_at_purchase` and `confidence_at_purchase` per item and sets `guarantee` = size bought equals the recommended size and confidence >= `guarantee.min_confidence` (0.7, `fitcore/config/zones.yaml`). Without consent, or with a revoked or invalid token, checkout still works and there is no guarantee |
| `POST /v1/store-events` | store cart | logs `nudge_shown`, `nudge_accepted`, `nudge_dismissed`: no user id, no body data |
| `POST /v1/returns` | store returns page | files a return for an order item and routes it (below). The order id from checkout identifies the order; production stores should authenticate the shopper themselves and call this from their backend |
| `GET /v1/biz/returns`, `GET /v1/biz/returns/{id}` | desk | evidence cards: unboxing overlay, ordered-size spec vs measured garment values, twin verdict at purchase (size, confidence), routing reason and log |
| `POST /v1/biz/returns/{id}/summary` | desk | neutral facts-only summary from the flash model (`?refresh=true` to redo). Falls back to a template if the model is unavailable or breaks the language rules |
| `POST /v1/biz/returns/{id}/decision` | desk | `approve`, `exchange` or `review` with an optional note, added to the decision log. There is no reject action |
| `GET /v1/biz/kpis` | desk | guaranteed share of orders, exchanges vs refunds, nudges accepted, seller label errors |

Desk access: a Firebase user whose uid is in `merchants/{id}.staff_uids` (dev mode: `Bearer dev:<uid>`). The desk never returns body measurements: only garment values and the twin's verdict at purchase.

### Routing (`fitcore/returns/routing.py`, tunables in `fitcore/config/returns.yaml`)
1. Guarantee and the unboxing measurement does not match the label: exchange approved automatically, SKU flagged for a label error.
2. Guarantee, too_small or too_big, label matches: exchange to the next size approved, logged as a model miss.
3. wrong_size_received and the label matches: a person reviews it.
4. No guarantee: the merchant's standard policy (`merchants.return_policy` = refund, exchange or review, default review; inside `return_window_days`, default 30; outside it a person decides).

Cases the rules do not cover go to a person or the standard policy: no unboxing measurement, a body-size chart that cannot be compared with a garment, no next size to exchange to, or a non-sizing reason on a guaranteed item. Nothing is rejected automatically and nothing penalises the shopper. Every step is written to `decision_log` with the numbers used. "Matches the label" means every measured zone is within `label_match_tolerance_cm` (1.5) of the ordered size's chart row.

Unboxing measurements (`unboxing_measurements/{id}`: order_id, sku, size_label, `measured_cm`, `overlay_image_url`, `source`) are **simulated** for now: nothing produces them from a photo yet. The routing and the desk only read them.

Demo data: `python seed/returns_demo.py` creates four returns, one per branch (all simulated), plus orders and nudge events. Desk sign-in in dev mode: user id `demo-staff`.

## Size chart studio

Staff-only (the same `merchants/{id}.staff_uids` check as the returns desk). UI: `/biz/charts` in the web app.

| Endpoint | Purpose |
|---|---|
| `POST /v1/biz/charts/measure` | one flat-lay photo (base64) + category + size label -> measured zones in cm with a confidence each, flags, and an overlay PNG. The studio calls it once per size, all at once |
| `POST /v1/biz/charts/grading-check` | `{category, sizes, explain?, product_url?}` -> grading flags and review notes. `explain: true` has the flash model word each flag in one sentence (the studio asks once, right after measuring); otherwise a template does, so editing the table is instant and costs nothing |
| `POST /v1/biz/charts` | publish: saves `charts/{id}` as a new version for the product (`product_ref`), with the grading flags at that moment |
| `GET /v1/biz/charts`, `GET /v1/biz/charts/{id}` | list and read |
| `GET /v1/biz/charts/{id}/export?unit=cm\|in` | `{json, html, unit}`: the chart as JSON and as an HTML table snippet (names are escaped) |
| `POST /v1/biz/charts/{id}/use-on-store`, `.../remove-from-store` | make a chart the live one for its `product_ref` (one live chart per product) |
| `GET /v1/store/charts/{product_ref}` | store-facing (merchant key + origin): the live chart, or an empty 204 when there is none (so a page without an override logs no error). The mock store's product pages call it and swap in the measured chart and the size widget's product |

Measuring is `fitcore.garment` (see `packages/fitcore/README.md`). Grading is `fitcore.chart.grading`: per zone, each size against the next smaller one. A value smaller than the previous size is always flagged; a step outside the expected grade for the category and zone (`fitcore/config/grading.yaml`) is flagged as too large or too small, allowing `jump_tolerance_cm` (1 cm) for measurement noise. Missing values scale the expected growth by the steps skipped.

Reviews beside the chart: with `product_url`, the studio reads the old chart and `ReviewSignals` the reader stored at `products/{sha256(url)}` (the same document the reader writes) and says, for example, "3 reviewers say it runs small at the chest; your measured chest is 2.0 cm below your old chart." A body-size old chart cannot be compared and the note says so.

Memory: a photo is decoded and warped in memory (a 12 MP photo needs a few hundred MB while measured), and four photos are measured at once, so give the API service at least 1 GiB, or limit its concurrency.

## Return simulation

Staff-only. UI: `/biz/simulate`. Nothing per body ever leaves the service: only aggregates.

| Endpoint | Purpose |
|---|---|
| `POST /v1/biz/simulate` | `{chart_id}` (a published studio chart) or `{chart: {sizes, chart_type, unit}, category}` (an upload), plus `fit_type`, `fabric`, `stretch_class`, `pre_shrunk`, `target` (reweight to the target market), `regrade`. Returns the size mix, shares per zone per size, the no-good-size share, regrade suggestions, and the population's banner. About a second on the cached population |
| `POST /v1/biz/simulate/memo` | the same request; returns the merchandiser memo. A separate call because the model takes tens of seconds, so the page shows the statistics first. Rate-limited per merchant |
| `POST /v1/biz/simulate/facts` | the exact statistics the memo is written from, so its numbers can be checked |

The population is read once per process from parquet at `POPULATION_PATH` (default `services/api/.data/population_standin_v1.parquet`; a `gs://` path works with `google-cloud-storage` installed) and generated and cached there if missing. It is a **statistical stand-in with invented distributions, not SMPL** (see `packages/fitcore/README.md`); the banner in every response says so and the page shows it permanently.

## Demo virtual try-on (one tee, one pair of trousers)

`apilib/tryon.py`, backed by `services/geometry/tryon.py`.

- `GET /api/tryon/garments` — the predefined catalog. Two entries: the
  "Demo Threads Co. Classic Crew Tee" (`kind: "top"`) and the
  "Demo Threads Co. Fitted Trousers" (`kind: "bottom"`).
- `GET /api/twin/{twin_id}/tryon/{garment_id}/glb?pose=standing` — streams one
  GLB scene holding two nodes, `body` and the garment (`demo_tee` or
  `demo_trousers`). Signed-in, owner-only (same rule as
  `GET /api/twin/{twin_id}/glb`), cached per twin **and** per garment. Both
  paths are also served without the `/api` prefix for direct Cloud Run calls.

Each garment is real 3D geometry, not an image overlay: a closed offset shell
(outer face, inner face, stitched rim at every opening) built from that twin's
own SMPL surface, so placement follows the twin by construction. Which vertices
a garment covers comes from the SMPL model's own skinning weights, so it works
for any gender and any betas. The body mesh is handed back unchanged, and both
meshes share one grounding transform so the garment cannot drift off the body.

- **Top** (`TopGarmentSpec`): a body panel plus two sleeves. Four openings —
  neckline, hem, two cuffs.
- **Bottom** (`BottomGarmentSpec`): a waistband plus two separate legs with a
  real inseam between them — never a single skirt-like tube. Three openings —
  the waistband and one hem per leg. The waistband is cut along the twin's own
  pelvis→neck axis, and each leg is cut at a distance measured along that leg's
  own hip→knee→ankle chain, so both survive a pose change.

`leg_length_fraction` and `ease_cm` are the length and fit dials: shorts,
cropped and full-length trousers are the same geometry code with one number
changed, which is how jeans or shorts get added later (a spec in
`tryon.GARMENTS` plus a blurb in `apilib/tryon.py`, no new geometry path).

It is a visualization: **no cloth simulation** (no drape, gravity, collision
solve or stretch) and **no fit verdict** — sizing answers stay with the Fit
Engine (`POST /v1/fit-query`) and the body-mesh fit heatmap (`POST /fit/mesh`).

Building a shell is an SMPL forward pass plus mesh work (~2–3 s on CPU), so
each instance keeps the last few results in memory; a cold instance rebuilds.

Verify it without the cloud:

```bash
python -m services.geometry.tryon --garment demo-threads-classic-crew-tee --out tee.glb
python -m services.geometry.tryon --garment demo-threads-fitted-trousers \
    --betas 0.4,-0.3,0.1,0,0.2,0,0,0,0,0 --out trousers.glb
pytest services/geometry/tests/test_tryon.py services/api/tests/test_tryon_api.py
```

The CLI prints the shell's measured diagnostics (watertight, boundary loops and
their ring sizes, clearance band in cm, folded-face ratio, and for a bottom the
gap between the two leg shells below the knee) and writes a GLB you can drop
into any glTF viewer. In the customer PWA, the picker at the top right of the
3D twin panel switches between **Body Twin**, **T-Shirt** and **Trousers**;
rotation, zoom and auto-rotate work on all three, and Body Twin always takes
the viewer back.

### Known limits

- No cloth simulation. The offset is a geometric ease field, so garments read
  as close-fitting knits/wovens; drape, bunching, folds at the knee and a
  hanging hem are not representable.
- Not a fit measurement. Clearance is a display parameter, not a measured
  garment dimension, so nothing here should be read as "this size fits".
- Around 1% of each shell's outer faces fold back on themselves in tight
  concave creases (armpit and under the chin for the tee; the crotch and the
  back of the knee for the trousers). The shell stays watertight; the ratio is
  reported as `folded_face_ratio`.
- The trousers have no fly, pockets, belt loops or seam allowance, and the
  waistband is a cut edge rather than a separate band.
- At the top of the inseam the two legs' offset surfaces come close enough to
  touch, as real trousers do; below the knee they are measurably apart
  (`leg_shell_min_gap_below_knee_cm`).
- The seated pose builds and stays watertight for both garments, but the tee's
  hem cut splits into five boundary loops there rather than four, so the
  endpoint defaults to standing.
