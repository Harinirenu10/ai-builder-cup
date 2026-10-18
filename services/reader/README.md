# Reader service

FastAPI + Playwright (Chromium, headless). Reads a product page, or a size-chart screenshot, into a fitcore `Product` plus `ReviewSignals`.
Run: `uvicorn main:app --port 8002` from `services/reader` (needs `pip install -e packages/fitcore playwright pillow` and `playwright install chromium`). Deploy: `make deploy-reader`. Tests: `python -m pytest services/reader/tests` (offline; real Chromium against a local server).

| Endpoint | Returns |
|---|---|
| `POST /read {url}` | `{status: ok / blocked / unsupported, reason, product, reviews, flags, cached}` |
| `POST /read-image {image (base64), page_url?}` | same shape, `reviews` is null. The fallback when a site blocks the reader |

## How a read works
1. **Safety:** only public http(s) URLs. Localhost, private and metadata addresses are refused (`READER_ALLOW_PRIVATE=1` only for tests).
2. **Cache:** `sha256(url)` in Firestore `products/{hash}`, fresh for 7 days (Firestore when `GOOGLE_CLOUD_PROJECT` is set, otherwise JSON files in `services/reader/.cache`; `READER_CACHE=memory|file|firestore` to force). Blocked results are cached too, so a site is not visited again for 7 days. Errors are not cached.
3. **Capture** (`readerlib/capture.py`): desktop viewport 1440x900, 20 s budget with a hard stop, network idle, scroll for reviews, click the first visible element matching `/size (chart|guide)|measurement/i` and screenshot the modal (if the chart panel is in the page but hidden, its text is read from the page). Collects visible text, JSON-LD, image URLs and a full-page screenshot capped at 8000 px.
4. **Blocked:** a captcha, a login wall, or HTTP 401/403/429 returns `status: "blocked"` at once (tests: well under 25 s). Nothing tries to get around it: no stealth, no retries, one visit per URL. A 404 or an unreachable host is `error`.
5. **Extraction:** one flash call (`fitcore.read.extract`, through `fitcore.llm`) fills `PageExtraction`. Text is capped at 20,000 characters so it fits a 16k-context local model.
6. **Post-processing in Python** (`fitcore.read.normalize`): units to cm, zone names to `MeasurementZone`, flat chest/waist/hip doubled, body-vs-garment decided by the chart label ("To fit chest" means body), sizes ordered, `non_monotonic:<zone>` flag when a value drops at a larger size, fabric percent cleaned, review counts capped at the number of reviews actually read.
7. **Reviews to fit:** `fitcore.fit.from_review_signals(reviews)` then `apply_reviews(signals, twin)` gives cm adjustments for `evaluate`.

Flags worth reading: `no_size_chart`, `chart_type_guessed`, `unit_inferred`, `unmapped_zone:<label>`, `fit_type_defaulted`, `fabric_percent_sum:<n>`, and on the chart `non_monotonic:<zone>`.

## CLI (no cloud)
```
python cli.py --compare                                   # extracted vs true chart values for the 5 samples, from recorded Gemini responses
python cli.py samples/products/demo_threads_tee.html      # one saved page
python cli.py --record [--only NAME]                      # re-record fixtures with the live LLM (uses LLM_BACKEND from .env)
python cli.py --url https://... --save-html samples/products/x.html   # visit one public page once and save it
```

## Samples (`samples/products/`)
- `demo_threads_{tee,denim,overshirt}.html`: pages for the fictional "Demo Threads Co." storefront. **Simulated**: invented reviews and charts; they cover a garment chart in cm, a body chart in inches with ranges, and a flat half-chest chart.
- `taylorstitch_shop_shirt.html`, `vistaprint_gildan_polo.html`: public pages saved once for testing. Third-party content, test use only. The first has a garment and a body chart in inches with fractions; the second has real reviews but no readable chart in the saved copy.
- `truth.json`: chart values transcribed by hand. Recorded Gemini responses live in `services/reader/tests/fixtures/llm/`.

Not done: reviews that need a click on a "Reviews" tab are not opened; sites that need a login or show a captcha are only reported as blocked (use `/read-image`).
