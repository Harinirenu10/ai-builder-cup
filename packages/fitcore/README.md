# fitcore

Pure-Python shared library for measured fit fashion commerce.
Provides domain schemas, Gemini LLM client wrappers, unit conversion helpers, and fit analysis routines.

## Fit engine (`fitcore.fit`)

Deterministic size and fit verdicts per zone, with wash shrinkage, pose strain, scoring and tailor alterations.
Only `explain()` and the translation in `tailor_card()` call Gemini (flash, through `fitcore.llm`).

```python
from fitcore.fit import evaluate, explain, tailor_card, render_png, apply_reviews
result = evaluate(body, product, preferences, review_adjustments, pose_deltas)
```

- Ease is `garment - body + personal_offset - review_adjustment` for garment charts; for body charts it is the ideal ease plus `chart - body`. Each zone records the path used.
- Verdicts: tight / ideal / loose (circumferences), short / ideal / long (lengths). The stretch allowance widens the low side of circumference zones. Wash and pose use the same threshold.
- Score: hard zones weigh 3, others 1; the worse of now and after-wash counts; seated strain longer than 30 min counts like tight. If a hard zone fails in every size, `recommended_size` is `None` with a warning.
- Alterations touch easy zones only (sleeve, hem, inseam, waist), capped by `config/alterations.yaml`.
- All numbers live in `fitcore/config/*.yaml` and are marked `# TUNE`. Size charts should be listed smallest first (ties go to the smaller size).
- `explain()` rejects model output that has banned body-judging words or numbers not in `explanation_facts` and returns the plain facts instead. `tailor_card()` drops a translation that changes a number, unit or size label. The Tamil PNG needs a Tamil font (Nirmala on Windows, Noto Sans Tamil elsewhere).

Demo on the simulated samples (offline unless `GOOGLE_CLOUD_PROJECT` is set):

```
python -m fitcore.fit.demo samples/product_kurta.json samples/twin.json \
    [--poses samples/poses_seated.json] [--reviews samples/reviews_kurta.json] [--card out/card]
```

Tests: `pytest packages/fitcore/tests` (offline).

## LLM backends

`LLM_BACKEND` (default `auto`), implemented in `fitcore.llm.generate_json`; callers do not change:

- `auto`: calls go to Gemma on Ollama when `OLLAMA_BASE_URL` and `OLLAMA_MODEL` are set, image parts included; on any failure Gemini is used.
- Images (bytes or genai `Part` with inline data) are sent to Ollama as base64 JPEG, downscaled to `OLLAMA_MAX_IMAGE_PX` (default 1280) so the 16k context is not filled. Keep few images per request.
- `ollama`, `vertex`, `gemini_api`: force one backend.
- Gemini uses the Developer API key (`GEMINI_API_KEY`) when set, otherwise Vertex (`GOOGLE_CLOUD_PROJECT`). Model IDs still come from `GEMINI_FLASH_MODEL` / `GEMINI_PRO_MODEL`.
- The key is dev-only and used sparingly: each key call is logged and capped per process by `GEMINI_API_MAX_CALLS` (default 30).
- Put settings in the environment or a gitignored `.env` (see `.env.example` at the repo root). Never commit keys. Body photos are privacy-sensitive: do not send real body photos to the shared Ollama relay; use `LLM_BACKEND=vertex` for them.
- Invalid JSON from Ollama gets one repair retry. The submitted demo runs on Gemini via Vertex.

## Calibrating ease (`fitcore.fit.calibrate`)

Size charts that list body and garment numbers per size give real ease values. Save each as JSON (`CalibrationChart`: category, fit_type, unit, `garment_basis` full|flat, source, sizes with `body` and `garment`), then:

```
python -m fitcore.fit.calibrate samples/calibration_example_a.json samples/calibration_example_b.json --out proposed.yaml
python -m fitcore.fit.calibrate --extract saved_chart.png    # drafts <file>.calib.json with the LLM; check every number
```

It pools garment minus body per category, fit type and zone, and prints current vs proposed min/ideal/max. A zone is proposed only with at least 3 sizes from at least 2 products. Nothing is applied: copy accepted values into `config/ease.yaml`. The two example charts are simulated. Use only pages you saved yourself from public product listings.
