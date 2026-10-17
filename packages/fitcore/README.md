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

## Garment photo measurement (`fitcore.garment`)

One flat-lay photo (garment on a contrasting surface, a plain white A4 sheet lying flat beside it) to cm per zone.

1. `geometry.detect_a4`: finds the sheet, then locates each edge to sub-pixel accuracy (intensity profiles, a line per edge). This matters: the sheet is small next to the garment, so a 1 px corner error becomes a few percent at the garment.
2. `geometry.rectify`: a homography from the sheet's four corners puts the photo on a millimetre plane (up to 3 px per mm), whatever the camera angle or distance.
3. `geometry.segment_garment`: garment outline from colour distance to a fitted smooth model of the surface (handles shading).
4. `landmarks`: a finder places named landmarks (`armpit_left`, `hem_right`, ...): `LlmLandmarkFinder` (Gemini pro, through `fitcore.llm`), `FileLandmarkFinder` (points written in a JSON file, for samples, tests and hand-annotated photos). Points are snapped to the outline.
5. `measure.measure_garment`: deterministic maths on those points. Flat widths are doubled to circumferences. Zones a front flat-lay cannot give (`back_rise`) are reported as unsupported, never invented. An overlay PNG shows the outline, the A4 sheet, the landmarks and every measured line.

`MeasureError` codes: `bad_image`, `no_a4`, `perspective_extreme`, `no_garment`, `no_landmarks`. Per-photo flags: `landmark_missing:<name>`, `landmark_off_garment:<name>`, `implausible:<zone>`, `a4_perspective_strong`, `garment_touches_photo_edge`, `zone_not_measured:<zone>`.

**Accuracy, honestly:** on the synthetic photos (`synthetic.py`, drawn with known sizes and random perspective) the median error is 0.1 to 0.3 cm and the worst about 2.5 cm, at the strongest perspective. Gemini's landmarks on two recorded synthetic photos were within 1.4 cm of the truth. No real photographs have been measured yet. Shoot from directly above and put the sheet close to the garment: the further the garment is from the sheet, the more a small error at the sheet is magnified.

```
python -m fitcore.garment.demo samples/garment/M.jpg --category shirt --landmarks samples/garment/M.landmarks.json --overlay out.png
python -m fitcore.garment.demo --make-samples samples/garment      # rewrites the SIMULATED sample photos
```

## Grading check (`fitcore.chart`)
`check_grading(category, sizes)` flags a value below the previous size, and steps outside the expected grade (`config/grading.yaml`, `# TUNE`, with `jump_tolerance_cm` for measurement noise). `explain_flags` has the flash model write one sentence per flag; a sentence with a number that is not in the facts, a second sentence, or accusatory or body-judging words is replaced by a template. `compare_reviews` sets `ReviewSignals` beside the chart.

## Return simulation (`fitcore.sim`, `fitcore.population`, `fitcore.sim_memo`)

Run a size chart against a population of bodies before the garment is produced.

```
python -m fitcore.sim samples/product_kurta.json [--no-regrade] [--target] [--memo]
```

- **Population** (`population.py`): arrays of body measurements (cm) with a weight each, cached as parquet (a local path, or `gs://bucket/path`). **The only source built so far is a statistical stand-in**: per gender (2,000 each, fixed seed) a correlated normal over the zone measurements, with parameters I chose by hand in `config/sim.yaml`. They are invented, not survey data and **not the SMPL shape space**. The SMPL sampler (betas from a standard normal, measured by the geometry service) is not built because the geometry service is not; when it exists it fills the same `Population` with `source="smpl"`. `Population.banner()` is what the page shows, so a stand-in can never be presented as SMPL.
- **Reweighting** (`Population.reweighted`): per-gender height bins are weighted to a target height distribution and a gender mix from `config/sim.yaml` (capped per body). It works on any population.
- **Simulation** (`sim.py`): for every body, the fit engine's verdicts give the best size. This is a NumPy version of `fit.engine.evaluate` (same ease, wash, stretch allowance, scoring, rounding and tie-breaking); `tests/sim/test_sim_equivalence.py` checks it against `evaluate` body by body on random charts, fits, fabrics and chart types. The speed matters because the regrade search runs it hundreds of times: 4,000 bodies with regrade take about a second.
- **Output**: the size mix (share of bodies whose best size is each size, plus "no size recommended"), the share below / inside / above each zone's range per size (among the bodies that size is best for, as bought or after the first wash), the share with **no good size** (no size where every hard-to-alter zone is ideal, as bought and after a wash), and regrade suggestions.
- **Regrade**: a coordinate search that moves each zone of each size by up to 2 cm either way (1 cm steps, three passes) to maximise the share of bodies with a good size, with a small credit for hard zones that are ideal at each body's best size. A change that would make a size smaller than the one below it is never allowed. Each suggested change also shows the coverage it would gain on its own.
- **Memo** (`sim_memo.py`): Gemini pro writes a one-page memo from `memo_facts(result)` only (every number already in quotable form). It is used only if every number in it appears in the facts, it fits a page, and it has no accusatory or body-judging wording; otherwise a template from the same facts is used. The caveat about the synthetic population is added by code. On the live model the memo took 35 to 45 s, so the API serves it separately from the statistics.
