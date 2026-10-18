# Real-garment validation fixtures (`fitcore.garment_photo`)

Not to be confused with `README.md` in this folder, which documents the separate **simulated**
`fitcore.garment` sample photos (`S.jpg`, `M.jpg`, `L.jpg`, `XL.jpg`, `XL_mislabeled.jpg`). This file
documents the `garment_photo_real_fixture*.csv` files instead, used by `eval/garment_photo_eval.py`
(`fitcore.garment_photo`'s evaluation harness) to validate against real, physically tape-measured
garments.

Two CSV files, same column format (`eval/garment_photo_eval.py`'s expected header):

- **`garment_photo_real_fixture.csv`** — blank template. This is the one to actually fill in and run.
  Replace `image_path` with real photo paths (relative to this folder) and fill in only the tape
  measurement columns that apply to each garment's category. See the visual measurement guide for how
  to take the photo and tape-measure each dimension.
- **`garment_photo_real_fixture.example.csv`** — **illustrative placeholder values only, not real
  measurements of any garment.** Shows what a filled-in row looks like. The `EXAMPLE_*` image paths do
  not exist, so running the eval script against this file will correctly report "missing image file"
  for every row — it is not meant to be run for a real evaluation, only to show the expected shape of a
  completed row.

Run the real evaluation with:

```bash
python eval/garment_photo_eval.py --input samples/garment/garment_photo_real_fixture.csv --output eval/garment_photo_real_results.json
```

Results get recorded in `eval/results.md` §11.4/§11.5 only after an actual run against real photos and
independent tape measurements — never fabricated ahead of time.
