"""CLI: measure one flat-lay photo (garment + A4 sheet) without the cloud.

    python -m fitcore.garment.demo samples/garment/shirt/M.jpg --category shirt --landmarks samples/garment/shirt/M.landmarks.json
    python -m fitcore.garment.demo photo.jpg --category kurta                 # asks the model (needs a configured backend)
    python -m fitcore.garment.demo --make-samples samples/garment             # writes the SIMULATED sample photos and landmark files

--landmarks takes {"armpit_left": [x, y], ...} in photo pixels, so a photo can be measured by hand-placed points with no model.
"""

import argparse
import json
from pathlib import Path

from fitcore.garment.geometry import MeasureError
from fitcore.garment.landmarks import FileLandmarkFinder
from fitcore.garment.measure import measure_garment
from fitcore.garment.synthetic import render_flatlay
from fitcore.schemas import Category, MeasurementZone as Z

# A shirt in four sizes (the numbers on the mock store's shirt chart, plus a waist), and a mislabeled XL for the studio demo.
SHIRT = {"S": (108, 104, 44, 60, 72), "M": (114, 110, 46, 61.5, 74), "L": (120, 116, 48, 63, 76), "XL": (126, 122, 50, 64.5, 78)}
ZONES = (Z.chest, Z.waist, Z.shoulder_width, Z.sleeve_length, Z.garment_length)


def make_samples(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sets = {**{s: dict(zip(ZONES, v)) for s, v in SHIRT.items()}, "XL_mislabeled": dict(zip(ZONES, (119, 122, 50, 64.5, 78)))}
    truth = {}
    for i, (label, zones) in enumerate(sets.items()):
        photo = render_flatlay(Category.shirt, zones, seed=10 + i, tilt=0.03, noise=2.0)
        (out / f"{label}.jpg").write_bytes(photo.jpeg)
        (out / f"{label}.landmarks.json").write_text(json.dumps({k: [round(v[0], 1), round(v[1], 1)] for k, v in photo.landmarks.items()}, indent=1), encoding="utf-8")
        truth[label] = {z.value: v for z, v in zones.items()}
    (out / "truth.json").write_text(json.dumps({"note": "SIMULATED photos drawn by fitcore.garment.synthetic; the numbers are the true full-circumference or length values in cm", "category": "shirt", "sizes": truth}, indent=1), encoding="utf-8")
    print(f"Wrote {len(sets)} simulated photos, their landmark files and truth.json to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo", nargs="?")
    ap.add_argument("--category", choices=[c.value for c in Category], default="shirt")
    ap.add_argument("--landmarks", help="JSON of landmark points in photo pixels (skips the model)")
    ap.add_argument("--overlay", help="write the overlay PNG here")
    ap.add_argument("--make-samples", metavar="DIR")
    a = ap.parse_args()
    if a.make_samples:
        return make_samples(Path(a.make_samples))
    if not a.photo:
        ap.error("give a photo, or --make-samples DIR")
    try:
        m = measure_garment(Path(a.photo).read_bytes(), Category(a.category), finder=FileLandmarkFinder(a.landmarks) if a.landmarks else None)
    except MeasureError as e:
        raise SystemExit(f"Cannot measure this photo ({e.code}): {e.message}")
    truth_file = Path(a.photo).with_name("truth.json")
    truth = json.loads(truth_file.read_text(encoding="utf-8"))["sizes"].get(Path(a.photo).stem) if truth_file.is_file() else None
    print(f"{'zone':<16}{'measured cm':>12}{'confidence':>12}" + (f"{'true cm':>10}{'error':>8}" if truth else ""))
    for z, v in m.zones.items():
        row = f"{z.value:<16}{v.value_cm:>12.1f}{v.confidence:>12.2f}"
        if truth and z.value in truth:
            row += f"{truth[z.value]:>10.1f}{v.value_cm - truth[z.value]:>+8.1f}"
        print(row)
    for f in m.flags:
        print(f"flag: {f}")
    if m.unsupported:
        print("not measurable from a front flat-lay: " + ", ".join(z.value for z in m.unsupported))
    if a.overlay:
        Path(a.overlay).write_bytes(m.overlay_png)
        print(f"overlay written to {a.overlay}")


if __name__ == "__main__":
    main()
