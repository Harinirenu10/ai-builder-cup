"""CLI: python -m fitcore.fit.demo samples/product_kurta.json samples/twin.json

Runs on local files. Gemini is used for the explanation and card translation only when a backend
is configured (GOOGLE_CLOUD_PROJECT, GEMINI_API_KEY or OLLAMA_BASE_URL); otherwise the plain facts are printed.
"""

import argparse
import json
import os
from pathlib import Path

from fitcore.fit import reviews
from fitcore.fit.engine import evaluate
from fitcore.fit.explain import explain
from fitcore.fit.tailor import render_png, tailor_card
from fitcore.schemas import FitResult, Measurements, PoseDeltas, Product, ReviewSignal


def _load(path: str) -> dict | list:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def print_table(result: FitResult) -> None:
    zones = sorted({z for s in result.sizes for z in s.zones}, key=lambda z: z.value)
    head = f"{'zone':<16}" + "".join(f"{s.size:>26}" for s in result.sizes)
    print(head)
    print("-" * len(head))
    for zone in zones:
        row = f"{zone.value + ('*' if next(s.zones[zone].hard for s in result.sizes if zone in s.zones) else ''):<16}"
        for s in result.sizes:
            z = s.zones.get(zone)
            if z is None:
                row += f"{'-':>26}"
                continue
            cell = f"{z.ease_now_cm:+.1f}/{z.ease_after_wash_cm:+.1f} {z.verdict.value}"
            if z.verdict_after_wash != z.verdict:
                cell += f">{z.verdict_after_wash.value}"
            if any(p.strain for p in z.poses.values()):
                cell += " !pose"
            row += f"{cell:>26}"
        print(row)
    print("-" * len(head))
    print(f"{'score':<16}" + "".join(f"{s.score:>26.1f}" for s in result.sizes))
    print("cells: ease now / ease after wash, verdict (> verdict after wash); * = hard to alter\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("product")
    ap.add_argument("twin")
    ap.add_argument("--poses", help="PoseDeltas JSON")
    ap.add_argument("--reviews", help="list of ReviewSignal JSON")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--card", help="write tailor card JSON and PNG with this path prefix")
    args = ap.parse_args()

    product = Product.model_validate(_load(args.product))
    twin = Measurements.model_validate(_load(args.twin))
    poses = PoseDeltas.model_validate(_load(args.poses)) if args.poses else None
    adjustments, notes = None, []
    if args.reviews:
        signals = [ReviewSignal.model_validate(s) for s in _load(args.reviews)]
        adjustments, notes = reviews.apply(signals, twin)

    print(f"{product.title} ({product.category.value}, {product.fit_type.value} fit)  [sample data is simulated]\n")
    result = evaluate(twin, product, review_adjustments=adjustments, pose_deltas=poses)
    print_table(result)
    for n in notes:
        print(f"review: {n}")

    if result.recommended_size:
        print(f"Recommended size: {result.recommended_size}  (confidence {result.confidence:.2f})")
    else:
        print("Recommended size: none")
    print("Alterations: " + ("; ".join(a.text for a in result.alterations) or "none"))
    for w in result.warnings:
        print(f"warning [{w.kind}]: {w.message}")

    from fitcore import llm  # loads .env

    online = bool(os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GEMINI_API_KEY") or os.environ.get("OLLAMA_BASE_URL"))
    print("\nExplanation:" if online else "\nExplanation (offline, plain facts):")
    print(explain(result, args.lang) if online else " ".join(result.explanation_facts))

    if args.card and result.recommended_size:
        card = tailor_card(result, twin, ["en", "ta"] if online else ["en"])
        Path(args.card + ".json").write_text(card.model_dump_json(indent=2), encoding="utf-8")
        png = render_png(card, args.card + ".png")
        print(f"\nTailor card written: {args.card}.json, {png}")


if __name__ == "__main__":
    main()
