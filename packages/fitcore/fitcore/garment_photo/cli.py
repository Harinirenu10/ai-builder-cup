"""Offline CLI for garment flat-lay photo measurement.

Runs the full pipeline against a local image file. By default this requires
network access for the Gemini keypoint call; pass --keypoints to supply
corrected/known keypoints and skip Gemini entirely for fully offline runs.

Usage:
    python -m fitcore.garment_photo.cli --image shirt.jpg --category tops --out out/
    python -m fitcore.garment_photo.cli --image shirt.jpg --category tops --keypoints kp.json --out out/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from fitcore.garment_photo.pipeline import A4NotFoundError, measure_garment_photo
from fitcore.garment_photo.schemas import GarmentCategory, KeypointXY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fitcore.garment_photo.cli")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measured Fit: Garment Flat-Lay Photo Measurement")
    parser.add_argument("--image", required=True, help="Path to garment flat-lay photo (JPEG/PNG)")
    parser.add_argument("--category", required=True, choices=[c.value for c in GarmentCategory])
    parser.add_argument("--keypoints", default=None, help="Optional path to a JSON file of corrected keypoints")
    parser.add_argument("--out", required=True, help="Directory to save output files")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    image_path = Path(args.image)
    out_dir = Path(args.out)

    if not image_path.exists() or not image_path.is_file():
        logger.error("Missing image file: %s", image_path)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    corrected_keypoints = None
    if args.keypoints:
        kp_path = Path(args.keypoints)
        if not kp_path.exists():
            logger.error("Missing keypoints file: %s", kp_path)
            return 1
        raw = json.loads(kp_path.read_text(encoding="utf-8"))
        corrected_keypoints = {name: KeypointXY(**v) for name, v in raw.items()}

    try:
        result = measure_garment_photo(
            image_bytes=image_path.read_bytes(),
            category=GarmentCategory(args.category),
            corrected_keypoints=corrected_keypoints,
        )
    except A4NotFoundError as e:
        logger.error("Error: %s", e)
        return 1
    except ValueError as e:
        logger.error("Error: %s", e)
        return 1

    (out_dir / "rectified.png").write_bytes(result.rectified_image_png)
    (out_dir / "overlay.png").write_bytes(result.overlay_image_png)
    (out_dir / "measurements.json").write_text(
        json.dumps(
            {
                "category": result.category.value,
                "a4": result.a4.model_dump(),
                "keypoints_source": result.keypoints_source,
                "keypoints": {k: v.model_dump() for k, v in result.keypoints.items()},
                "quality": result.quality.model_dump(),
                "measurements": result.measurements.model_dump(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote output to %s", out_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
