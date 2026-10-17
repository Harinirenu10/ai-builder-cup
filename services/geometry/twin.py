"""Offline CLI for 3D body twin reconstruction from 4 orthogonal photos.

Executes 100% offline without Firebase, Cloud Tasks, Gemini, Ollama, or any external network calls:
1. Validates input photos locally using deterministic quality heuristics (blur, illumination).
2. Extracts 33 pose landmarks and silhouette segmentation masks via MediaPipe Pose Landmarker Heavy CPU.
3. Computes all 12 anatomical zone measurements using Ramanujan ellipse approximations.
4. Fits SMPL 3D human body mesh shape (10 betas) via PyTorch Adam on CPU.
5. Renders weak-perspective silhouette projections and generates 4 diagnostic overlay images.
6. Exports final grounded metric 3D mesh to twin.glb and validates GLTF byte integrity.
7. Outputs measurements.json with measurements, confidence scores, and fit diagnostics.

Usage:
    python -m geometry.twin --front f.jpg --back b.jpg --left l.jpg --right r.jpg --height 172 --out out/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fitcore.schemas import MeasurementZone, Measurements
from services.geometry.critic import (
    compute_view_iou,
    generate_overlay_image,
    render_weak_perspective_silhouette,
)
from services.geometry.exporter import export_fit_result_to_glb, validate_glb_bytes
from services.geometry.fitting import fit_smpl_shape
from services.geometry.landmarks import (
    MediaPipePoseExtractor,
    extract_four_view_landmarks_and_silhouettes,
)
from services.geometry.measurements import calculate_photo_measurements
from services.geometry.validator import check_local_image_quality

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("geometry.twin")


def run_offline_pipeline(
    front_path: Path | str,
    back_path: Path | str,
    left_path: Path | str,
    right_path: Path | str,
    height_cm: float,
    output_dir: Path | str,
    gender: str = "neutral",
    pose_extractor: MediaPipePoseExtractor | None = None,
) -> dict[str, Any]:
    """Execute complete deterministic offline reconstruction pipeline.

    Args:
        front_path: Path to front orthogonal view photo.
        back_path: Path to back orthogonal view photo.
        left_path: Path to left profile orthogonal view photo.
        right_path: Path to right profile orthogonal view photo.
        height_cm: Subject stature in centimeters (50.0 to 260.0).
        output_dir: Destination folder for output files.
        gender: SMPL template gender ('neutral', 'female', 'male').
        pose_extractor: Optional injected MediaPipe landmarker for offline testing.

    Returns:
        Summary dict containing output file paths, metrics, and diagnostics.

    Raises:
        ValueError: If arguments, photos, or inputs fail quality or range checks.
        FileNotFoundError: If input files do not exist.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stature range check
    if height_cm <= 50.0 or height_cm >= 260.0:
        raise ValueError(
            f"Invalid height: {height_cm} cm. Stature must be between 50.0 and 260.0 cm."
        )

    # 2. Verify photo file paths
    view_paths: dict[str, Path] = {
        "front": Path(front_path),
        "back": Path(back_path),
        "left": Path(left_path),
        "right": Path(right_path),
    }

    for view, p in view_paths.items():
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Missing required photo for {view} view: {p}")

    # 3. Read and validate images using deterministic local heuristics
    photo_bytes: dict[str, bytes] = {}
    photos_rgb: dict[str, np.ndarray] = {}

    for view, p in view_paths.items():
        raw_b = p.read_bytes()
        is_ok, issues, fix_inst = check_local_image_quality(raw_b)
        if not is_ok:
            raise ValueError(
                f"Quality check failed for {view} view ({p.name}): {fix_inst} "
                f"Issues detected: {', '.join(issues)}."
            )
        photo_bytes[view] = raw_b

        # Decode to RGB array
        nparr = np.frombuffer(raw_b, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Could not decode image for {view} view ({p.name}).")
        photos_rgb[view] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    logger.info("All 4 photos passed deterministic local quality checks.")

    # 4. Extract landmarks and segmentation masks
    logger.info("Extracting MediaPipe landmarks and silhouettes for 4 views...")
    landmark_results, binary_masks, _ = extract_four_view_landmarks_and_silhouettes(
        views=photo_bytes,
        user_height_cm=height_cm,
        extractor=pose_extractor,
    )

    # 5. Compute photo-based body measurements
    logger.info("Computing 12 zone measurements and confidence scores...")
    measurements = calculate_photo_measurements(
        landmark_results=landmark_results,
        masks=binary_masks,
        user_height_cm=height_cm,
    )

    # 6. Fit SMPL shape (10 betas)
    logger.info("Optimizing SMPL shape parameters on CPU...")
    fit = fit_smpl_shape(
        measurements=measurements,
        target_height_cm=height_cm,
        gender=gender,  # type: ignore[arg-type]
        max_steps=250,
    )
    if not fit.is_valid:
        raise RuntimeError(f"SMPL shape fitting failed: {fit.rejection_reason}")

    # 7. Generate diagnostic overlays & compute per-view IoU metrics
    logger.info("Generating diagnostic overlays and projection metrics...")
    overlay_paths: dict[str, str] = {}
    per_view_iou: dict[str, float] = {}

    for view in ("front", "back", "left", "right"):
        lm_res = landmark_results[view]
        p_mask = binary_masks[view]
        p_cols = np.where(p_mask > 0)[1]
        cx = int(round((p_cols.min() + p_cols.max()) / 2.0)) if len(p_cols) > 0 else lm_res.image_width // 2

        mesh_mask = render_weak_perspective_silhouette(
            vertices=fit.vertices,
            faces=fit.faces,
            view=view,
            image_width=lm_res.image_width,
            image_height=lm_res.image_height,
            mask_top_row=lm_res.mask_top_row,
            mask_bottom_row=lm_res.mask_bottom_row,
            center_x=cx,
        )

        iou = compute_view_iou(p_mask, mesh_mask)
        per_view_iou[view] = round(iou, 4)

        overlay_img = generate_overlay_image(photos_rgb[view], p_mask, mesh_mask)
        overlay_filename = f"overlay_{view}.png"
        overlay_target = out_dir / overlay_filename
        # Save as BGR for OpenCV
        cv2.imwrite(str(overlay_target), cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
        overlay_paths[view] = str(overlay_target)

    mean_iou = round(float(np.mean(list(per_view_iou.values()))), 4)

    # 8. Export twin.glb and validate integrity
    logger.info("Exporting binary GLB 3D avatar...")
    glb_target = out_dir / "twin.glb"
    glb_bytes = export_fit_result_to_glb(fit, output_path=glb_target, ground_mesh=True)
    is_valid, err_msg, _ = validate_glb_bytes(glb_bytes)
    if not is_valid:
        raise RuntimeError(f"GLB validation failed: {err_msg}")

    # 9. Write measurements.json
    logger.info("Writing measurements.json...")
    zones_output = {}
    for zone, m in measurements.zones.items():
        zones_output[zone.value] = {
            "value_cm": m.value_cm,
            "confidence": m.confidence,
            "source": m.source,
        }

    measurements_json_target = out_dir / "measurements.json"
    output_data = {
        "user_height_cm": round(height_cm, 1),
        "gender": gender,
        "fitted_height_cm": round(fit.fitted_height_cm, 1),
        "mean_silhouette_iou": mean_iou,
        "per_view_iou": per_view_iou,
        "betas": [round(float(b), 4) for b in fit.betas],
        "optimization": {
            "loss": round(float(fit.loss), 5),
            "steps": fit.steps,
            "converged": fit.converged,
        },
        "zones": zones_output,
        "diagnostics": {
            "mean_confidence": round(float(np.mean([m.confidence for m in measurements.zones.values()])), 2),
            "critic_refinement": "skipped_offline_mode",
            "overlay_files": {v: Path(p).name for v, p in overlay_paths.items()},
            "glb_file": glb_target.name,
        },
    }

    measurements_json_target.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

    logger.info("Offline twin generation completed successfully.")
    logger.info("Output saved to: %s", out_dir.resolve())
    return {
        "measurements_json": str(measurements_json_target),
        "glb_path": str(glb_target),
        "overlays": overlay_paths,
        "mean_iou": mean_iou,
        "zones_count": len(zones_output),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measured Fit: Deterministic Offline 3D Body Twin Generator"
    )
    parser.add_argument("--front", required=True, help="Path to front orthogonal photo (JPEG/PNG)")
    parser.add_argument("--back", required=True, help="Path to back orthogonal photo (JPEG/PNG)")
    parser.add_argument("--left", required=True, help="Path to left profile orthogonal photo (JPEG/PNG)")
    parser.add_argument("--right", required=True, help="Path to right profile orthogonal photo (JPEG/PNG)")
    parser.add_argument("--height", required=True, type=float, help="Subject stature in centimeters")
    parser.add_argument("--out", required=True, help="Directory to save output files")
    parser.add_argument(
        "--gender",
        default="neutral",
        choices=["neutral", "female", "male"],
        help="Base SMPL model template (default: neutral)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_offline_pipeline(
            front_path=args.front,
            back_path=args.back,
            left_path=args.left,
            right_path=args.right,
            height_cm=args.height,
            output_dir=args.out,
            gender=args.gender,
        )
        return 0
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        logger.error("Error: %s", e)
        return 1
    except Exception as e:
        logger.exception("Unexpected execution failure: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
