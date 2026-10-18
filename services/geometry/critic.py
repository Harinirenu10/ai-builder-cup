"""Multimodal Gemini critic refinement loop for 3D body twin fitting.

Compares SMPL 3D mesh against four-view photo silhouettes (front, back, left, right),
generates diagnostic contour overlays, computes per-view IoU, and executes up to 5
refinement rounds via structured Gemini critique.

Refinement rules (Prompt 1 specification):
- Fixed weak-perspective cameras: yaw 0 deg (front), 180 deg (back), 90 deg (left), 270 deg (right).
- CPU-only contour and silhouette rasterization (no OpenGL or GPU).
- Gemini critique schema: regions, issues, causes, severity, trust, and stop flag.
- model_error suggestions with trust >= 0.5 adjust target dimensions by at most
  min(2.0, severity * 3.0) cm in suggested direction.
- Other causes (loose clothing, pose, occlusion, hair) halve zone confidence.
- Acceptance rule: candidate accepted only if mean IoU improves by at least +0.002.
  Otherwise, rollback to best candidate and halve step size.
- Application decides acceptance and logs every round as a RoundLogDocument.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import logging
import math
import os
import time
from typing import Any, Callable
import cv2
import numpy as np

logger = logging.getLogger("geometry.critic")

from fitcore.schemas import (
    CriticCause,
    CriticIssue,
    CriticRegion,
    CriticReport,
    MeasurementZone,
    Measurement,
    Measurements,
    RoundLogDocument,
)
from services.geometry.fitting import FitResult, fit_smpl_shape
from services.geometry.landmarks import ViewLandmarkResult

# Map orthogonal view names to weak-perspective yaw angles
VIEW_YAW_ANGLES: dict[str, float] = {
    "front": 0.0,
    "back": math.pi,
    "left": math.pi / 2.0,
    "right": -math.pi / 2.0,
}

# Map critic region enum to body measurement zone
REGION_TO_ZONE: dict[CriticRegion, MeasurementZone] = {
    CriticRegion.shoulders: MeasurementZone.shoulder_width,
    CriticRegion.chest: MeasurementZone.chest,
    CriticRegion.waist: MeasurementZone.waist,
    CriticRegion.hips: MeasurementZone.hip,
    CriticRegion.thighs: MeasurementZone.thigh,
    CriticRegion.arms: MeasurementZone.sleeve_length,
    CriticRegion.torso_length: MeasurementZone.torso_length,
    CriticRegion.legs: MeasurementZone.inseam,
    CriticRegion.leg_length: MeasurementZone.inseam,
}


def render_weak_perspective_silhouette(
    vertices: np.ndarray,
    faces: np.ndarray,
    view: str,
    image_width: int,
    image_height: int,
    mask_top_row: int,
    mask_bottom_row: int,
    center_x: int | None = None,
) -> np.ndarray:
    """Rasterize SMPL mesh to a binary silhouette on CPU via weak-perspective camera.

    Args:
        vertices: (6890, 3) float32 coordinates in meters.
        faces: (13776, 3) int32 triangle index array.
        view: One of 'front', 'back', 'left', 'right'.
        image_width: Canvas width in pixels.
        image_height: Canvas height in pixels.
        mask_top_row: Vertical row of crown in target photo.
        mask_bottom_row: Vertical row of sole/heel in target photo.
        center_x: Horizontal center in pixels (defaults to image_width // 2).

    Returns:
        (image_height, image_width) uint8 binary silhouette mask (0 or 1).
    """
    yaw = VIEW_YAW_ANGLES.get(view.lower().strip(), 0.0)

    # 1. Rotate vertices around Y-axis by camera yaw
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    x_rot = x * cos_y + z * sin_y
    y_rot = y

    # 2. Scale and translate to match photo bounding box
    mesh_y_max = float(np.max(y_rot))
    mesh_y_min = float(np.min(y_rot))
    mesh_h_m = max(1e-4, mesh_y_max - mesh_y_min)

    target_h_px = max(10, mask_bottom_row - mask_top_row + 1)
    scale_y = target_h_px / mesh_h_m

    # Project to pixel coordinates: image Y grows downwards
    v_px = mask_top_row + (mesh_y_max - y_rot) * scale_y

    cx = center_x if center_x is not None else image_width // 2
    u_px = cx + x_rot * scale_y

    pts_2d = np.stack([u_px, v_px], axis=-1).astype(np.int32)
    triangles = pts_2d[faces]

    # 3. CPU rasterization via cv2.fillPoly
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.fillPoly(mask, triangles, 1)
    return mask


def compute_view_iou(photo_mask: np.ndarray, mesh_mask: np.ndarray) -> float:
    """Calculate Intersection-over-Union between photo and projected mesh masks."""
    if photo_mask.shape != mesh_mask.shape:
        mesh_mask = cv2.resize(mesh_mask, (photo_mask.shape[1], photo_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    p_bin = (photo_mask > 0).astype(bool)
    m_bin = (mesh_mask > 0).astype(bool)

    intersection = int(np.count_nonzero(p_bin & m_bin))
    union = int(np.count_nonzero(p_bin | m_bin))

    if union == 0:
        return 0.0
    return float(intersection / union)


def generate_overlay_image(
    photo_rgb: np.ndarray,
    photo_mask: np.ndarray,
    mesh_mask: np.ndarray,
) -> np.ndarray:
    """Generate diagnostic visual overlay.

    Aesthetics:
    - Darkened photo background (50% brightness).
    - Cyan outline: Ground-truth photo segmentation mask contour.
    - Magenta outline: Current SMPL 3D twin silhouette contour.

    Args:
        photo_rgb: (H, W, 3) uint8 photo in RGB space.
        photo_mask: (H, W) uint8 binary photo mask.
        mesh_mask: (H, W) uint8 binary mesh silhouette.

    Returns:
        (H, W, 3) uint8 composite RGB image.
    """
    h, w = photo_rgb.shape[:2]
    if photo_mask.shape[:2] != (h, w):
        photo_mask = cv2.resize(photo_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if mesh_mask.shape[:2] != (h, w):
        mesh_mask = cv2.resize(mesh_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Darken photo
    darkened = (photo_rgb.astype(np.float32) * 0.50).clip(0, 255).astype(np.uint8)

    # Find contours
    p_contours, _ = cv2.findContours((photo_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m_contours, _ = cv2.findContours((mesh_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = darkened.copy()

    # Cyan in RGB: (0, 255, 255)
    cv2.drawContours(overlay, p_contours, -1, (0, 255, 255), 2)
    # Magenta in RGB: (255, 0, 255)
    cv2.drawContours(overlay, m_contours, -1, (255, 0, 255), 2)

    return overlay


def run_critic_refinement_loop(
    initial_fit: FitResult,
    target_measurements: Measurements,
    target_height_cm: float,
    gender: str,
    landmark_results: dict[str, ViewLandmarkResult],
    photo_masks: dict[str, np.ndarray],
    photos_rgb: dict[str, np.ndarray] | None = None,
    critic_llm_fn: Callable[[dict[str, np.ndarray], Measurements], CriticReport] | None = None,
    max_rounds: int = 5,
    min_iou_delta: float = 0.002,
    storage_client: Any | None = None,
    overlay_path_prefix: str | None = None,
    critic_timeout_s: float | None = None,
) -> tuple[FitResult, list[RoundLogDocument], dict[str, np.ndarray]]:
    """Execute iterative Gemini multimodal critic refinement loop.

    Args:
        initial_fit: Base SMPL fit from Phase 5.
        target_measurements: Starting photo measurements.
        target_height_cm: Subject stature in centimeters.
        gender: 'neutral', 'female', or 'male'.
        landmark_results: Extracted MediaPipe landmarks per view.
        photo_masks: Cleaned binary silhouettes per view.
        photos_rgb: Optional original RGB photos (or synthesized images).
        critic_llm_fn: Optional callable returning CriticReport (dependency-injected for offline tests).
        max_rounds: Maximum iterations (default 5).
        min_iou_delta: Minimum IoU improvement required to accept a round (default +0.002).
        storage_client: Optional object implementing upload_bytes(destination_path, data,
            content_type) (e.g. services.geometry.cloud_client.StorageClientProtocol). When
            provided along with overlay_path_prefix, each round's overlay images are uploaded
            and their storage URLs are recorded on that round's RoundLogDocument. When omitted
            (the default, used by offline/CLI and tests), overlay_urls stays empty -- overlays
            exist only as in-memory arrays and are not persisted anywhere to have a URL/path.
        overlay_path_prefix: Optional storage path prefix (e.g. "users/{uid}/jobs/{job_id}/
            critic_overlays") under which this round's per-view overlay PNGs are uploaded.

    Returns:
        (best_fit_result, round_logs, final_overlays)
    """

    def _persist_overlays(overlays: dict[str, np.ndarray], round_num: int) -> dict[str, str]:
        if storage_client is None or overlay_path_prefix is None:
            return {}
        urls: dict[str, str] = {}
        for view_name, overlay_rgb in overlays.items():
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(".png", overlay_bgr)
            if not ok:
                continue
            dest = f"{overlay_path_prefix}/round_{round_num}_{view_name}.png"
            try:
                urls[view_name] = storage_client.upload_bytes(
                    destination_path=dest, data=encoded.tobytes(), content_type="image/png"
                )
            except Exception as exc:
                logger.warning("Failed to upload critic overlay %s: %s", dest, exc)
        return urls
    views = [v for v in ["front", "back", "left", "right"] if v in landmark_results and v in photo_masks]
    if "front" not in views:
        raise ValueError("Front view is required for critic refinement loop.")

    # Synthesize neutral RGB backgrounds if not provided
    if photos_rgb is None:
        photos_rgb = {}
        for v in views:
            mask = photo_masks[v]
            synth = np.full((mask.shape[0], mask.shape[1], 3), 120, dtype=np.uint8)
            synth[mask > 0] = [180, 160, 150]
            photos_rgb[v] = synth

    def _render_all_views(fit: FitResult) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, np.ndarray]]:
        per_view_iou: dict[str, float] = {}
        mesh_masks: dict[str, np.ndarray] = {}
        overlays: dict[str, np.ndarray] = {}

        for v in views:
            lm_res = landmark_results[v]
            # Compute horizontal centroid of photo mask to ensure perfect camera alignment
            p_mask = photo_masks[v]
            p_cols = np.where(p_mask > 0)[1]
            cx = int(round((p_cols.min() + p_cols.max()) / 2.0)) if len(p_cols) > 0 else lm_res.image_width // 2

            m_mask = render_weak_perspective_silhouette(
                vertices=fit.vertices,
                faces=fit.faces,
                view=v,
                image_width=lm_res.image_width,
                image_height=lm_res.image_height,
                mask_top_row=lm_res.mask_top_row,
                mask_bottom_row=lm_res.mask_bottom_row,
                center_x=cx,
            )
            mesh_masks[v] = m_mask
            iou = compute_view_iou(photo_masks[v], m_mask)
            per_view_iou[v] = round(iou, 4)

            ov = generate_overlay_image(photos_rgb[v], photo_masks[v], m_mask)
            overlays[v] = ov

        return per_view_iou, mesh_masks, overlays

    # Baseline evaluation
    best_fit = initial_fit
    best_per_view_iou, _, best_overlays = _render_all_views(best_fit)
    best_mean_iou = float(np.mean(list(best_per_view_iou.values())))

    # State preservation: best_measurements is retained on rejection
    best_measurements = Measurements(
        zones={z: Measurement(**m.model_dump()) for z, m in target_measurements.zones.items()}
    )
    step_size = 1.0
    round_logs: list[RoundLogDocument] = []

    for round_num in range(1, max_rounds + 1):
        t0 = time.time()
        mean_iou_before = best_mean_iou

        # 1. Obtain Critic Report (via injected hook or default LLM wrapper)
        critic_error_occurred = False
        critic_notes = ""
        critic_report: CriticReport | None = None

        timeout_limit = (
            critic_timeout_s
            if critic_timeout_s is not None
            else float(os.environ.get("CRITIC_TIMEOUT_S", "25.0"))
        )

        def _get_critic_report() -> CriticReport | None:
            if critic_llm_fn is not None:
                return critic_llm_fn(best_overlays, best_measurements)
            from fitcore.llm import generate_json
            critic_system = (
                "You are an expert fashion photogrammetry critic evaluating a 3D digital twin avatar. "
                "Compare the magenta mesh silhouette outline against the cyan photo segmentation outline "
                "across the provided views. Identify regional discrepancies, attribute their causes "
                "(model_error, loose_clothing, pose, occlusion, or hair), and provide an objective summary."
            )
            measurement_table = "\n".join(
                f"{zone.value}: {m.value_cm} cm (confidence {m.confidence})"
                for zone, m in best_measurements.zones.items()
            )
            critic_parts: list[Any] = [
                "Review the attached orthogonal view overlays (one per labeled view) and "
                "critique the mesh-vs-photo silhouette fit per region.\n\n"
                "Current measurement table (photo/fit-derived target values the mesh is being "
                "fitted to), for context on which regions may already be under-trusted:\n"
                f"{measurement_table}"
            ]
            for v in views:
                critic_parts.append(f"View: {v}")
                overlay_bgr = cv2.cvtColor(best_overlays[v], cv2.COLOR_RGB2BGR)
                ok, encoded = cv2.imencode(".jpg", overlay_bgr)
                if ok:
                    critic_parts.append(encoded.tobytes())

            return generate_json(
                role="pro",
                parts=critic_parts,
                schema=CriticReport,
                system=critic_system,
            )

        try:
            if timeout_limit > 0:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                try:
                    future = executor.submit(_get_critic_report)
                    critic_report = future.result(timeout=timeout_limit)
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)
            else:
                critic_report = _get_critic_report()
            critic_notes = critic_report.summary if critic_report else ""
        except (concurrent.futures.TimeoutError, TimeoutError) as e:
            critic_error_occurred = True
            err_detail = str(e).strip()
            if err_detail:
                critic_notes = f"Critic review unavailable this round: {err_detail} (overall limit {timeout_limit:.1f}s exceeded)"
            else:
                critic_notes = f"Critic review unavailable this round: The read operation timed out (overall limit {timeout_limit:.1f}s exceeded)"
            logger.warning(
                "Critic review timed out after %.1fs for round %d (continuing with current best fit): %s",
                timeout_limit, round_num, critic_notes,
            )
        except Exception as e:
            critic_error_occurred = True
            # Clearly marked as UNAVAILABLE, not a failure of the fit itself: the already
            # -computed SMPL fit (best_fit) is real and valid, this only means no further
            # multimodal critique could be obtained this round. Logged for ops visibility
            # (previously silent outside this round's own Firestore record).
            critic_notes = f"Critic review unavailable this round: {e}"
            logger.warning(
                "Critic review unavailable for round %d (continuing with current best fit): %s",
                round_num, e,
            )

        critic_latency_ms = int((time.time() - t0) * 1000)

        # Handle critic exception or missing report: stop refining and return the current
        # best (already-computed, already-validated) fit as-is -- never fabricate a critic
        # verdict or treat this as a fit failure.
        if critic_error_occurred or critic_report is None:
            round_logs.append(
                RoundLogDocument(
                    round=round_num,
                    mean_iou_before=round(mean_iou_before, 4),
                    mean_iou_after=round(mean_iou_before, 4),
                    iou_delta=0.0,
                    accepted=False,
                    step_size=round(step_size, 3),
                    zone_adjustments={},
                    per_view_iou=best_per_view_iou,
                    overlay_urls=_persist_overlays(best_overlays, round_num),
                    critic_notes=critic_notes,
                    critic_latency_ms=critic_latency_ms,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                )
            )
            break

        # 2. Check early stopping condition
        if critic_report.stop:
            round_logs.append(
                RoundLogDocument(
                    round=round_num,
                    mean_iou_before=round(mean_iou_before, 4),
                    mean_iou_after=round(mean_iou_before, 4),
                    iou_delta=0.0,
                    accepted=False,
                    step_size=round(step_size, 3),
                    zone_adjustments={},
                    per_view_iou=best_per_view_iou,
                    overlay_urls=_persist_overlays(best_overlays, round_num),
                    critic_notes=critic_report.summary,
                    critic_latency_ms=critic_latency_ms,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                )
            )
            break

        # 3. Create fresh candidate measurements from current best
        candidate_measurements = Measurements(
            zones={z: Measurement(**m.model_dump()) for z, m in best_measurements.zones.items()}
        )

        # Parse feedback and aggregate adjustments per zone (averages contradictory feedback)
        zone_deltas: dict[MeasurementZone, list[float]] = {}
        for fb in critic_report.regions:
            zone = REGION_TO_ZONE.get(fb.region)
            if not zone or zone not in candidate_measurements.zones:
                continue

            current_m = candidate_measurements.zones[zone]

            if fb.cause == CriticCause.model_error and fb.trust >= 0.5:
                # model_error rule: adjust target dimension by at most min(2.0, severity * 3.0) cm
                max_delta = min(2.0, fb.severity * 3.0) * step_size

                delta = 0.0
                if fb.issue in (CriticIssue.too_wide, CriticIssue.too_long):
                    delta = -max_delta
                elif fb.issue in (CriticIssue.too_narrow, CriticIssue.too_short):
                    delta = +max_delta

                if abs(delta) > 0.05:
                    if zone not in zone_deltas:
                        zone_deltas[zone] = []
                    zone_deltas[zone].append(delta)

            elif fb.cause != CriticCause.model_error:
                # Other causes (loose clothing, pose, occlusion, hair): halve fitting weight & confidence
                new_conf = max(0.10, current_m.confidence * 0.5)
                candidate_measurements.zones[zone] = Measurement(
                    value_cm=current_m.value_cm,
                    confidence=round(new_conf, 2),
                    source=current_m.source,
                )

        # Apply averaged deltas to candidate measurements
        applied_adjustments: dict[MeasurementZone, float] = {}
        for zone, deltas in zone_deltas.items():
            avg_delta = float(np.mean(deltas))
            current_m = candidate_measurements.zones[zone]
            new_val = max(10.0, current_m.value_cm + avg_delta)
            candidate_measurements.zones[zone] = Measurement(
                value_cm=round(new_val, 1),
                confidence=current_m.confidence,
                source=current_m.source,
            )
            applied_adjustments[zone] = round(avg_delta, 2)

        # 4. Refit candidate model with adjusted targets
        candidate_fit = fit_smpl_shape(
            measurements=candidate_measurements,
            target_height_cm=target_height_cm,
            gender=gender,
            max_steps=50,
            lr=0.05,
        )

        cand_per_view_iou, _, cand_overlays = _render_all_views(candidate_fit)
        mean_iou_after = float(np.mean(list(cand_per_view_iou.values())))
        iou_delta = mean_iou_after - best_mean_iou

        # 5. Application acceptance rule: must improve mean IoU by >= min_iou_delta (+0.002)
        accepted = (iou_delta >= min_iou_delta) and candidate_fit.is_valid

        round_logs.append(
            RoundLogDocument(
                round=round_num,
                mean_iou_before=round(mean_iou_before, 4),
                mean_iou_after=round(mean_iou_after, 4),
                iou_delta=round(iou_delta, 4),
                accepted=accepted,
                step_size=round(step_size, 3),
                zone_adjustments=applied_adjustments,
                per_view_iou=cand_per_view_iou if accepted else best_per_view_iou,
                overlay_urls=_persist_overlays(cand_overlays, round_num),
                critic_notes=critic_report.summary,
                critic_latency_ms=critic_latency_ms,
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )
        )

        if accepted:
            # Retain accepted candidate and its measurements
            best_fit = candidate_fit
            best_measurements = candidate_measurements
            best_per_view_iou = cand_per_view_iou
            best_mean_iou = mean_iou_after
            best_overlays = cand_overlays
        else:
            # Reject and rollback to best candidate, halving step size
            step_size = step_size * 0.5
            if step_size < 0.0625:
                # Step size vanished; stop early
                break

    return best_fit, round_logs, best_overlays
