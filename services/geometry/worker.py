"""Worker processing pipeline for 3D body twin generation jobs.

Invoked via Cloud Tasks worker HTTP requests to POST /twin/process/{job_id}.
Orchestrates:
1. Job fetching and initial validation
2. 4-photo quality, framing, and safety checks (with needs_retake/rejected status)
3. MediaPipe Pose Landmarker Heavy pose and silhouette extraction
4. Ramanujan circumference and zone measurements computation
5. PyTorch Adam SMPL shape fitting
6. Iterative multimodal critic refinement loop and round logging
7. 3D GLB avatar export, validation, and Cloud Storage upload
8. Twin document creation and job completion
9. Strict zero-retention raw photo privacy cleanup
"""

from __future__ import annotations

import datetime
import logging
import uuid
from typing import Any, Callable

import cv2
import numpy as np

from fitcore.schemas import (
    JobDocument,
    JobStatus,
    Measurement,
    Measurements,
    MeasurementZone,
    RoundLogDocument,
    TwinDocument,
)
from services.geometry.cloud_client import (
    FirestoreClientProtocol,
    StorageClientProtocol,
    get_firestore_client,
    get_storage_client,
)
from services.geometry.critic import run_critic_refinement_loop
from services.geometry.exporter import export_fit_result_to_glb, validate_glb_bytes
from services.geometry.fitting import fit_smpl_shape
from services.geometry.landmarks import (
    MediaPipePoseExtractor,
    extract_four_view_landmarks_and_silhouettes,
)
from services.geometry.measurements import calculate_photo_measurements
from services.geometry.validator import check_chest_arm_separation, validate_photos

logger = logging.getLogger("geometry.worker")


class JobNotFoundError(Exception):
    """Raised when the requested job_id does not exist in Firestore."""
    pass


class JobValidationError(Exception):
    """Raised when job input photos or parameters fail validation."""
    def __init__(self, status: JobStatus, error_code: str | None, error_message: str | None) -> None:
        super().__init__(error_message or error_code or "Job validation failed")
        self.status = status
        self.error_code = error_code
        self.error_message = error_message


def _iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _cleanup_storage_photos(
    storage_client: StorageClientProtocol,
    photo_refs: dict[str, str],
    opt_in_keep_front: bool,
) -> None:
    """Delete raw uploaded photos from storage enforcing zero-retention privacy rules."""
    for view, ref in photo_refs.items():
        if opt_in_keep_front and view.lower() == "front":
            logger.info("Preserving front photo in storage per explicit user opt-in: %s", ref)
            continue
        try:
            storage_client.delete_file(ref)
            logger.debug("Deleted raw photo from storage (%s): %s", view, ref)
        except Exception as exc:
            logger.warning("Could not delete photo %s: %s", ref, exc)


def _cleanup_job_record_photos(
    firestore_client: FirestoreClientProtocol,
    job_id: str,
    raw_photos: Any,
    opt_in_keep_front: bool,
) -> None:
    """Strip raw photo data out of the persisted jobs/{job_id} document itself.

    _cleanup_storage_photos (above) only ever deletes files at a storage
    reference -- it does nothing when a caller submits photos as inline data
    (a data: URL in the request body's 'photos' dict, which is how every
    current caller -- services/api/main.py's create_twin_job_endpoint --
    submits them). In that case the full-resolution raw photo bytes are
    stored directly on the job document and were never being removed,
    confirmed by a real end-to-end run: all 4 photos were still present on a
    'done' job with opt_in_keep_front_photo=False. This closes that gap by
    clearing (or, for front on opt-in, trimming to front-only) the job
    document's own 'photos' field, the same zero-retention rule
    _cleanup_storage_photos already applies to actual storage blobs.
    """
    if not isinstance(raw_photos, dict) or not raw_photos:
        return
    kept = {"front": raw_photos["front"]} if (opt_in_keep_front and "front" in raw_photos) else {}
    try:
        firestore_client.update_job(job_id, {"photos": kept})
        logger.debug("Cleared raw photos from job record %s (kept front: %s)", job_id, bool(kept))
    except Exception as exc:
        logger.warning("Could not clear raw photos from job record %s: %s", job_id, exc)


def process_twin_job(
    job_id: str,
    firestore_client: FirestoreClientProtocol | None = None,
    storage_client: StorageClientProtocol | None = None,
    critic_llm_fn: Callable[..., Any] | None = None,
    pose_extractor: MediaPipePoseExtractor | None = None,
    critic_timeout_s: float | None = None,
) -> dict[str, Any]:
    """Execute the end-to-end twin generation pipeline for a Cloud Tasks job.

    Args:
        job_id: Unique job identifier in Firestore jobs/{job_id}.
        firestore_client: Optional injected Firestore client (defaults to global).
        storage_client: Optional injected Storage client (defaults to global).
        critic_llm_fn: Optional injected critic callback for testing.
        pose_extractor: Optional injected MediaPipe pose extractor for testing.
        critic_timeout_s: Optional strict overall timeout in seconds for critic review.

    Returns:
        Summary dict containing status, twin_id, final_iou, and metrics.

    Raises:
        JobNotFoundError: If job document does not exist in Firestore.
    """
    fs = firestore_client or get_firestore_client()
    st = storage_client or get_storage_client()

    # 1. Read job document from Firestore
    job_data = fs.get_job(job_id)
    if job_data is None:
        logger.error("Job %s not found in Firestore", job_id)
        raise JobNotFoundError(f"Job {job_id} not found in Firestore.")

    user_id = str(job_data.get("user_id", ""))
    height_cm = float(job_data.get("height_cm", 175.0))
    gender = str(job_data.get("gender", "neutral")).lower()
    if gender not in ("neutral", "female", "male"):
        gender = "neutral"
    opt_in_keep_front = bool(job_data.get("opt_in_keep_front_photo", False))
    age_18_confirmed = bool(job_data.get("age_18_confirmed", False))

    # Resolve photo references (from 'photos' dict, 'photo_<view>' fields, or standard storage paths)
    raw_photos = job_data.get("photos", {})
    photo_refs: dict[str, str] = {}
    for v in ("front", "back", "left", "right"):
        if isinstance(raw_photos, dict) and v in raw_photos:
            photo_refs[v] = str(raw_photos[v])
        elif f"photo_{v}" in job_data:
            photo_refs[v] = str(job_data[f"photo_{v}"])
        else:
            photo_refs[v] = f"users/{user_id}/jobs/{job_id}/{v}.jpg"

    # Track photo bytes for processing
    photo_bytes: dict[str, bytes] = {}

    try:
        # 2. Download 4 photos from storage
        for v in ("front", "back", "left", "right"):
            ref = photo_refs[v]
            try:
                photo_bytes[v] = st.download_bytes(ref)
            except Exception as e:
                fs.update_job(
                    job_id,
                    {
                        "status": JobStatus.failed.value,
                        "error_code": "MISSING_PHOTOS",
                        "error_message": f"Could not load {v} photo: {e}",
                        "updated_at": _iso_now(),
                    },
                )
                _cleanup_storage_photos(st, photo_refs, opt_in_keep_front=False)
                _cleanup_job_record_photos(fs, job_id, raw_photos, opt_in_keep_front=False)
                return {
                    "status": JobStatus.failed.value,
                    "job_id": job_id,
                    "error_code": "MISSING_PHOTOS",
                    "error_message": f"Could not load {v} photo: {e}",
                }

        # 3. Status: checking -> Photo Quality & Safety Validation
        fs.update_job(
            job_id,
            {
                "status": JobStatus.checking.value,
                "current_step": "validating_photos",
                "updated_at": _iso_now(),
            },
        )

        validation_outcome = validate_photos(
            front=photo_bytes["front"],
            back=photo_bytes["back"],
            left=photo_bytes["left"],
            right=photo_bytes["right"],
            age_18_confirmed=age_18_confirmed,
        )

        if not validation_outcome.can_proceed:
            logger.warning(
                "Job %s validation failed: status=%s, code=%s, msg=%s",
                job_id,
                validation_outcome.status.value,
                validation_outcome.error_code,
                validation_outcome.error_message,
            )
            fs.update_job(
                job_id,
                {
                    "status": validation_outcome.status.value,
                    "error_code": validation_outcome.error_code,
                    "error_message": validation_outcome.error_message,
                    "current_step": "validation_failed",
                    "updated_at": _iso_now(),
                },
            )
            # Enforce privacy cleanup upon validation failure
            _cleanup_storage_photos(st, photo_refs, opt_in_keep_front=False)
            _cleanup_job_record_photos(fs, job_id, raw_photos, opt_in_keep_front=False)
            return {
                "status": validation_outcome.status.value,
                "job_id": job_id,
                "error_code": validation_outcome.error_code,
                "error_message": validation_outcome.error_message,
            }

        # 4. Status: measuring -> Landmark and measurement extraction
        fs.update_job(
            job_id,
            {
                "status": JobStatus.measuring.value,
                "current_step": "extracting_measurements",
                "updated_at": _iso_now(),
            },
        )

        landmark_results, binary_masks, _ = extract_four_view_landmarks_and_silhouettes(
            views=photo_bytes,
            user_height_cm=height_cm,
            extractor=pose_extractor,
        )

        # 4b. Deterministic chest-specific gate: confirm the arms are genuinely
        # separated from the torso at the chest row before trusting the chest
        # measurement. Runs here (not in validate_photos) because it needs the
        # segmentation mask and landmarks, which only exist after MediaPipe
        # extraction. See services/geometry/validator.py::check_chest_arm_separation
        # for why this exists: no deterministic width-window fix can correct a
        # chest measurement when the arm and torso share the same contiguous
        # silhouette region, confirmed via a synthetic SMPL ground-truth round
        # trip and this pipeline's own real photo testing.
        front_lm = landmark_results["front"]
        l_sh = front_lm.get("LEFT_SHOULDER")
        r_sh = front_lm.get("RIGHT_SHOULDER")
        l_hip = front_lm.get("LEFT_HIP")
        r_hip = front_lm.get("RIGHT_HIP")
        if l_sh and r_sh and l_hip and r_hip:
            chest_gap_ok, chest_gap_msg = check_chest_arm_separation(
                front_mask=binary_masks["front"],
                y_shoulders_px=(l_sh.pixel_y + r_sh.pixel_y) / 2.0,
                y_hips_px=(l_hip.pixel_y + r_hip.pixel_y) / 2.0,
                l_shoulder_x_px=l_sh.pixel_x,
                r_shoulder_x_px=r_sh.pixel_x,
                l_hip_x_px=l_hip.pixel_x,
                r_hip_x_px=r_hip.pixel_x,
                cm_per_pixel=front_lm.cm_per_pixel,
            )
            if not chest_gap_ok:
                logger.warning("Job %s chest arm-separation gate failed: %s", job_id, chest_gap_msg)
                fs.update_job(
                    job_id,
                    {
                        "status": JobStatus.needs_retake.value,
                        "error_code": "PHOTO_QUALITY_CHECK_FAILED",
                        "error_message": f"Front: {chest_gap_msg}",
                        "current_step": "validation_failed",
                        "updated_at": _iso_now(),
                    },
                )
                _cleanup_storage_photos(st, photo_refs, opt_in_keep_front=False)
                _cleanup_job_record_photos(fs, job_id, raw_photos, opt_in_keep_front=False)
                return {
                    "status": JobStatus.needs_retake.value,
                    "job_id": job_id,
                    "error_code": "PHOTO_QUALITY_CHECK_FAILED",
                    "error_message": f"Front: {chest_gap_msg}",
                }

        measurements = calculate_photo_measurements(
            landmark_results=landmark_results,
            masks=binary_masks,
            user_height_cm=height_cm,
        )

        # Decode photos to RGB for critic overlay generation
        photos_rgb: dict[str, np.ndarray] = {}
        for v, b in photo_bytes.items():
            nparr = np.frombuffer(b, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                photos_rgb[v] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 5. Status: fitting -> PyTorch Adam SMPL shape fitting
        fs.update_job(
            job_id,
            {
                "status": JobStatus.fitting.value,
                "current_step": "fitting_smpl_shape",
                "updated_at": _iso_now(),
            },
        )

        initial_fit = fit_smpl_shape(
            measurements=measurements,
            target_height_cm=height_cm,
            gender=gender,  # type: ignore[arg-type]
            max_steps=200,
        )

        # 6. Status: refining -> Critic refinement loop
        fs.update_job(
            job_id,
            {
                "status": JobStatus.refining.value,
                "current_step": "critic_refinement_loop",
                "updated_at": _iso_now(),
            },
        )

        best_fit, round_logs, _ = run_critic_refinement_loop(
            initial_fit=initial_fit,
            target_measurements=measurements,
            target_height_cm=height_cm,
            gender=gender,
            landmark_results=landmark_results,
            photo_masks=binary_masks,
            photos_rgb=photos_rgb,
            critic_llm_fn=critic_llm_fn,
            max_rounds=5,
            storage_client=st,
            overlay_path_prefix=f"users/{user_id}/jobs/{job_id}/critic_overlays",
            critic_timeout_s=critic_timeout_s,
        )

        # Log each round to Firestore jobs/{job_id}/rounds/{round_num}
        for r_doc in round_logs:
            fs.create_round_log(job_id=job_id, round_num=r_doc.round, log_data=r_doc)

        glb_bytes = export_fit_result_to_glb(best_fit, ground_mesh=True)
        is_valid, err_msg, _ = validate_glb_bytes(glb_bytes)
        if not is_valid:
            raise RuntimeError(f"GLB validation failed: {err_msg}")

        # 8. Upload GLB to Cloud Storage
        twin_id = f"twin_{uuid.uuid4().hex[:12]}"
        glb_storage_path = f"users/{user_id}/twins/{twin_id}/twin.glb"
        glb_storage_url = st.upload_bytes(
            destination_path=glb_storage_path,
            data=glb_bytes,
            content_type="model/gltf-binary",
        )

        # 9. Build final measurements model for TwinDocument
        final_zones: dict[MeasurementZone, Measurement] = {}
        for z, m in measurements.zones.items():
            fitted_val = best_fit.mesh_measurements.get(z, m.value_cm)
            final_zones[z] = Measurement(
                value_cm=round(float(fitted_val), 1),
                confidence=m.confidence,
                source="smpl_fit",
            )
        for z, val in best_fit.mesh_measurements.items():
            if z not in final_zones:
                final_zones[z] = Measurement(
                    value_cm=round(float(val), 1),
                    confidence=0.90,
                    source="smpl_fit",
                )
        final_measurements = Measurements(zones=final_zones)

        final_mean_iou = round(float(round_logs[-1].mean_iou_after), 4) if round_logs else None

        twin_doc = TwinDocument(
            id=twin_id,
            user_id=user_id,
            gender=gender,  # type: ignore[arg-type]
            height_cm=round(height_cm, 1),
            betas=[round(float(b), 4) for b in best_fit.betas],
            measurements=final_measurements,
            glb_storage_url=glb_storage_url,
            final_mean_iou=final_mean_iou,
            created_at=_iso_now(),
        )
        fs.create_twin(twin_id=twin_id, twin_data=twin_doc)

        # 10. Update JobDocument to Done
        fs.update_job(
            job_id,
            {
                "status": JobStatus.done.value,
                "current_step": "completed",
                "twin_id": twin_id,
                "updated_at": _iso_now(),
            },
        )

        # 11. Enforce Privacy Cleanup: delete raw photos
        _cleanup_storage_photos(st, photo_refs, opt_in_keep_front=opt_in_keep_front)
        _cleanup_job_record_photos(fs, job_id, raw_photos, opt_in_keep_front=opt_in_keep_front)

        return {
            "status": JobStatus.done.value,
            "job_id": job_id,
            "twin_id": twin_id,
            "glb_storage_url": glb_storage_url,
            "final_mean_iou": final_mean_iou,
            "rounds_executed": len(round_logs),
        }

    except Exception as exc:
        logger.exception("Unexpected error in twin job %s: %s", job_id, exc)
        fs.update_job(
            job_id,
            {
                "status": JobStatus.failed.value,
                "error_code": "PROCESSING_ERROR",
                "error_message": str(exc),
                "current_step": "failed",
                "updated_at": _iso_now(),
            },
        )
        # Attempt privacy cleanup even on unexpected crash
        _cleanup_storage_photos(st, photo_refs, opt_in_keep_front=opt_in_keep_front)
        _cleanup_job_record_photos(fs, job_id, raw_photos, opt_in_keep_front=opt_in_keep_front)
        return {
            "status": JobStatus.failed.value,
            "job_id": job_id,
            "error_code": "PROCESSING_ERROR",
            "error_message": str(exc),
        }
