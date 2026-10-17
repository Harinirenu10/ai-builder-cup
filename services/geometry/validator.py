"""Photo validation and privacy enforcement for body measurement pipeline.

Validates 4 orthogonal body photos (front, back, left, right) using a single
Gemini Flash call and local image quality heuristics (blur, brightness).

Enforces:
- 18+ age consent requirement.
- Rejection of child imagery, nudity, or underwear.
- Retake requests for quality or framing deficiencies.
- Strict ephemeral file cleanup (unlinking local temporary photos).
  Note: OS memory pagination and interpreter garbage collection cannot
  guarantee complete RAM zeroing.
"""

from __future__ import annotations

import base64
import contextlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Generator, Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field

from fitcore.llm import generate_json
from fitcore.schemas import JobStatus
from services.geometry.measurements import _interpolated_taper_window, _TAPER_MARGIN_FACTOR

logger = logging.getLogger("geometry.validator")

# Supported views
PhotoView = Literal["front", "back", "left", "right"]

# Allowed quality and framing issues
ValidationIssue = Literal[
    "not_full_body",
    "arms_against_body",
    "loose_clothing",
    "poor_lighting",
    "multiple_people",
    "wrong_view",
    "head_or_feet_cropped",
]

ALLOWED_ISSUES: set[str] = {
    "not_full_body",
    "arms_against_body",
    "loose_clothing",
    "poor_lighting",
    "multiple_people",
    "wrong_view",
    "head_or_feet_cropped",
}


class ViewValidationResult(BaseModel):
    """Validation outcome for a single orthogonal view."""
    ok: bool = Field(..., description="Whether the photo passes all quality criteria for this view")
    issues: list[ValidationIssue] = Field(
        default_factory=list,
        description="List of detected quality, posture, or framing issues"
    )
    fix_instruction: str = Field(
        default="",
        description="One short sentence instructing the shopper how to correct the issue"
    )


class PhotoSafetyAssessment(BaseModel):
    """Safety and policy compliance assessment."""
    is_child: bool = Field(default=False, description="Whether any subject appears to be a child or minor")
    is_inappropriate: bool = Field(
        default=False,
        description="Whether subject is unclothed, semi-nude, or in underwear"
    )
    safety_notes: str = Field(default="", description="Neutral observation regarding safety/compliance")


class GeminiPhotoCheckResponse(BaseModel):
    """Structured response from the single multimodal Gemini Flash validation call."""
    views: dict[str, ViewValidationResult] = Field(
        ...,
        description="Per-view validation assessment keyed by view name ('front', 'back', 'left', 'right')"
    )
    safety: PhotoSafetyAssessment = Field(
        default_factory=PhotoSafetyAssessment,
        description="Safety, age, and apparel policy assessment"
    )
    overall_summary: str = Field(
        default="",
        description="Neutral, objective summary of overall photo quality"
    )


class ValidationOutcome(BaseModel):
    """Overall outcome of the 4-photo validation step."""
    status: JobStatus = Field(..., description="Resulting job status (measuring, needs_retake, rejected, failed)")
    can_proceed: bool = Field(..., description="True if photos passed and pipeline can proceed to measurement")
    error_code: str | None = Field(default=None, description="Standardized error code if not proceeding")
    error_message: str | None = Field(default=None, description="User-facing neutral message")
    views: dict[str, ViewValidationResult] = Field(default_factory=dict, description="Detailed per-view results")


def decode_image_bytes(data_or_path: str | bytes | Path) -> bytes:
    """Decode raw image bytes from path, base64 data URL, or bytes payload."""
    if isinstance(data_or_path, bytes):
        return data_or_path
    if isinstance(data_or_path, Path):
        return data_or_path.read_bytes()
    if isinstance(data_or_path, str):
        if data_or_path.startswith("data:image/") and ";base64," in data_or_path:
            _, b64_str = data_or_path.split(";base64,", 1)
            return base64.b64decode(b64_str)
        # Check if it is a local file path
        p = Path(data_or_path)
        if p.exists() and p.is_file():
            return p.read_bytes()
        # Assume base64 string
        try:
            return base64.b64decode(data_or_path)
        except Exception:
            return data_or_path.encode("utf-8")
    raise ValueError(f"Unsupported image input type: {type(data_or_path)}")


def check_local_image_quality(img_bytes: bytes) -> tuple[bool, list[ValidationIssue], str]:
    """Run fast deterministic heuristic checks on image bytes (blur, illumination).

    Returns:
        (is_ok, detected_issues, fix_instruction)
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return False, ["poor_lighting"], "Please upload a valid image file."

    issues: list[ValidationIssue] = []
    instructions: list[str] = []

    # 1. Blur check via Laplacian variance, measured on a central subject crop.
    # Full-frame variance is biased by how much flat background surrounds the
    # subject (common in studio/product-style photos): a sharp photo with a
    # large plain backdrop can score as "blurry" purely because most of the
    # frame has no edges to measure, not because the subject is out of focus.
    # Cropping to the region where the subject is expected keeps the measure
    # about actual focus, not framing/backdrop ratio.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    x0, x1 = int(w * 0.25), int(w * 0.75)
    y0, y1 = int(h * 0.05), int(h * 0.95)
    subject_crop = gray[y0:y1, x0:x1]
    laplacian_var = float(cv2.Laplacian(subject_crop, cv2.CV_64F).var())
    # Threshold lowered from 80.0 to 45.0: real full-body studio photos with large
    # areas of plain/solid clothing (leggings, plain tee) inherently have lower
    # edge-density than a face closeup even when genuinely in focus, so 80.0 was
    # rejecting visibly sharp real photos. 45.0 was verified against real consented
    # test photos (including one that visually confirmed sharp hair/skin detail)
    # while still catching an artificially Gaussian-blurred control image.
    if laplacian_var < 45.0:
        issues.append("poor_lighting")
        instructions.append("Photo appears blurry; please steady the camera.")

    # 2. Illumination check via mean brightness
    mean_val = float(np.mean(gray))
    if mean_val < 35.0:
        if "poor_lighting" not in issues:
            issues.append("poor_lighting")
        instructions.append("Photo is underexposed; please take in better lighting.")
    elif mean_val > 245.0:
        if "poor_lighting" not in issues:
            issues.append("poor_lighting")
        instructions.append("Photo is overexposed; please avoid direct glare.")

    is_ok = len(issues) == 0
    return is_ok, issues, " ".join(instructions)


# Minimum background gap (cm) required beyond the expected torso window edge
# to confirm the arm is genuinely separated from the torso at the chest row,
# rather than merged into the same contiguous silhouette region. Chosen as a
# physically meaningful background gap (clearly wider than the few-pixel
# segmentation noise observed on real photos, which show zero internal gaps
# at this row even when arms ARE separated elsewhere) -- a judgment call,
# like the margin factors in measurements.py, not derived from tape
# measurements or any ground-truth dataset.
_CHEST_GAP_MIN_CM = 2.0

# Isolated foreground pixels within the gap-check buffer, up to this count,
# are treated as segmentation noise rather than genuine arm contamination.
# Real photos tested showed zero such noise (a single clean contiguous run),
# so this is a defensive tolerance, not a value tuned against observed noise.
_CHEST_GAP_NOISE_TOLERANCE_PX = 2


def check_chest_arm_separation(
    front_mask: np.ndarray,
    y_shoulders_px: float,
    y_hips_px: float,
    l_shoulder_x_px: float,
    r_shoulder_x_px: float,
    l_hip_x_px: float,
    r_hip_x_px: float,
    cm_per_pixel: float,
) -> tuple[bool, str]:
    """Deterministically check whether the arms are genuinely separated from
    the torso at the chest row, using only the front-view segmentation mask
    and already-extracted landmark positions (shoulders, hips) -- no LLM call.

    Background: a synthetic SMPL ground-truth round trip and a real photo
    both confirmed that when the arm is not held clearly away from the torso,
    the chest-row width scan (services/geometry/measurements.py) cannot
    distinguish arm from torso -- there is no background gap in the mask to
    find, for any landmark-derived column window. No deterministic width
    correction can fix that after the fact, so this check runs BEFORE
    measurement and asks the shopper to retake the photo with clearer arm
    separation instead of silently reporting a contaminated chest value.

    This reuses the exact same shoulder->hip taper window
    (measurements.py._interpolated_taper_window) that
    calculate_photo_measurements uses for the chest width scan, so a "pass"
    here is consistent with what the measurement will actually see.

    Args:
        front_mask: 2D binary uint8 front-view segmentation mask.
        y_shoulders_px, y_hips_px: Front-view shoulder/hip row (pixel y).
        l_shoulder_x_px, r_shoulder_x_px: Front-view shoulder landmark columns.
        l_hip_x_px, r_hip_x_px: Front-view hip landmark columns.
        cm_per_pixel: Front-view metric scale.

    Returns:
        (is_ok, fix_instruction). is_ok is False if a genuine background gap
        could not be confirmed on both sides of the expected torso window, or
        if the torso boundary could not be identified at all (e.g. the window
        region itself has no foreground pixels).
    """
    h, w = front_mask.shape[:2]
    torso_span = y_hips_px - y_shoulders_px
    if torso_span <= 0 or cm_per_pixel <= 0:
        return False, "Could not determine body proportions from this front photo; please retake it."

    y_chest = int(round(y_shoulders_px + 0.22 * torso_span))
    y_chest = max(0, min(h - 1, y_chest))

    x_lo, x_hi = _interpolated_taper_window(
        y_chest,
        l_shoulder_x_px, r_shoulder_x_px, y_shoulders_px,
        l_hip_x_px, r_hip_x_px, y_hips_px,
        _TAPER_MARGIN_FACTOR,
    )
    x_lo_i, x_hi_i = max(0, int(round(x_lo))), min(w - 1, int(round(x_hi)))

    row = front_mask[y_chest, :]
    if x_hi_i <= x_lo_i or np.count_nonzero(row[x_lo_i:x_hi_i + 1]) == 0:
        return False, (
            "Could not reliably identify the torso boundary in the front photo at chest height; "
            "please retake with the torso and arms clearly visible."
        )

    buffer_px = max(1, int(round(_CHEST_GAP_MIN_CM / cm_per_pixel)))

    def _buffer_is_clear(lo: int, hi: int) -> bool:
        lo = max(0, lo)
        hi = min(w, hi)
        if hi <= lo:
            return True  # buffer falls off the frame edge; nothing to contaminate it
        return int(np.count_nonzero(row[lo:hi])) <= _CHEST_GAP_NOISE_TOLERANCE_PX

    left_clear = _buffer_is_clear(x_lo_i - buffer_px, x_lo_i)
    right_clear = _buffer_is_clear(x_hi_i + 1, x_hi_i + 1 + buffer_px)

    if left_clear and right_clear:
        return True, ""

    return False, (
        "Arms appear too close to the torso in the front photo to measure the chest accurately; "
        "please stand with your arms held slightly further out from your sides and retake the front photo."
    )


def cleanup_local_photos(
    photo_paths: dict[str, Path | str],
    opt_in_keep_front: bool = False,
) -> None:
    """Explicitly delete temporary photo files from disk according to privacy retention rules.

    Args:
        photo_paths: Mapping of view name ('front', 'back', etc.) to file Path or string.
        opt_in_keep_front: If True, preserves 'front' photo and deletes all others.
                          If False, deletes all photo files.
    """
    for view, raw_p in photo_paths.items():
        if opt_in_keep_front and view.lower() == "front":
            logger.info("Retaining front photo per explicit user opt-in: %s", raw_p)
            continue
        try:
            p = Path(raw_p)
            if p.exists() and p.is_file():
                p.unlink()
                logger.debug("Deleted temporary photo (%s): %s", view, p)
        except Exception as exc:
            logger.warning("Could not unlink temporary photo %s: %s", raw_p, exc)


@contextlib.contextmanager
def temporary_photo_workspace(
    photos: dict[str, bytes | str | Path],
    opt_in_keep_front: bool = False,
) -> Generator[dict[str, Path], None, None]:
    """Context manager saving incoming photo payloads to a temporary folder and ensuring cleanup.

    Guarantees that files are deleted upon exit (both on success and on error/cancellation).
    """
    temp_dir = tempfile.mkdtemp(prefix="measured_fit_photos_")
    local_paths: dict[str, Path] = {}
    try:
        for view, payload in photos.items():
            b = decode_image_bytes(payload)
            target_path = Path(temp_dir) / f"{view}.jpg"
            target_path.write_bytes(b)
            local_paths[view] = target_path

        yield local_paths

    finally:
        cleanup_local_photos(local_paths, opt_in_keep_front=opt_in_keep_front)
        # Attempt to remove temporary directory if empty
        try:
            if not opt_in_keep_front or len(list(Path(temp_dir).glob("*"))) == 0:
                Path(temp_dir).rmdir()
        except Exception:
            pass


def validate_photos(
    front: bytes | str | Path,
    back: bytes | str | Path,
    left: bytes | str | Path,
    right: bytes | str | Path,
    age_18_confirmed: bool,
    gemini_client: Any | None = None,
) -> ValidationOutcome:
    """Validate 4 orthogonal body photos against quality, framing, and safety rules.

    Args:
        front: Front view photo payload or path
        back: Back view photo payload or path
        left: Left view photo payload or path
        right: Right view photo payload or path
        age_18_confirmed: User confirmation that subject is 18 years or older
        gemini_client: Optional pre-configured GenAI client for testing

    Returns:
        ValidationOutcome with status (measuring, needs_retake, or rejected).
    """
    # 1. Age Consent Gate: strictly required before processing
    if not age_18_confirmed:
        logger.warning("Photo validation rejected: missing adult confirmation (age_18_confirmed=False).")
        return ValidationOutcome(
            status=JobStatus.rejected,
            can_proceed=False,
            error_code="AGE_CONSENT_REQUIRED",
            error_message="Explicit confirmation that the subject is 18 years or older is required.",
            views={},
        )

    # 2. Decode bytes for all 4 views
    views_raw: dict[str, bytes] = {
        "front": decode_image_bytes(front),
        "back": decode_image_bytes(back),
        "left": decode_image_bytes(left),
        "right": decode_image_bytes(right),
    }

    # 3. Local Heuristic Pre-Check
    local_issues_found = False
    preliminary_views: dict[str, ViewValidationResult] = {}
    for view_name, raw_bytes in views_raw.items():
        is_ok, issues, fix_inst = check_local_image_quality(raw_bytes)
        if not is_ok:
            local_issues_found = True
            preliminary_views[view_name] = ViewValidationResult(
                ok=False,
                issues=issues,
                fix_instruction=fix_inst,
            )

    if local_issues_found:
        instructions = [f"{v.capitalize()}: {res.fix_instruction}" for v, res in preliminary_views.items() if res.fix_instruction]
        summary_msg = "Please retake photos. " + " ".join(instructions)
        logger.info("Local photo quality check failed: %s", summary_msg)
        return ValidationOutcome(
            status=JobStatus.needs_retake,
            can_proceed=False,
            error_code="PHOTO_QUALITY_CHECK_FAILED",
            error_message=summary_msg.strip(),
            views=preliminary_views,
        )

    # 4. Construct Multimodal Prompt for Single Gemini Flash Call
    from google.genai import types

    content_parts: list[Any] = [
        "You are an expert photogrammetry quality inspection system for 3D body measurement reconstruction.\n"
        "Inspect the attached 4 full-body orthogonal photos (front, back, left side, right side).\n"
        "Quality standards:\n"
        "1. Full body must be visible from head to feet in every photo.\n"
        "2. Subject must stand in an A-pose with arms slightly abducted away from the torso.\n"
        "3. Clothing must be form-fitting (e.g. activewear/leggings); loose, baggy, or oversized clothing is not acceptable.\n"
        "4. Lighting must be clear and even, without harsh silhouettes or darkness.\n"
        "5. Exactly one subject must be present; no additional people.\n"
        "6. Orthogonal orientation must match the specified view (front, back, left profile, right profile).\n"
        "Safety and compliance rules:\n"
        "- If any photo depicts a child or minor under 18, set safety.is_child=true.\n"
        "- If subject is unclothed, semi-nude, or wearing only underwear, set safety.is_inappropriate=true.\n"
        "Allowed issues per view: not_full_body, arms_against_body, loose_clothing, poor_lighting, "
        "multiple_people, wrong_view, head_or_feet_cropped.\n"
        "Provide one neutral, concise, shopper-readable sentence for fix_instruction when an issue is detected.\n"
    ]

    for view_name in ("front", "back", "left", "right"):
        content_parts.append(f"View: {view_name}")
        content_parts.append(
            types.Part.from_bytes(data=views_raw[view_name], mime_type="image/jpeg")
        )

    system_instruction = (
        "Perform strict photographic quality and safety assessment for 3D human body reconstruction. "
        "Adhere strictly to the requested JSON schema. Never judge the body; describe only the photo quality and clothing fit."
    )

    try:
        report: GeminiPhotoCheckResponse = generate_json(
            role="flash",
            parts=content_parts,
            schema=GeminiPhotoCheckResponse,
            system=system_instruction,
            client=gemini_client,
        )
    except Exception as exc:
        logger.error("Gemini photo validation call failed: %s", exc)
        return ValidationOutcome(
            status=JobStatus.failed,
            can_proceed=False,
            error_code="INSPECTION_SERVICE_ERROR",
            error_message="Photo inspection service encountered an unexpected error. Please try again.",
            views={},
        )

    # 5. Evaluate Safety & Policy Compliance
    if report.safety.is_child or report.safety.is_inappropriate:
        logger.warning(
            "Photo validation rejected for policy compliance (child=%s, inappropriate=%s).",
            report.safety.is_child,
            report.safety.is_inappropriate,
        )
        return ValidationOutcome(
            status=JobStatus.rejected,
            can_proceed=False,
            error_code="POLICY_VIOLATION",
            error_message="Photos could not be processed due to safety and apparel policy guidelines.",
            views=report.views,
        )

    # 6. Evaluate Quality & Framing per View
    views_combined: dict[str, ViewValidationResult] = {}
    failing_views: list[str] = []
    instructions: list[str] = []

    for view_name in ("front", "back", "left", "right"):
        view_res = report.views.get(view_name)
        if view_res is None:
            view_res = ViewValidationResult(
                ok=False,
                issues=["wrong_view"],
                fix_instruction=f"Missing or unrecognizable {view_name} view.",
            )

        # Merge local issues if any
        if view_name in preliminary_views:
            local_res = preliminary_views[view_name]
            merged_issues = list(set(view_res.issues + local_res.issues))
            view_res = ViewValidationResult(
                ok=False,
                issues=merged_issues,  # type: ignore
                fix_instruction=view_res.fix_instruction or local_res.fix_instruction,
            )

        views_combined[view_name] = view_res

        if not view_res.ok or len(view_res.issues) > 0:
            failing_views.append(view_name)
            if view_res.fix_instruction:
                instructions.append(f"{view_name.capitalize()}: {view_res.fix_instruction}")

    if failing_views:
        summary_msg = "Please retake photos. " + " ".join(instructions)
        logger.info("Photo validation needs retake for views %s: %s", failing_views, summary_msg)
        return ValidationOutcome(
            status=JobStatus.needs_retake,
            can_proceed=False,
            error_code="PHOTO_QUALITY_CHECK_FAILED",
            error_message=summary_msg.strip(),
            views=views_combined,
        )

    # 7. Success: All checks passed
    logger.info("All 4 photos successfully validated. Transitioning to measuring.")
    return ValidationOutcome(
        status=JobStatus.measuring,
        can_proceed=True,
        error_code=None,
        error_message=None,
        views=views_combined,
    )
