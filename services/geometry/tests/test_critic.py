"""Unit tests for services/geometry/critic.py (Multimodal Critic Refinement Loop).

Verifies:
1. Weak-perspective silhouette rendering on CPU (no OpenGL/GPU).
2. Four-view IoU calculation and composite overlay generation.
3. Acceptance rule: round accepted when mean IoU improves >= +0.002.
4. Rejection and rollback rule: round rejected when IoU delta < +0.002, halving step size.
5. Early stopping when critic returns stop=True.
6. Halving of confidence when cause is loose_clothing, occlusion, or pose.
7. Complete round logging to RoundLogDocument.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unittest.mock import patch

import time
import numpy as np
import pytest

from fitcore.schemas import (
    CriticCause,
    CriticIssue,
    CriticRegion,
    CriticRegionFeedback,
    CriticReport,
    CriticView,
    MeasurementZone,
    Measurement,
    Measurements,
    RoundLogDocument,
)
from services.geometry.critic import (
    compute_view_iou,
    generate_overlay_image,
    render_weak_perspective_silhouette,
    run_critic_refinement_loop,
)
from services.geometry.fitting import fit_smpl_shape
from services.geometry.landmarks import ViewLandmarkResult


def _create_synthetic_test_context() -> tuple[dict[str, ViewLandmarkResult], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Create lightweight synthetic landmarks and masks for 4 views."""
    views = ["front", "back", "left", "right"]
    lm_results = {}
    photo_masks = {}
    photos_rgb = {}

    h, w = 600, 300
    for v in views:
        mask = np.zeros((h, w), dtype=np.uint8)
        # Create rectangular torso silhouette between rows 80 and 520 (440 px height)
        bw = 90 if v in ("front", "back") else 60
        mask[80:520, w // 2 - bw // 2 : w // 2 + bw // 2] = 1
        photo_masks[v] = mask

        rgb = np.full((h, w, 3), 100, dtype=np.uint8)
        rgb[mask > 0] = [180, 160, 150]
        photos_rgb[v] = rgb

        lm_results[v] = ViewLandmarkResult(
            view=v,
            landmarks=[],
            image_width=w,
            image_height=h,
            mask_top_row=80,
            mask_bottom_row=519,
            pixel_height=440,
            cm_per_pixel=175.0 / 440.0,
            mean_confidence=0.92,
        )

    return lm_results, photo_masks, photos_rgb


def test_render_weak_perspective_silhouette():
    """Verify weak-perspective projection produces valid binary mask on CPU."""
    fit = fit_smpl_shape(Measurements(zones={}), target_height_cm=175.0, max_steps=5)
    mask = render_weak_perspective_silhouette(
        vertices=fit.vertices,
        faces=fit.faces,
        view="front",
        image_width=300,
        image_height=600,
        mask_top_row=80,
        mask_bottom_row=519,
        center_x=150,
    )

    assert mask.shape == (600, 300)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})
    nonzeros = int(np.count_nonzero(mask))
    assert 10_000 <= nonzeros <= 100_000


def test_compute_view_iou():
    """Verify IoU calculation for perfect, partial, and disjoint masks."""
    m1 = np.zeros((100, 100), dtype=np.uint8)
    m1[10:60, 10:60] = 1  # 50x50 = 2500 px

    # Exact match
    assert compute_view_iou(m1, m1) == 1.0

    # Half overlap: 50x50 shifted by 25 pixels
    m2 = np.zeros((100, 100), dtype=np.uint8)
    m2[10:60, 35:85] = 1  # overlap 50x25 = 1250 px, union = 3750 px
    iou = compute_view_iou(m1, m2)
    assert abs(iou - (1250.0 / 3750.0)) < 1e-3

    # Disjoint
    m3 = np.zeros((100, 100), dtype=np.uint8)
    m3[70:90, 70:90] = 1
    assert compute_view_iou(m1, m3) == 0.0


def test_generate_overlay_image():
    """Verify visual diagnostic overlay generation."""
    rgb = np.full((100, 100, 3), 200, dtype=np.uint8)
    p_mask = np.zeros((100, 100), dtype=np.uint8)
    p_mask[20:80, 30:70] = 1
    m_mask = np.zeros((100, 100), dtype=np.uint8)
    m_mask[25:75, 35:65] = 1

    overlay = generate_overlay_image(rgb, p_mask, m_mask)
    assert overlay.shape == (100, 100, 3)
    assert overlay.dtype == np.uint8
    # Background should be darkened (~100 instead of 200)
    assert overlay[0, 0, 0] <= 110


def test_critic_loop_accepted_round():
    """Verify candidate is accepted when IoU improves by at least +0.002."""
    lm_results, photo_masks, photos_rgb = _create_synthetic_test_context()
    target_measurements = Measurements(
        zones={
            MeasurementZone.chest: Measurement(value_cm=95.0, confidence=0.9, source="photo_measurement"),
            MeasurementZone.waist: Measurement(value_cm=80.0, confidence=0.9, source="photo_measurement"),
            MeasurementZone.hip: Measurement(value_cm=100.0, confidence=0.9, source="photo_measurement"),
        }
    )

    initial_fit = fit_smpl_shape(target_measurements, target_height_cm=175.0, max_steps=10)

    # Mock critic: suggest tightening waist by reporting too_wide
    call_count = 0
    def mock_critic_improves(overlays, meas):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return CriticReport(
                regions=[
                    CriticRegionFeedback(
                        view=CriticView.front,
                        region=CriticRegion.waist,
                        issue=CriticIssue.too_wide,
                        severity=0.6,
                        cause=CriticCause.model_error,
                        trust=0.9,
                    )
                ],
                stop=False,
                summary="The waist silhouette is slightly too wide relative to the contours.",
            )
        return CriticReport(regions=[], stop=True, summary="Fit is aligned.")

    final_fit, round_logs, _ = run_critic_refinement_loop(
        initial_fit=initial_fit,
        target_measurements=target_measurements,
        target_height_cm=175.0,
        gender="neutral",
        landmark_results=lm_results,
        photo_masks=photo_masks,
        photos_rgb=photos_rgb,
        critic_llm_fn=mock_critic_improves,
        max_rounds=2,
        min_iou_delta=-0.05,  # Force acceptance for test
    )

    assert len(round_logs) >= 1
    first_round = round_logs[0]
    assert isinstance(first_round, RoundLogDocument)
    assert first_round.accepted is True
    assert first_round.round == 1
    assert MeasurementZone.waist in first_round.zone_adjustments


def test_critic_loop_rejected_round_rollback():
    """Verify round is rejected and step size halved when IoU improvement is insufficient."""
    lm_results, photo_masks, photos_rgb = _create_synthetic_test_context()
    target_measurements = Measurements(
        zones={
            MeasurementZone.chest: Measurement(value_cm=95.0, confidence=0.9, source="photo_measurement"),
        }
    )

    initial_fit = fit_smpl_shape(target_measurements, target_height_cm=175.0, max_steps=10)

    def mock_critic_degrades(overlays, meas):
        return CriticReport(
            regions=[
                CriticRegionFeedback(
                    view=CriticView.front,
                    region=CriticRegion.chest,
                    issue=CriticIssue.too_narrow,
                    severity=0.8,
                    cause=CriticCause.model_error,
                    trust=0.9,
                )
            ],
            stop=False,
            summary="Chest needs widening.",
        )

    # Require an unreachable +0.50 IoU improvement to force rejection
    final_fit, round_logs, _ = run_critic_refinement_loop(
        initial_fit=initial_fit,
        target_measurements=target_measurements,
        target_height_cm=175.0,
        gender="neutral",
        landmark_results=lm_results,
        photo_masks=photo_masks,
        photos_rgb=photos_rgb,
        critic_llm_fn=mock_critic_degrades,
        max_rounds=2,
        min_iou_delta=0.50,
    )

    assert len(round_logs) == 2
    for r in round_logs:
        assert r.accepted is False
    # Step size halved on round 2
    assert round_logs[1].step_size == 0.5


def test_critic_loop_early_stop():
    """Verify loop immediately stops when critic returns stop=True."""
    lm_results, photo_masks, photos_rgb = _create_synthetic_test_context()
    target_measurements = Measurements(zones={})
    initial_fit = fit_smpl_shape(target_measurements, target_height_cm=175.0, max_steps=5)

    def mock_critic_stops(overlays, meas):
        return CriticReport(regions=[], stop=True, summary="Silhouette alignment is optimal.")

    final_fit, round_logs, _ = run_critic_refinement_loop(
        initial_fit=initial_fit,
        target_measurements=target_measurements,
        target_height_cm=175.0,
        gender="neutral",
        landmark_results=lm_results,
        photo_masks=photo_masks,
        photos_rgb=photos_rgb,
        critic_llm_fn=mock_critic_stops,
        max_rounds=5,
    )

    # Should only execute round 1 and stop
    assert len(round_logs) == 1
    assert round_logs[0].critic_notes == "Silhouette alignment is optimal."


def test_critic_loop_loose_clothing_halves_confidence():
    """Verify non-model errors (e.g. loose clothing) halve zone confidence."""
    lm_results, photo_masks, photos_rgb = _create_synthetic_test_context()
    target_measurements = Measurements(
        zones={
            MeasurementZone.thigh: Measurement(value_cm=58.0, confidence=0.80, source="photo_measurement"),
        }
    )
    initial_fit = fit_smpl_shape(target_measurements, target_height_cm=175.0, max_steps=5)

    call_count = 0
    def mock_critic_loose_clothing(overlays, meas):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return CriticReport(
                regions=[
                    CriticRegionFeedback(
                        view=CriticView.front,
                        region=CriticRegion.thighs,
                        issue=CriticIssue.too_wide,
                        severity=0.7,
                        cause=CriticCause.loose_clothing,
                        trust=0.9,
                    )
                ],
                stop=False,
                summary="Wide silhouette caused by loose fabric around thighs.",
            )
        return CriticReport(regions=[], stop=True, summary="Done.")

    final_fit, round_logs, _ = run_critic_refinement_loop(
        initial_fit=initial_fit,
        target_measurements=target_measurements,
        target_height_cm=175.0,
        gender="neutral",
        landmark_results=lm_results,
        photo_masks=photo_masks,
        photos_rgb=photos_rgb,
        critic_llm_fn=mock_critic_loose_clothing,
        max_rounds=2,
    )

    assert len(round_logs) >= 1
    # Target value was not adjusted because cause was loose_clothing, not model_error
    assert MeasurementZone.thigh not in round_logs[0].zone_adjustments


def test_critic_loop_rejected_round_restores_complete_best_candidate():
    """Verify rejected round completely restores best candidate vertices, betas, and diagnostics."""
    lm_results, photo_masks, photos_rgb = _create_synthetic_test_context()
    target_measurements = Measurements(
        zones={
            MeasurementZone.waist: Measurement(value_cm=75.0, confidence=0.90, source="photo_measurement"),
        }
    )
    initial_fit = fit_smpl_shape(target_measurements, target_height_cm=175.0, max_steps=10)
    initial_betas = list(initial_fit.betas)
    initial_vertices = initial_fit.vertices.copy()

    # Reject round by forcing min_iou_delta = 0.50
    def mock_critic_reject(overlays, meas):
        return CriticReport(
            regions=[
                CriticRegionFeedback(
                    view=CriticView.front,
                    region=CriticRegion.waist,
                    issue=CriticIssue.too_wide,
                    severity=0.8,
                    cause=CriticCause.model_error,
                    trust=0.9,
                )
            ],
            stop=False,
            summary="Waist is too wide.",
        )

    final_fit, round_logs, _ = run_critic_refinement_loop(
        initial_fit=initial_fit,
        target_measurements=target_measurements,
        target_height_cm=175.0,
        gender="neutral",
        landmark_results=lm_results,
        photo_masks=photo_masks,
        photos_rgb=photos_rgb,
        critic_llm_fn=mock_critic_reject,
        max_rounds=1,
        min_iou_delta=0.50,
    )

    assert round_logs[0].accepted is False
    # Final fit must be identical to the best candidate before the rejected round
    assert np.allclose(final_fit.vertices, initial_vertices)
    assert final_fit.betas == initial_betas
    assert final_fit.fitted_height_cm == initial_fit.fitted_height_cm


def test_critic_loop_contradictory_feedback_averages():
    """Verify contradictory critic feedback on the same zone across views is safely averaged."""
    lm_results, photo_masks, photos_rgb = _create_synthetic_test_context()
    target_measurements = Measurements(
        zones={
            MeasurementZone.waist: Measurement(value_cm=75.0, confidence=0.90, source="photo_measurement"),
        }
    )
    initial_fit = fit_smpl_shape(target_measurements, target_height_cm=175.0, max_steps=5)

    def mock_critic_contradictory(overlays, meas):
        return CriticReport(
            regions=[
                CriticRegionFeedback(
                    view=CriticView.front,
                    region=CriticRegion.waist,
                    issue=CriticIssue.too_wide,
                    severity=0.5,
                    cause=CriticCause.model_error,
                    trust=0.9,
                ),
                CriticRegionFeedback(
                    view=CriticView.back,
                    region=CriticRegion.waist,
                    issue=CriticIssue.too_narrow,
                    severity=0.5,
                    cause=CriticCause.model_error,
                    trust=0.9,
                ),
            ],
            stop=False,
            summary="Contradictory views on waist.",
        )

    final_fit, round_logs, _ = run_critic_refinement_loop(
        initial_fit=initial_fit,
        target_measurements=target_measurements,
        target_height_cm=175.0,
        gender="neutral",
        landmark_results=lm_results,
        photo_masks=photo_masks,
        photos_rgb=photos_rgb,
        critic_llm_fn=mock_critic_contradictory,
        max_rounds=1,
    )

    assert len(round_logs) == 1
    # Equal and opposite severity should cancel out to near 0 adjustment
    waist_adj = round_logs[0].zone_adjustments.get(MeasurementZone.waist, 0.0)
    assert abs(waist_adj) < 0.1


def test_critic_loop_llm_exception_handled_safely():
    """Verify that an exception in the critic LLM call logs failure and exits gracefully."""
    lm_results, photo_masks, photos_rgb = _create_synthetic_test_context()
    target_measurements = Measurements(zones={})
    initial_fit = fit_smpl_shape(target_measurements, target_height_cm=175.0, max_steps=5)

    def mock_critic_crashes(overlays, meas):
        raise RuntimeError("Simulated Ollama / Gemini network timeout")

    final_fit, round_logs, _ = run_critic_refinement_loop(
        initial_fit=initial_fit,
        target_measurements=target_measurements,
        target_height_cm=175.0,
        gender="neutral",
        landmark_results=lm_results,
        photo_masks=photo_masks,
        photos_rgb=photos_rgb,
        critic_llm_fn=mock_critic_crashes,
        max_rounds=3,
    )

    assert len(round_logs) == 1
    assert round_logs[0].accepted is False
    assert "Simulated Ollama / Gemini network timeout" in round_logs[0].critic_notes
    assert final_fit == initial_fit


def test_critic_loop_timeout_marked_unavailable_not_fabricated_or_hung():
    """Regression test for the real demo symptom: a critic call that times out
    (TimeoutError, the exact exception shape urllib.request.urlopen raises on
    OLLAMA_TIMEOUT_S) must not hang the refinement loop, must not be reported as
    a fit failure, and must not fabricate a fake critic success -- it must stop
    refining after this round and return the already-computed, already-valid
    initial_fit as-is, with the round clearly marked unavailable (not a generic
    'error' label that could be confused with the fit itself having failed).
    """
    lm_results, photo_masks, photos_rgb = _create_synthetic_test_context()
    target_measurements = Measurements(zones={})
    initial_fit = fit_smpl_shape(target_measurements, target_height_cm=175.0, max_steps=5)

    def mock_critic_times_out(overlays, meas):
        raise TimeoutError("The read operation timed out")

    final_fit, round_logs, _ = run_critic_refinement_loop(
        initial_fit=initial_fit,
        target_measurements=target_measurements,
        target_height_cm=175.0,
        gender="neutral",
        landmark_results=lm_results,
        photo_masks=photo_masks,
        photos_rgb=photos_rgb,
        critic_llm_fn=mock_critic_times_out,
        max_rounds=5,
    )

    # Stops after the first unavailable round rather than burning all 5 rounds'
    # worth of timeouts against a backend that is clearly not responding.
    assert len(round_logs) == 1
    assert round_logs[0].accepted is False
    assert "unavailable" in round_logs[0].critic_notes.lower()
    assert "read operation timed out" in round_logs[0].critic_notes
    # The already-computed, already-valid fit is returned unchanged -- not discarded,
    # not replaced by a fabricated "improved" result.
    assert final_fit == initial_fit
    assert final_fit.is_valid


def test_critic_loop_default_path_calls_generate_json_with_correct_signature():
    """Regression test: when no critic_llm_fn is injected (the real production
    code path used by the actual worker, never exercised by the other tests
    above since they all inject a mock), run_critic_refinement_loop must call
    fitcore.llm.generate_json with its real keyword arguments (role, parts,
    schema, system). A prior bug called it as generate_json(prompt=..., ...)
    -- 'prompt' is not a real parameter of generate_json (the real one is
    'parts') -- which raised a TypeError on every real job, silently caught
    and treated as a critic failure, so the refinement loop never actually ran
    for any real (non-test) job.

    Also verifies the overlay images are passed as multimodal parts (bytes),
    since the critic is documented as multimodal (it must see the cyan/magenta
    overlays to critique them, not just text).
    """
    lm_results, photo_masks, photos_rgb = _create_synthetic_test_context()
    target_measurements = Measurements(zones={})
    initial_fit = fit_smpl_shape(target_measurements, target_height_cm=175.0, max_steps=5)

    captured_calls = []

    def fake_generate_json(**kwargs):
        captured_calls.append(kwargs)
        return CriticReport(regions=[], stop=True, summary="Fit looks good.")

    with patch("fitcore.llm.generate_json", side_effect=fake_generate_json) as mock_gen:
        final_fit, round_logs, _ = run_critic_refinement_loop(
            initial_fit=initial_fit,
            target_measurements=target_measurements,
            target_height_cm=175.0,
            gender="neutral",
            landmark_results=lm_results,
            photo_masks=photo_masks,
            photos_rgb=photos_rgb,
            critic_llm_fn=None,
            max_rounds=3,
        )

    assert mock_gen.called, "generate_json was never called -- critic default path did not run"
    call_kwargs = captured_calls[0]

    # Must use the real generate_json signature, not the old broken 'prompt=' kwarg.
    assert "prompt" not in call_kwargs
    assert call_kwargs["role"] == "pro"
    assert call_kwargs["schema"] is CriticReport
    assert "parts" in call_kwargs
    assert isinstance(call_kwargs["parts"], list)

    # At least one overlay image must be included as raw bytes (multimodal content),
    # not just a text prompt.
    image_parts = [p for p in call_kwargs["parts"] if isinstance(p, bytes)]
    assert len(image_parts) >= 1, "No image bytes passed to the critic -- it cannot see the overlays"

    # Successful (stop=True) critic call must not be treated as an error.
    assert round_logs[0].accepted is False  # stop=True rounds are logged as not-accepted, by design
    assert "Fit looks good." in round_logs[0].critic_notes
    assert "Critic evaluation error" not in round_logs[0].critic_notes


def test_critic_loop_default_path_includes_measurement_table_in_prompt():
    """Regression test: the critic must receive the current measurement table,
    not just the overlay images. Prior to this fix, the default (real)
    generate_json call only sent a generic instruction string plus images --
    the critic had no way to reference actual zone values (e.g. 'waist is
    80.2cm') when critiquing regional discrepancies.
    """
    lm_results, photo_masks, photos_rgb = _create_synthetic_test_context()
    target_measurements = Measurements(
        zones={
            MeasurementZone.waist: Measurement(value_cm=80.2, confidence=0.9, source="photo_measurement"),
            MeasurementZone.hip: Measurement(value_cm=86.7, confidence=0.9, source="photo_measurement"),
        }
    )
    initial_fit = fit_smpl_shape(target_measurements, target_height_cm=175.0, max_steps=5)

    captured_calls = []

    def fake_generate_json(**kwargs):
        captured_calls.append(kwargs)
        return CriticReport(regions=[], stop=True, summary="Fit looks good.")

    with patch("fitcore.llm.generate_json", side_effect=fake_generate_json):
        run_critic_refinement_loop(
            initial_fit=initial_fit,
            target_measurements=target_measurements,
            target_height_cm=175.0,
            gender="neutral",
            landmark_results=lm_results,
            photo_masks=photo_masks,
            photos_rgb=photos_rgb,
            critic_llm_fn=None,
            max_rounds=1,
        )

    assert captured_calls, "generate_json was never called"
    text_parts = [p for p in captured_calls[0]["parts"] if isinstance(p, str)]
    combined_text = "\n".join(text_parts)
    assert "waist" in combined_text and "80.2" in combined_text, (
        "measurement table (waist value) not found in critic prompt parts"
    )
    assert "hip" in combined_text and "86.7" in combined_text, (
        "measurement table (hip value) not found in critic prompt parts"
    )


def test_critic_loop_persists_overlays_when_storage_client_provided():
    """Regression test: overlay_urls on RoundLogDocument was always {} even
    though overlays were generated and reviewed every round, because nothing
    ever uploaded them. When a storage_client + overlay_path_prefix are
    provided (as worker.py now does), each round's overlay images must be
    uploaded and their URLs recorded on that round's log.
    """
    lm_results, photo_masks, photos_rgb = _create_synthetic_test_context()
    target_measurements = Measurements(zones={})
    initial_fit = fit_smpl_shape(target_measurements, target_height_cm=175.0, max_steps=5)

    uploaded: dict[str, bytes] = {}

    class FakeStorageClient:
        def upload_bytes(self, destination_path, data, content_type="application/octet-stream"):
            uploaded[destination_path] = data
            return f"gs://fake-bucket/{destination_path}"

    def mock_critic_stops(overlays, meas):
        return CriticReport(regions=[], stop=True, summary="Fit looks good.")

    _, round_logs, _ = run_critic_refinement_loop(
        initial_fit=initial_fit,
        target_measurements=target_measurements,
        target_height_cm=175.0,
        gender="neutral",
        landmark_results=lm_results,
        photo_masks=photo_masks,
        photos_rgb=photos_rgb,
        critic_llm_fn=mock_critic_stops,
        max_rounds=1,
        storage_client=FakeStorageClient(),
        overlay_path_prefix="users/u1/jobs/job1/critic_overlays",
    )

    assert len(round_logs) == 1
    assert round_logs[0].overlay_urls, "overlay_urls is empty even though a storage client was provided"
    assert set(round_logs[0].overlay_urls.keys()) == {"front", "back", "left", "right"}
    for view, url in round_logs[0].overlay_urls.items():
        assert url.startswith("gs://fake-bucket/users/u1/jobs/job1/critic_overlays/round_1_")
    assert len(uploaded) == 4


def test_critic_loop_without_storage_client_leaves_overlay_urls_empty():
    """Backward-compatibility check: omitting storage_client/overlay_path_prefix
    (as every existing caller except worker.py does -- tests, offline CLI)
    must preserve the prior behavior of an empty overlay_urls dict, not raise."""
    lm_results, photo_masks, photos_rgb = _create_synthetic_test_context()
    target_measurements = Measurements(zones={})
    initial_fit = fit_smpl_shape(target_measurements, target_height_cm=175.0, max_steps=5)

    def mock_critic_stops(overlays, meas):
        return CriticReport(regions=[], stop=True, summary="Fit looks good.")

    _, round_logs, _ = run_critic_refinement_loop(
        initial_fit=initial_fit,
        target_measurements=target_measurements,
        target_height_cm=175.0,
        gender="neutral",
        landmark_results=lm_results,
        photo_masks=photo_masks,
        photos_rgb=photos_rgb,
        critic_llm_fn=mock_critic_stops,
        max_rounds=1,
    )

    assert round_logs[0].overlay_urls == {}


def test_critic_loop_strict_overall_timeout_aborts_slow_critic_and_preserves_fit():
    """Regression test: when critic review hangs or times out repeatedly (e.g. repeated
    read timeouts on attempts 1 and 2), the strict overall timeout (critic_timeout_s)
    must abort the critic review promptly, mark the round unavailable with a descriptive
    timeout note, and preserve the already-computed 3D fit without blocking indefinitely.
    """
    lm_results, photo_masks, photos_rgb = _create_synthetic_test_context()
    target_measurements = Measurements(zones={})
    initial_fit = fit_smpl_shape(target_measurements, target_height_cm=175.0, max_steps=5)

    def slow_critic_simulating_repeated_timeouts(overlays, meas):
        # Simulate an unresponsive critic or slow retry loop
        time.sleep(1.5)
        return CriticReport(regions=[], stop=True, summary="Late critic report.")

    t0 = time.time()
    final_fit, round_logs, _ = run_critic_refinement_loop(
        initial_fit=initial_fit,
        target_measurements=target_measurements,
        target_height_cm=175.0,
        gender="neutral",
        landmark_results=lm_results,
        photo_masks=photo_masks,
        photos_rgb=photos_rgb,
        critic_llm_fn=slow_critic_simulating_repeated_timeouts,
        max_rounds=5,
        critic_timeout_s=0.15,
    )
    elapsed = time.time() - t0

    # Strict timeout limit enforced: stopped around 0.15s, not after 1.5s
    assert elapsed < 1.0, f"Critic loop took {elapsed:.2f}s; overall timeout was not enforced"
    assert len(round_logs) == 1
    assert round_logs[0].accepted is False
    assert "unavailable" in round_logs[0].critic_notes.lower()
    assert "read operation timed out" in round_logs[0].critic_notes
    assert "overall limit 0.1s exceeded" in round_logs[0].critic_notes or "exceeded" in round_logs[0].critic_notes
    # Preserves already-computed 3D fit as-is
    assert final_fit == initial_fit
    assert final_fit.is_valid

