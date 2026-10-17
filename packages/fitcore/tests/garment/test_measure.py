import statistics

import cv2
import numpy as np
import pytest

from fitcore.garment.geometry import MeasureError, decode
from fitcore.garment.landmarks import LlmLandmarkFinder, required_landmarks
from fitcore.garment.measure import measure_garment
from fitcore.garment.synthetic import render_flatlay
from fitcore.schemas import Category, MeasurementZone as Z
from tests.garment.helpers import CASES, model_reply, oracle

SEEDS = range(4)
MAX_ERR_CM = 2.5          # worst case with strong perspective: a small sheet far from the garment magnifies corner error
MEDIAN_ERR_CM = 0.6


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "t")
    monkeypatch.setenv("GEMINI_PRO_MODEL", "gemini-exp-custom")


@pytest.mark.parametrize("name", list(CASES))
def test_measures_each_category_from_synthetic_photos(name):
    cat, truth = CASES[name]
    errors = []
    for seed in SEEDS:
        photo = render_flatlay(cat, truth, seed=seed, a4_landscape=bool(seed % 2))
        m = measure_garment(photo.jpeg, cat, finder=oracle(photo))
        assert set(m.zones) >= {z for z in truth}, (name, seed, m.flags)
        for z, t in truth.items():
            errors.append(abs(m.zones[z].value_cm - t))
            assert m.zones[z].source == "flat_lay" and 0 < m.zones[z].confidence <= 1
        assert not [f for f in m.flags if "missing" in f or "off_garment" in f]
    assert max(errors) <= MAX_ERR_CM and statistics.median(errors) <= MEDIAN_ERR_CM, (name, max(errors), statistics.median(errors))


def test_scale_does_not_depend_on_how_far_the_camera_is():
    cat, truth = CASES["shirt"]
    near = measure_garment((p1 := render_flatlay(cat, truth, seed=1, scale=1.0)).jpeg, cat, finder=oracle(p1))
    far = measure_garment((p2 := render_flatlay(cat, truth, seed=1, scale=0.8)).jpeg, cat, finder=oracle(p2))
    for z in truth:
        assert abs(near.zones[z].value_cm - far.zones[z].value_cm) <= 2.0


def test_flat_widths_become_circumferences_and_lengths_stay_lengths():
    cat, truth = CASES["shirt"]
    photo = render_flatlay(cat, truth, seed=3)
    m = measure_garment(photo.jpeg, cat, finder=oracle(photo))
    assert 105 < m.zones[Z.chest].value_cm < 123      # a 114 cm circumference, not the ~57 cm flat width
    assert 70 < m.zones[Z.garment_length].value_cm < 78


def test_back_rise_is_reported_as_unsupported_not_invented():
    cat, truth = CASES["trousers"]
    photo = render_flatlay(cat, truth, seed=0)
    m = measure_garment(photo.jpeg, cat, finder=oracle(photo))
    assert Z.back_rise in m.unsupported and Z.back_rise not in m.zones


def test_overlay_is_a_png_with_the_measurements_drawn():
    cat, truth = CASES["shirt"]
    photo = render_flatlay(cat, truth, seed=0)
    m = measure_garment(photo.jpeg, cat, finder=oracle(photo))
    img = cv2.imdecode(np.frombuffer(m.overlay_png, np.uint8), cv2.IMREAD_COLOR)
    assert m.overlay_png[:8] == b"\x89PNG\r\n\x1a\n" and max(img.shape[:2]) <= 1100 and min(img.shape[:2]) > 300
    assert "overlay_png" not in m.model_dump()          # bytes stay out of the JSON


# ----------------------------------------------------------------------------- photos that cannot be measured


def test_no_a4_sheet_is_reported_with_advice():
    cat, truth = CASES["shirt"]
    photo = render_flatlay(cat, truth, seed=0, with_a4=False)
    with pytest.raises(MeasureError) as e:
        measure_garment(photo.jpeg, cat, finder=oracle(photo))
    assert e.value.code == "no_a4" and "A4" in e.value.message


def test_no_garment_in_the_photo():
    cat, truth = CASES["shirt"]
    photo = render_flatlay(cat, truth, seed=0, with_garment=False)
    with pytest.raises(MeasureError) as e:
        measure_garment(photo.jpeg, cat, finder=oracle(photo))
    assert e.value.code == "no_garment"


@pytest.mark.parametrize("data", [b"not an image", b"", cv2.imencode(".png", np.zeros((50, 50, 3), np.uint8))[1].tobytes()])
def test_unusable_files(data):
    with pytest.raises(MeasureError) as e:
        measure_garment(data, Category.shirt, finder=lambda *a: {})
    assert e.value.code == "bad_image"


def test_extreme_tilt_is_refused_or_not_found_rather_than_measured_wrongly():
    cat, truth = CASES["shirt"]
    photo = render_flatlay(cat, truth, seed=0, tilt=0.38)
    try:
        m = measure_garment(photo.jpeg, cat, finder=oracle(photo))
    except MeasureError as e:
        assert e.code in {"perspective_extreme", "no_a4", "no_garment", "no_landmarks"}
    else:
        assert any(f.startswith("a4_perspective") for f in m.flags)


def test_decode_rejects_tiny_images():
    with pytest.raises(MeasureError):
        decode(cv2.imencode(".png", np.zeros((100, 100, 3), np.uint8))[1].tobytes())


# ----------------------------------------------------------------------------- the model path (Gemini places the landmarks)


def test_llm_finder_reads_the_models_reply_and_measures():
    cat, truth = CASES["shirt"]
    photo = render_flatlay(cat, truth, seed=2)
    client = model_reply(photo)
    m = measure_garment(photo.jpeg, cat, finder=LlmLandmarkFinder(client=client))
    for z, t in truth.items():
        assert abs(m.zones[z].value_cm - t) <= MAX_ERR_CM
    contents = client.models.generate_content.call_args.kwargs["contents"]
    assert "shirt" in contents[0] and "armpit_left" in contents[0] and "sleeve_cuff_left" in contents[0]
    assert contents[1].inline_data.mime_type == "image/jpeg" and contents[1].inline_data.data[:2] == b"\xff\xd8"


def test_measure_garment_uses_the_llm_finder_by_default():
    cat, truth = CASES["tshirt"]
    photo = render_flatlay(cat, truth, seed=0)
    m = measure_garment(photo.jpeg, cat, client=model_reply(photo))
    assert Z.chest in m.zones


def test_landmarks_the_model_leaves_out_flag_those_zones_only():
    cat, truth = CASES["shirt"]
    photo = render_flatlay(cat, truth, seed=2)
    m = measure_garment(photo.jpeg, cat, finder=LlmLandmarkFinder(client=model_reply(photo, drop=("sleeve_cuff_left",))))
    assert Z.sleeve_length not in m.zones and "landmark_missing:sleeve_cuff_left" in m.flags and "zone_not_measured:sleeve_length" in m.flags
    assert {Z.chest, Z.waist, Z.shoulder_width, Z.garment_length} <= set(m.zones)


def test_points_the_model_puts_off_the_garment_are_not_used():
    cat, truth = CASES["shirt"]
    photo = render_flatlay(cat, truth, seed=2)
    client = model_reply(photo)
    import json
    reply = json.loads(client.models.generate_content.return_value.text)
    for lm in reply["landmarks"]:
        if lm["name"] == "hem_left":
            lm["x"], lm["y"] = 960.0, 960.0                     # far corner of the image: surface, not garment
    client.models.generate_content.return_value.text = json.dumps(reply)
    m = measure_garment(photo.jpeg, cat, finder=LlmLandmarkFinder(client=client))
    assert "landmark_off_garment:hem_left" in m.flags and Z.garment_length not in m.zones


def test_all_landmarks_missing_is_an_error():
    cat, truth = CASES["shirt"]
    photo = render_flatlay(cat, truth, seed=2)
    with pytest.raises(MeasureError) as e:
        measure_garment(photo.jpeg, cat, finder=LlmLandmarkFinder(client=model_reply(photo, drop=tuple(photo.landmarks))))
    assert e.value.code == "no_landmarks"


def test_snapping_to_the_outline_absorbs_small_placement_errors():
    cat, truth = CASES["shirt"]
    errors = []
    for seed in SEEDS:
        photo = render_flatlay(cat, truth, seed=seed)
        m = measure_garment(photo.jpeg, cat, finder=LlmLandmarkFinder(client=model_reply(photo, jitter_px=6.0, seed=seed)))
        errors += [abs(m.zones[z].value_cm - t) for z, t in truth.items() if z in m.zones]
    assert statistics.median(errors) <= 1.2 and max(errors) <= 4.5


def test_required_landmarks_follow_the_category():
    assert set(required_landmarks(Category.tshirt, [Z.chest, Z.shoulder_width, Z.garment_length])) == {
        "armpit_left", "armpit_right", "shoulder_tip_left", "shoulder_tip_right", "neck_left", "neck_right", "hem_left", "hem_right"}
    assert "crotch" in required_landmarks(Category.jeans, [Z.thigh, Z.inseam]) and "sleeve_cuff_left" not in required_landmarks(Category.jeans, [Z.inseam])


# ----------------------------------------------------------------------------- recorded from the live model


@pytest.mark.parametrize("name,seed", [("shirt", 2), ("trousers", 1)])
def test_recorded_gemini_landmarks_give_the_same_measurements_offline(name, seed):
    """tests/fixtures/llm/garment_landmarks_*.json hold what Gemini (pro) really answered for these synthetic photos."""
    import json
    from pathlib import Path
    from unittest.mock import MagicMock

    fx = json.loads((Path(__file__).parent.parent / "fixtures" / "llm" / f"garment_landmarks_{name}_seed{seed}.json").read_text(encoding="utf-8"))
    cat, truth = CASES[name]
    photo = render_flatlay(cat, truth, seed=seed)
    client = MagicMock()
    client.models.generate_content.return_value.text = json.dumps(fx["reply"])
    m = measure_garment(photo.jpeg, cat, finder=LlmLandmarkFinder(client=client))
    for z, t in truth.items():
        assert abs(m.zones[z].value_cm - fx["measured"][z.value]) <= 0.05      # the recorded run is reproduced exactly
        assert abs(m.zones[z].value_cm - t) <= 2.0                              # and the live model was close to the truth
