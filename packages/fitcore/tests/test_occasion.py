"""Occasion test: dress code, circle, seated-time strain, and wording."""

import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "seed"))
import persona  # noqa: E402

from fitcore.occasion import occasion_test, resolve_start  # noqa: E402
from fitcore.schemas import Measurements  # noqa: E402

TWIN = Measurements.model_validate({"zones": persona.twin_zones()})
BANNED = re.compile(r"\b(weight|fat|flatter\w*|flaw\w*|overweight|heavy|skinny)\b", re.I)


def test_resolve_start_tomorrow_and_saturday():
    wed = date(2026, 10, 14)
    assert resolve_start(1, "10:00", wed) == "2026-10-15T10:00:00+05:30"
    assert resolve_start("saturday", "18:30", wed) == "2026-10-17T18:30:00+05:30"
    sat = date(2026, 10, 17)
    assert resolve_start("saturday", "18:30", sat) == "2026-10-24T18:30:00+05:30"      # at least two days out
    fri = date(2026, 10, 16)
    assert resolve_start("saturday", "18:30", fri) == "2026-10-24T18:30:00+05:30"      # tomorrow is Saturday: use the next one


def test_every_event_gets_a_full_outfit_that_fits_the_dress_code():
    closet = persona.closet()
    assert len(closet) == 20
    for e in persona.events():
        r = occasion_test(e, closet, TWIN, persona.seated_delta())
        assert {"top", "bottom", "shoes"} <= {p.slot for p in r.outfit} and not r.missing_slots
        assert not BANNED.search(r.model_dump_json())
    wedding = occasion_test(next(e for e in persona.events() if e.title == "Wedding reception"), closet, TWIN, persona.seated_delta())
    top = next(p for p in wedding.outfit if p.slot == "top")
    assert "kurta" in top.name.lower()                                                  # ethnic, formal


def test_seated_time_flags_a_close_cut_piece():
    closet = persona.closet()
    wedding = next(e for e in persona.events() if e.title == "Wedding reception")
    r = occasion_test(wedding, closet, TWIN, persona.seated_delta(), alternatives=5)
    churidar = next(p for ps in r.alternatives.values() for p in ps if "churidar" in p.name.lower())
    assert churidar.fit_state == "measured_watch" and any("Seated" in n for n in churidar.fit_notes)
    standing = occasion_test(wedding.model_copy(update={"seated_minutes": 0}), closet, TWIN, persona.seated_delta(), alternatives=5)
    again = next(p for ps in [*standing.alternatives.values(), standing.outfit] for p in ps if "churidar" in p.name.lower())
    assert not any("Seated" in n for n in again.fit_notes)
