import pytest
from fitcore.units import (
    cm_to_inches,
    inches_to_cm,
    round_measurement,
    format_cm,
    format_inches,
)


def test_cm_to_inches_and_back():
    assert cm_to_inches(2.54) == 1.0
    assert inches_to_cm(1.0) == 2.54
    assert cm_to_inches(100.0, precision=1) == 39.4
    assert inches_to_cm(39.4, precision=1) == 100.1



def test_round_measurement():
    assert round_measurement(102.3, step=0.5) == 102.5
    assert round_measurement(102.1, step=0.5) == 102.0
    assert round_measurement(102.7, step=0.5) == 102.5
    assert round_measurement(102.8, step=0.5) == 103.0


def test_format_strings():
    assert format_cm(102.5) == "102.5 cm"
    assert format_inches(40.354, precision=1) == "40.4 in"
