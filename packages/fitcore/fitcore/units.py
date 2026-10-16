"""Unit conversion and metric/imperial rounding helpers.

All internal representations across the system use centimeters (cm).
"""

INCH_TO_CM = 2.54
CM_TO_INCH = 1.0 / 2.54


def cm_to_inches(cm: float, precision: int = 2) -> float:
    """Convert centimeters to inches with specified decimal precision."""
    return round(cm * CM_TO_INCH, precision)


def inches_to_cm(inches: float, precision: int = 2) -> float:
    """Convert inches to centimeters with specified decimal precision."""
    return round(inches * INCH_TO_CM, precision)



def round_measurement(val: float, step: float = 0.5) -> float:
    """Round a measurement to the nearest step (default 0.5 cm)."""
    return round(val / step) * step


def format_cm(cm: float, precision: int = 1) -> str:
    """Format centimeters as a human-readable string (e.g. '102.5 cm')."""
    return f"{cm:.{precision}f} cm"


def format_inches(inches: float, precision: int = 1) -> str:
    """Format inches as a human-readable string (e.g. '40.4 in')."""
    return f"{inches:.{precision}f} in"
