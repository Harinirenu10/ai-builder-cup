"""Builders shared by the fit engine tests."""

from fitcore.schemas import Measurement, Measurements, MeasurementZone, Product, SizeChart

Z = MeasurementZone


def body(**cm: float) -> Measurements:
    return Measurements(zones={Z(k): Measurement(value_cm=v, confidence=0.9, source="test") for k, v in cm.items()})


def product(category: str, sizes: dict, *, chart_type: str = "garment", fit_type: str = "regular", fabric=None,
            stretch: str = "none", pre_shrunk=None, unit: str = "cm") -> Product:
    return Product(
        id="p", brand="b", title="t", category=category, fit_type=fit_type,
        fabric=fabric if fabric is not None else [{"fiber": "cotton", "percent": 100}],
        stretch_class=stretch, pre_shrunk=pre_shrunk,
        size_chart=SizeChart(sizes={s: {Z(k): v for k, v in row.items()} for s, row in sizes.items()}, chart_type=chart_type, unit=unit),
    )
