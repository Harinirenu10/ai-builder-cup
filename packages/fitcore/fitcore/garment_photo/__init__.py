"""Garment flat-lay photo measurement: A4-referenced homography rectification,
Gemini Flash keypoint detection, and deterministic centimeter measurement.
"""

from fitcore.garment_photo.pipeline import (
    A4_NOT_FOUND_MESSAGE,
    A4NotFoundError,
    GarmentPhotoPipelineResult,
    measure_garment_photo,
)
from fitcore.garment_photo.schemas import (
    CATEGORY_KEYPOINTS,
    GARMENT_ZONE_DEFINITIONS,
    A4DetectionResult,
    FitPrefsDocument,
    FitsTwinVerdict,
    GarmentCategory,
    GarmentFlatMeasurements,
    GarmentMeasurement,
    GarmentQualityFlags,
    GeminiGarmentKeypoints,
    KeypointXY,
    LabelCheck,
    UnboxingResult,
)

__all__ = [
    "A4_NOT_FOUND_MESSAGE",
    "A4NotFoundError",
    "GarmentPhotoPipelineResult",
    "measure_garment_photo",
    "CATEGORY_KEYPOINTS",
    "GARMENT_ZONE_DEFINITIONS",
    "A4DetectionResult",
    "FitPrefsDocument",
    "FitsTwinVerdict",
    "GarmentCategory",
    "GarmentFlatMeasurements",
    "GarmentMeasurement",
    "GarmentQualityFlags",
    "GeminiGarmentKeypoints",
    "KeypointXY",
    "LabelCheck",
    "UnboxingResult",
]
