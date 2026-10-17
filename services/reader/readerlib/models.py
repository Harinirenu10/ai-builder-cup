import base64
import binascii
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from fitcore.schemas import Product, ReviewSignals

MAX_IMAGE_BYTES = 8 * 1024 * 1024


class ReadRequest(BaseModel):
    url: str


class ReadImageRequest(BaseModel):
    image: str = Field(description="Base64 of a size-chart screenshot (a data: URL is accepted)")
    page_url: str | None = None

    @field_validator("image")
    @classmethod
    def _check(cls, v: str) -> str:
        raw = v.split(",", 1)[1] if v.startswith("data:") else v
        try:
            data = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image must be base64") from exc
        if not data or len(data) > MAX_IMAGE_BYTES:
            raise ValueError("image is empty or larger than 8 MB")
        return raw


class ReadResponse(BaseModel):
    status: Literal["ok", "blocked", "unsupported", "error"]
    reason: str | None = None
    url: str | None = None
    product: Product | None = None
    reviews: ReviewSignals | None = None
    flags: list[str] = Field(default_factory=list)
    cached: bool = False
