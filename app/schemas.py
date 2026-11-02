from typing import Literal, Optional
from pydantic import BaseModel, Field

class ForecastRequest(BaseModel):
    horizon: int = Field(default=24, ge=1, le=1680)
    model: Literal["global", "chronos"] = "global"
    interval_width: float = Field(default=0.90, gt=0.0, lt=1.0)
    use_exogenous: bool = True

class ForecastPoint(BaseModel):
    timestamp: str
    prediction: float
    lower_50: float
    upper_50: float
    lower_90: float
    upper_90: float

class ForecastResponse(BaseModel):
    series_id: str
    model: str
    horizon: int
    forecasts: list[ForecastPoint]
    metrics: dict
    warnings: list[str] = []
