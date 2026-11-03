"""Validated response models for API information, readiness, and predictions."""

from typing import Literal

from pydantic import BaseModel, Field


class APIInfo(BaseModel):
    name: str
    model: str
    supported_formats: list[Literal["JPEG", "PNG"]]


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool


class Prediction(BaseModel):
    label: str = Field(min_length=1)
    score: float = Field(ge=0, le=1, allow_inf_nan=False)


class PredictionResponse(BaseModel):
    model: str
    predictions: list[Prediction] = Field(min_length=3, max_length=3)
