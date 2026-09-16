"""FastAPI service for laptop price predictions."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from models.laptop_features import (
    parse_ram_gb,
    parse_resolution,
    parse_screen_size,
    parse_storage_gb,
    parse_weight_kg,
)


MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/laptop_price_model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "models/metrics.json"))


class LaptopInput(BaseModel):
    brand: str = Field(..., min_length=1, examples=["Lenovo"])
    processor: str = Field(..., min_length=1, examples=["Intel Core i7-13620H"])
    video_graphics: str = Field("Unknown", examples=["Nvidia GeForce RTX 4060 8GB"])
    ram: str = Field(..., min_length=1, examples=["16GB DDR5"])
    hard_drive: str = Field(..., min_length=1, examples=["1TB NVMe PCIe SSD"])
    display: str = Field(..., min_length=1, examples=['15.6" FHD IPS 144Hz'])
    display_resolution: str = Field(..., min_length=1, examples=["1920x1080"])
    display_refresh_rate: str = Field("Unknown", examples=["144 Hz"])
    operating_system: str = Field("Unknown", examples=["Windows 11 Home"])
    battery: str = Field("Unknown", examples=["60 Wh"])
    weight: str = Field(..., min_length=1, examples=["2.2 kg"])
    colors: str = Field("Unknown", examples=["Black"])
    warranty: str = Field("Unknown", examples=["1 Year"])
    processor_generation: str = Field("Unknown", examples=["13th generation"])

    @field_validator("*")
    @classmethod
    def strip_nonempty_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned

    @field_validator("ram", "hard_drive", "display", "display_resolution", "weight")
    @classmethod
    def validate_key_specifications(cls, value: str, info: ValidationInfo) -> str:
        if info.field_name == "display_resolution":
            width, height = parse_resolution(value)
            valid = width is not None and height is not None and width >= 640 and height >= 480
        else:
            parsers = {
                "ram": (parse_ram_gb, 1, 256),
                "hard_drive": (parse_storage_gb, 32, 16_384),
                "display": (parse_screen_size, 8, 25),
                "weight": (parse_weight_kg, 0.2, 10),
            }
            parser, minimum, maximum = parsers[info.field_name]
            parsed = parser(value)
            valid = parsed is not None and minimum <= parsed <= maximum
            if info.field_name == "weight":
                valid = valid and bool(re.search(r"\d\s*(?:kg|g)\b", value, flags=re.I))
        if not valid:
            raise ValueError(f"invalid {info.field_name} specification")
        return value


def model_input_to_dataframe(payload: LaptopInput) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Brand": payload.brand,
                "Processor": payload.processor,
                "Video graphics": payload.video_graphics,
                "RAM": payload.ram,
                "Hard drive": payload.hard_drive,
                "Display": payload.display,
                "Display Resolution": payload.display_resolution,
                "Display Refresh Rate": payload.display_refresh_rate,
                "Operating System": payload.operating_system,
                "Battery": payload.battery,
                "Weight": payload.weight,
                "Colors": payload.colors,
                "Warranty": payload.warranty,
                "Processor Generation": payload.processor_generation,
            }
        ]
    )


def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file was not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


app = FastAPI(title="Laptop Price Prediction API", version="1.0.0")
model: Any | None = None


@app.on_event("startup")
def startup() -> None:
    global model
    model = load_model()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_path": str(MODEL_PATH)}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.post("/predict")
def predict(payload: LaptopInput) -> dict[str, float]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    prediction = float(model.predict(model_input_to_dataframe(payload))[0])
    return {"predicted_price": round(prediction, 2)}
