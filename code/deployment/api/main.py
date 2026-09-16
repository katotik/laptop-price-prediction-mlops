"""FastAPI service for laptop price predictions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/laptop_price_model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "models/metrics.json"))


class LaptopInput(BaseModel):
    brand: str = Field("Lenovo", examples=["Lenovo"])
    processor: str = Field("Intel Core i7-13620H", examples=["Intel Core i7-13620H"])
    video_graphics: str = Field("Nvidia GeForce RTX 4060 8GB", examples=["Nvidia GeForce RTX 4060 8GB"])
    ram: str = Field("16GB DDR5", examples=["16GB DDR5"])
    hard_drive: str = Field("1TB NVMe PCIe SSD", examples=["1TB NVMe PCIe SSD"])
    display: str = Field('15.6" FHD IPS 144Hz', examples=['15.6" FHD IPS 144Hz'])
    display_resolution: str = Field("1920x1080", examples=["1920x1080"])
    display_refresh_rate: str = Field("144 Hz", examples=["144 Hz"])
    operating_system: str = Field("Windows 11 Home", examples=["Windows 11 Home"])
    battery: str = Field("60 Wh", examples=["60 Wh"])
    weight: str = Field("2.2 kg", examples=["2.2 kg"])
    colors: str = Field("Black", examples=["Black"])
    warranty: str = Field("1 Year", examples=["1 Year"])
    processor_generation: str = Field("13th generation", examples=["13th generation"])


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
