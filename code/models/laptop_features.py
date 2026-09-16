"""Feature extraction utilities shared by training and serving."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


NUMERIC_FEATURES = [
    "ram_gb",
    "storage_gb",
    "screen_size_inches",
    "resolution_width",
    "resolution_height",
    "refresh_rate_hz",
    "weight_kg",
    "battery_wh",
    "processor_generation",
    "warranty_years",
]

CATEGORICAL_FEATURES = [
    "Brand",
    "Processor",
    "Video graphics",
    "Operating System",
    "Colors",
]

MODEL_INPUT_COLUMNS = [
    "Brand",
    "Processor",
    "Video graphics",
    "RAM",
    "Hard drive",
    "Display",
    "Display Resolution",
    "Display Refresh Rate",
    "Operating System",
    "Battery",
    "Weight",
    "Colors",
    "Warranty",
    "Processor Generation",
]


def first_number(value: Any) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else None


def parse_ram_gb(value: Any) -> float | None:
    return first_number(value)


def parse_storage_gb(value: Any) -> float | None:
    text = str(value).lower()
    matches = list(re.finditer(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", text))
    if not matches:
        return None
    amounts = [
        float(match.group(1)) * (1024 if match.group(2) == "tb" else 1)
        for match in matches
    ]
    if (
        len(matches) > 1
        and not text[: matches[0].start()].strip()
        and text[matches[0].end() : matches[1].start()].lstrip().startswith(",")
        and sum(amounts[1:]) <= amounts[0]
    ):
        # The leading capacity is a stated total; later capacities describe its components.
        return amounts[0]
    return sum(amounts)


def parse_screen_size(value: Any) -> float | None:
    text = str(value).lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:\"|''|inch|inches)", text)
    return float(match.group(1)) if match else first_number(value)


def parse_resolution(value: Any) -> tuple[float | None, float | None]:
    match = re.search(r"(\d{3,5})\s*[x*×]\s*(\d{3,5})", str(value), flags=re.I)
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def parse_refresh_rate(value: Any) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*hz", str(value), flags=re.I)
    return float(match.group(1)) if match else first_number(value)


def parse_battery_wh(value: Any) -> float | None:
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:battery\s*)?(?:\(\s*)?(?:wh(?:rs?)?|w/h)\b",
        str(value),
        flags=re.I,
    )
    return float(match.group(1)) if match else None


def parse_warranty_years(value: Any) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else None


class LaptopFeatureBuilder(BaseEstimator, TransformerMixin):
    """Convert raw laptop specification columns into model-ready feature columns."""

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> "LaptopFeatureBuilder":
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        dataframe = pd.DataFrame(x).copy()
        if "Brand:" in dataframe.columns and "Brand" not in dataframe.columns:
            dataframe = dataframe.rename(columns={"Brand:": "Brand"})

        output = pd.DataFrame(index=dataframe.index)
        output["ram_gb"] = dataframe.get("RAM", pd.Series(index=dataframe.index)).map(parse_ram_gb)
        output["storage_gb"] = dataframe.get("Hard drive", pd.Series(index=dataframe.index)).map(parse_storage_gb)
        output["screen_size_inches"] = dataframe.get("Display", pd.Series(index=dataframe.index)).map(parse_screen_size)

        resolution = dataframe.get("Display Resolution", pd.Series(index=dataframe.index)).map(parse_resolution)
        output["resolution_width"] = resolution.map(lambda item: item[0])
        output["resolution_height"] = resolution.map(lambda item: item[1])

        output["refresh_rate_hz"] = dataframe.get("Display Refresh Rate", pd.Series(index=dataframe.index)).map(parse_refresh_rate)
        output["weight_kg"] = dataframe.get("Weight", pd.Series(index=dataframe.index)).map(first_number)
        output["battery_wh"] = dataframe.get("Battery", pd.Series(index=dataframe.index)).map(parse_battery_wh)
        output["processor_generation"] = dataframe.get("Processor Generation", pd.Series(index=dataframe.index)).map(first_number)
        output["warranty_years"] = dataframe.get("Warranty", pd.Series(index=dataframe.index)).map(parse_warranty_years)

        for column in CATEGORICAL_FEATURES:
            output[column] = dataframe.get(column, pd.Series(index=dataframe.index)).fillna("Unknown").astype(str)

        return output[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
