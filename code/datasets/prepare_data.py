"""Prepare raw laptop data for model training."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_RAW_PATH = Path("data/raw/laptops_Dataset.csv")
DEFAULT_PROCESSED_DIR = Path("data/processed")
TARGET = "Price"


KEEP_COLUMNS = [
    "Brand:",
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
    TARGET,
]


def remove_price_outliers(dataframe: pd.DataFrame) -> pd.DataFrame:
    log_price = dataframe[TARGET].map(math.log)
    q1 = log_price.quantile(0.25)
    q3 = log_price.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 2.5 * iqr
    upper = q3 + 2.5 * iqr
    return dataframe[log_price.between(lower, upper)].copy()


def prepare(raw_path: Path, processed_dir: Path, test_size: float, random_state: int) -> None:
    dataframe = pd.read_csv(raw_path, encoding="utf-8-sig")
    raw_rows = len(dataframe)
    missing_columns = sorted(set(KEEP_COLUMNS) - set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    dataframe = dataframe[KEEP_COLUMNS].rename(columns={"Brand:": "Brand"})
    dataframe = dataframe.drop_duplicates()
    dataframe = dataframe.dropna(subset=[TARGET])
    dataframe = dataframe[dataframe[TARGET] > 0]
    feature_columns = dataframe.columns.drop(TARGET)
    dataframe[feature_columns] = dataframe[feature_columns].fillna("Unknown")

    train, test = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=random_state,
    )
    train_rows_before_outliers = len(train)
    train = remove_price_outliers(train)

    processed_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(processed_dir / "train.csv", index=False)
    test.to_csv(processed_dir / "test.csv", index=False)

    summary = pd.DataFrame(
        [
            {"artifact": "raw_rows", "value": raw_rows},
            {"artifact": "prepared_rows", "value": len(train) + len(test)},
            {"artifact": "train_outliers_removed", "value": train_rows_before_outliers - len(train)},
            {"artifact": "train_rows", "value": len(train)},
            {"artifact": "test_rows", "value": len(test)},
            {"artifact": "columns", "value": len(dataframe.columns)},
        ]
    )
    summary.to_csv(processed_dir / "data_summary.csv", index=False)
    print(f"Saved {len(train)} train rows and {len(test)} test rows to {processed_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(args.raw_path, args.processed_dir, args.test_size, args.random_state)
