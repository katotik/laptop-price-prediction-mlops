"""Prepare raw laptop data for model training."""

from __future__ import annotations

import argparse
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
    q1 = dataframe[TARGET].quantile(0.25)
    q3 = dataframe[TARGET].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return dataframe[dataframe[TARGET].between(lower, upper)].copy()


def prepare(raw_path: Path, processed_dir: Path, test_size: float, random_state: int) -> None:
    dataframe = pd.read_csv(raw_path, encoding="utf-8-sig")
    dataframe = dataframe.rename(columns={"Brand:": "Brand"})

    keep_columns = [column.replace("Brand:", "Brand") for column in KEEP_COLUMNS]
    dataframe = dataframe[[column for column in keep_columns if column in dataframe.columns]]
    dataframe = dataframe.drop_duplicates()
    dataframe = dataframe.dropna(subset=[TARGET])
    dataframe = dataframe[dataframe[TARGET] > 0]
    dataframe = remove_price_outliers(dataframe)

    train, test = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=random_state,
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(processed_dir / "train.csv", index=False)
    test.to_csv(processed_dir / "test.csv", index=False)

    summary = pd.DataFrame(
        [
            {"artifact": "raw_rows", "value": len(pd.read_csv(raw_path, encoding="utf-8-sig"))},
            {"artifact": "prepared_rows", "value": len(dataframe)},
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
