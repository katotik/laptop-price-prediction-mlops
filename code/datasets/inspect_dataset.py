"""Inspect a raw laptop dataset without modifying it."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PARSING_PATTERNS = {
    "RAM": r"\d+(?:\.\d+)?\s*GB",
    "Hard drive": r"\d+(?:\.\d+)?\s*(?:TB|GB)",
    "Display": r"\d+(?:\.\d+)?\s*(?:\"|″|''|inch|inches)",
    "Display Resolution": r"\d{3,5}\s*[x*×]\s*\d{3,5}",
    "Display Refresh Rate": r"\d+\s*Hz",
    "Dimensions": r"\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?",
    "Weight": r"\d+(?:\.\d+)?\s*kg",
    "Battery": r"\d+(?:\.\d+)?\s*(?:Wh|Whr)",
    "Warranty": r"\d+(?:\.\d+)?\s*year",
}


def print_section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def print_column_list(title: str, columns: list[str]) -> None:
    print_section(title)
    for column in columns:
        print(f"- {column}")


def inspect_dataset(path: Path) -> None:
    dataframe = pd.read_csv(path)

    print_section("1. FILE")
    print(path)

    print_section("2. SHAPE")
    print(f"Rows: {len(dataframe)}")
    print(f"Columns: {len(dataframe.columns)}")

    print_section("3-4. COLUMNS AND DATA TYPES")
    for number, (column, dtype) in enumerate(dataframe.dtypes.items(), start=1):
        print(f"{number:02d}. {column!r}: {dtype}")

    print_section("5. TARGET COLUMN")
    target = "Price" if "Price" in dataframe.columns else None
    print(target if target else "Price column was not found")

    print_section("6. MISSING VALUES")
    missing = dataframe.isna().sum()
    for column, count in missing.items():
        percent = count / len(dataframe) * 100
        print(f"{column!r}: {count} ({percent:.1f}%)")

    print_section("7. DUPLICATES")
    print(f"Full duplicate rows: {dataframe.duplicated().sum()}")
    print(f"Rows in duplicate groups: {dataframe.duplicated(keep=False).sum()}")

    print_column_list(
        "8. CATEGORICAL COLUMNS",
        dataframe.select_dtypes(include=["object", "category"]).columns.tolist(),
    )
    print_column_list(
        "9. NUMERICAL COLUMNS",
        dataframe.select_dtypes(include="number").columns.tolist(),
    )

    print_section("10. PARSING AND CLEANING CANDIDATES")
    for column, pattern in PARSING_PATTERNS.items():
        if column not in dataframe.columns:
            continue
        values = dataframe[column].dropna().astype(str)
        matched = values.str.contains(pattern, case=False, regex=True).sum()
        coverage = matched / len(values) * 100 if len(values) else 0
        print(f"{column!r}: {matched}/{len(values)} values match a unit pattern ({coverage:.1f}%)")
    if "Price" in dataframe.columns:
        price_text = dataframe["Price"].astype(str)
        currency_values = price_text.str.contains(
            r"[$€£₽₹]|EGP|USD", case=False, regex=True
        ).sum()
        print(f"'Price': currency/text markers found in {currency_values} values")

    print_section("11. SPARSE OR SUSPICIOUS COLUMNS")
    for column in dataframe.columns:
        non_null = dataframe[column].notna().sum()
        if non_null < len(dataframe) * 0.10:
            print(f"{column!r}: only {non_null}/{len(dataframe)} values are present")
    index_columns = [
        column
        for column in dataframe.columns
        if column.lower().startswith("unnamed") or column.lower() in {"id", "index"}
    ]
    print(f"Index-like columns: {index_columns or 'none found'}")
    print(f"All-null columns: {dataframe.columns[dataframe.isna().all()].tolist()}")

    print_section("12. POSSIBLE PRICE OUTLIERS")
    if target:
        price = dataframe[target]
        q1, q3 = price.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = (price < lower_bound) | (price > upper_bound)
        print(price.describe().to_string())
        print(f"IQR bounds: {lower_bound:.2f} .. {upper_bound:.2f}")
        print(f"Possible IQR outliers: {outliers.sum()}")
        print("Highest prices:")
        print(
            dataframe.nlargest(5, target)
            [[column for column in ["Brand:", "Product name", target] if column in dataframe]]
            .to_string(index=False)
        )

    print_section("13. RECOMMENDED FEATURES")
    recommended_numeric = [
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
    recommended_categorical = [
        column
        for column in ["Brand:", "Processor", "Video graphics", "Operating System", "Colors"]
        if column in dataframe.columns
    ]
    print("Numeric features after parsing:")
    for feature in recommended_numeric:
        print(f"- {feature}")
    print("Categorical features:")
    for feature in recommended_categorical:
        print(f"- {feature}")
    print("Exclude initially: Product number, Product name, long free-text fields, and nearly empty columns.")

    print_section("PREPROCESSING PLAN")
    print("1. Rename 'Brand:' to 'Brand'.")
    print("2. Remove full duplicate rows before splitting the data.")
    print("3. Drop nearly empty and identifier-like columns.")
    print("4. Parse units into numeric features listed above.")
    print("5. Fill numeric missing values with train-set medians.")
    print("6. Fill categorical missing values with 'Unknown'.")
    print("7. One-hot encode selected categorical features with unknown-category handling.")
    print("8. Split into train/test before fitting imputers and encoders.")
    print("9. Filter extreme prices in the training split only; keep the full test price range.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=Path("data/raw/laptops_Dataset.csv"),
        help="Path to the raw CSV file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    inspect_dataset(arguments.csv_path)
