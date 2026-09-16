"""Train and evaluate a laptop price prediction model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from models.laptop_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, LaptopFeatureBuilder


TARGET = "Price"
DEFAULT_TRAIN_PATH = Path("data/processed/train.csv")
DEFAULT_TEST_PATH = Path("data/processed/test.csv")
DEFAULT_MODEL_PATH = Path("models/laptop_price_model.joblib")
DEFAULT_METRICS_PATH = Path("models/metrics.json")


def build_pipeline(random_state: int) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("features", LaptopFeatureBuilder()),
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train(train_path: Path, test_path: Path, model_path: Path, metrics_path: Path, random_state: int) -> None:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    x_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    x_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    pipeline = build_pipeline(random_state)
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    mse = mean_squared_error(y_test, predictions)
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 2),
        "rmse": round(float(mse**0.5), 2),
        "r2": round(float(r2_score(y_test, predictions)), 4),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "model": "RandomForestRegressor",
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.train_path, args.test_path, args.model_path, args.metrics_path, args.random_state)
