"""Focused checks for data preparation, feature parsing, and scheduling."""

from pathlib import Path
import json
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from datasets.prepare_data import KEEP_COLUMNS, prepare, remove_price_outliers
from models.laptop_features import parse_battery_wh, parse_ram_gb, parse_screen_size, parse_storage_gb, parse_weight_kg
from models.train_model import train as train_model
from pipeline import run_on_schedule


class PrepareDataTests(TestCase):
    def test_output_files_have_no_missing_input_values(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            raw_path = root / "raw.csv"
            processed_dir = root / "processed"
            data = {column: ["Unknown"] * 5 for column in KEEP_COLUMNS if column != "Price"}
            data["Brand:"] = ["A", None, "B", "C", "D"]
            data["RAM"] = ["8GB", "16GB", None, "32GB", "64GB"]
            data["Price"] = [100, 101, 102, 103, 104]
            pd.DataFrame(data).to_csv(raw_path, index=False)

            prepare(raw_path, processed_dir, test_size=0.2, random_state=42)

            train = pd.read_csv(processed_dir / "train.csv")
            test = pd.read_csv(processed_dir / "test.csv")
            self.assertFalse(train.isna().any().any())
            self.assertFalse(test.isna().any().any())
            self.assertIn("Unknown", pd.concat([train, test])["Brand"].tolist())

    def test_missing_required_column_fails_before_writing_outputs(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            raw_path = root / "raw.csv"
            pd.DataFrame({"Price": [100, 200]}).to_csv(raw_path, index=False)

            with self.assertRaisesRegex(ValueError, "Missing required columns"):
                prepare(raw_path, root / "processed", test_size=0.2, random_state=42)
            self.assertFalse((root / "processed").exists())

    def test_extreme_train_price_is_removed(self) -> None:
        prices = [20_000 + index * 100 for index in range(40)] + [1_000_000]
        dataframe = pd.DataFrame({"Price": prices})
        cleaned = remove_price_outliers(dataframe)
        self.assertEqual(len(cleaned), 40)

    def test_real_premium_prices_are_preserved(self) -> None:
        raw_path = Path(__file__).resolve().parents[1] / "data/raw/laptops_Dataset.csv"
        with TemporaryDirectory() as temporary_dir:
            processed_dir = Path(temporary_dir)
            prepare(raw_path, processed_dir, test_size=0.2, random_state=42)
            train = pd.read_csv(processed_dir / "train.csv")
            test = pd.read_csv(processed_dir / "test.csv")
            raw_max = pd.read_csv(raw_path)["Price"].max()
            self.assertEqual(max(train["Price"].max(), test["Price"].max()), raw_max)


class FeatureParsingTests(TestCase):
    def test_ram_capacity_ignores_module_count(self) -> None:
        self.assertEqual(parse_ram_gb("1x 16GB SO-DIMM DDR5-4800"), 16)
        self.assertEqual(parse_ram_gb("96GB DDR5 48GB*2"), 96)

    def test_weight_and_screen_units(self) -> None:
        self.assertEqual(parse_weight_kg("895 g"), 0.895)
        self.assertEqual(parse_weight_kg("2.4 kg"), 2.4)
        self.assertEqual(parse_screen_size("39.6 cm (15.6) diagonal"), 15.6)
        self.assertEqual(parse_screen_size("39.6 cm (15.6 in) diagonal"), 15.6)

    def test_battery_capacity_in_parenthesized_units(self) -> None:
        self.assertEqual(parse_battery_wh("4-Cell 99 Battery (Whr)"), 99)
        self.assertEqual(parse_battery_wh("4-cell, 70W/h lithium-ion"), 70)
        self.assertEqual(parse_battery_wh("90WHrs, 4S1P, 4-cell Li-ion"), 90)
        self.assertIsNone(parse_battery_wh("65W AC adapter"))

    def test_storage_total_is_not_added_to_its_components(self) -> None:
        self.assertEqual(
            parse_storage_gb("6TB, 2TB NVMe SSD + 2TB NVMe SSD x2"),
            6144,
        )
        self.assertEqual(parse_storage_gb("1TB SSD + 512GB HDD"), 1536)


class TrainingIntegrationTests(TestCase):
    def test_trained_model_can_be_reloaded_and_predict(self) -> None:
        raw_path = Path(__file__).resolve().parents[1] / "data/raw/laptops_Dataset.csv"
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            processed = root / "processed"
            model_path = root / "model.joblib"
            metrics_path = root / "metrics.json"
            prepare(raw_path, processed, test_size=0.2, random_state=42)
            train_model(processed / "train.csv", processed / "test.csv", model_path, metrics_path, 42)

            model = joblib.load(model_path)
            test_data = pd.read_csv(processed / "test.csv")
            prediction = model.predict(test_data.drop(columns=["Price"]).head(1))[0]
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertGreater(prediction, 0)
            self.assertEqual(metrics["test_rows"], len(test_data))


class ScheduleTests(TestCase):
    def test_failed_run_is_retried_from_start_time(self) -> None:
        with (
            patch("pipeline.run_once", side_effect=[RuntimeError("failed"), KeyboardInterrupt]) as run,
            patch("pipeline.time.monotonic", side_effect=[100, 105, 400]),
            patch("pipeline.time.sleep") as sleep,
            patch("pipeline.logging.exception") as log_failure,
        ):
            with self.assertRaises(KeyboardInterrupt):
                run_on_schedule(deploy=True, interval_seconds=300)

        self.assertEqual(run.call_count, 2)
        sleep.assert_called_once_with(295)
        log_failure.assert_called_once()
