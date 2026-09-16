"""Focused checks for data preparation, feature parsing, and scheduling."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from datasets.prepare_data import prepare
from models.laptop_features import parse_battery_wh, parse_storage_gb
from pipeline import run_on_schedule


class PrepareDataTests(TestCase):
    def test_output_files_have_no_missing_input_values(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            raw_path = root / "raw.csv"
            processed_dir = root / "processed"
            pd.DataFrame(
                {
                    "Brand:": ["A", None, "B", "C", "D"],
                    "RAM": ["8GB", "16GB", None, "32GB", "64GB"],
                    "Price": [100, 101, 102, 103, 104],
                }
            ).to_csv(raw_path, index=False)

            prepare(raw_path, processed_dir, test_size=0.2, random_state=42)

            train = pd.read_csv(processed_dir / "train.csv")
            test = pd.read_csv(processed_dir / "test.csv")
            self.assertFalse(train.isna().any().any())
            self.assertFalse(test.isna().any().any())
            self.assertIn("Unknown", pd.concat([train, test])["Brand"].tolist())


class FeatureParsingTests(TestCase):
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
