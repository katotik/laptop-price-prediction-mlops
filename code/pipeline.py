"""Run the full data, model, and deployment pipeline."""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path

from datasets.prepare_data import prepare
from models.train_model import train


PROCESSED_DIR = Path("data/processed")
RAW_PATH = Path("data/raw/laptops_Dataset.csv")
MODEL_PATH = Path("models/laptop_price_model.joblib")
METRICS_PATH = Path("models/metrics.json")
COMPOSE_PATH = Path("code/deployment/docker-compose.yml")


def run_once(deploy: bool) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Starting pipeline run")
    prepare(RAW_PATH, PROCESSED_DIR, test_size=0.2, random_state=42)
    train(PROCESSED_DIR / "train.csv", PROCESSED_DIR / "test.csv", MODEL_PATH, METRICS_PATH, random_state=42)
    if deploy:
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_PATH), "up", "--build", "-d"],
            check=True,
        )
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Pipeline run finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy", action="store_true", help="Build and start API/app containers after training.")
    parser.add_argument("--watch", action="store_true", help="Run forever on a fixed schedule.")
    parser.add_argument("--interval-seconds", type=int, default=300, help="Schedule interval; default is 5 minutes.")
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    return args


def run_on_schedule(deploy: bool, interval_seconds: int) -> None:
    while True:
        started_at = time.monotonic()
        try:
            run_once(deploy=deploy)
        except Exception:
            logging.exception("Pipeline run failed; retrying on the next interval")
        time.sleep(max(0, interval_seconds - (time.monotonic() - started_at)))


if __name__ == "__main__":
    args = parse_args()
    if args.watch:
        run_on_schedule(deploy=args.deploy, interval_seconds=args.interval_seconds)
    else:
        run_once(deploy=args.deploy)
