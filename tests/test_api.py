"""HTTP contract checks for the prediction API."""

import json
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from deployment.api.main import LaptopInput, app, model_input_to_dataframe


class FakeModel:
    def predict(self, dataframe):
        assert len(dataframe) == 1
        return [12_345.67]


class ApiTests(TestCase):
    def setUp(self) -> None:
        self.payload = json.loads(
            (Path(__file__).resolve().parents[1] / "sample_request.json").read_text(encoding="utf-8")
        )
        self.model_patch = patch("deployment.api.main.load_model", return_value=FakeModel())
        self.model_patch.start()
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.model_patch.stop()

    def test_prediction_uses_provided_specs(self) -> None:
        response = self.client.post("/predict", json=self.payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predicted_price": 12_345.67})

    def test_empty_request_is_rejected(self) -> None:
        response = self.client.post("/predict", json={})
        self.assertEqual(response.status_code, 422)

    def test_missing_optional_specs_remain_unknown(self) -> None:
        required = {
            key: self.payload[key]
            for key in ("brand", "processor", "ram", "hard_drive", "display", "display_resolution", "weight")
        }
        response = self.client.post("/predict", json=required)
        self.assertEqual(response.status_code, 200)
        dataframe = model_input_to_dataframe(LaptopInput(**required))
        self.assertEqual(dataframe.loc[0, "Video graphics"], "Unknown")

    def test_invalid_ram_and_weight_are_rejected(self) -> None:
        self.payload.update(ram="1x banana", weight="895 bananas")
        response = self.client.post("/predict", json=self.payload)
        self.assertEqual(response.status_code, 422)
        fields = {error["loc"][-1] for error in response.json()["detail"]}
        self.assertEqual(fields, {"ram", "weight"})
