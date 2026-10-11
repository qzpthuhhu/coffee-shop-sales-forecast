import json
import unittest
from pathlib import Path

import pandas as pd

from export_dashboard_data import MODEL_DIR, build_dashboard_payload


ROOT = Path(__file__).parent


class DashboardExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = build_dashboard_payload(
            pd.read_csv(MODEL_DIR / "test_predictions.csv"),
            pd.read_csv(MODEL_DIR / "daily_predictions.csv"),
            pd.read_csv(MODEL_DIR / "shap_importance.csv"),
            generated_at="2026-10-12T00:00:00+00:00",
        )

    def test_snapshot_reports_current_holdout_metrics(self):
        metadata = self.payload["metadata"]
        metrics = self.payload["metrics"]

        self.assertEqual(metadata["validation_start"], "2023-06-24")
        self.assertEqual(metadata["validation_end"], "2023-06-30")
        self.assertEqual(metadata["validation_days"], 7)
        self.assertEqual(metadata["sample_count"], 1680)
        self.assertEqual(metadata["feature_count"], 44)
        self.assertAlmostEqual(metrics["mae"], 2.67)
        self.assertAlmostEqual(metrics["rmse"], 3.89)
        self.assertAlmostEqual(metrics["r2"], 0.652)
        self.assertAlmostEqual(metrics["within_10pct"], 11.7)
        self.assertAlmostEqual(metrics["within_20pct"], 27.3)

    def test_snapshot_does_not_restore_leaking_feature(self):
        feature_names = {
            row["feature"] for row in self.payload["feature_importance"]
        }
        self.assertNotIn("sku_share", feature_names)
        self.assertIn("sku_share_28d", feature_names)

    def test_generated_json_matches_current_artifacts(self):
        generated = json.loads(
            (ROOT / "dashboard" / "data.json").read_text(encoding="utf-8")
        )
        self.assertEqual(generated["metrics"], self.payload["metrics"])
        self.assertEqual(
            generated["metadata"]["sample_count"],
            self.payload["metadata"]["sample_count"],
        )

    def test_dashboard_has_no_browser_side_api_key_flow_or_false_model_claim(self):
        html = (ROOT / "dashboard" / "index.html").read_text(encoding="utf-8")

        self.assertNotIn("apiKeyInput", html)
        self.assertNotIn("openrouter_key", html)
        self.assertNotIn("openrouter.ai", html)
        self.assertNotIn("Global LightGBM", html)
        self.assertNotIn("0.961", html)
        self.assertNotIn("69.6%", html)


if __name__ == "__main__":
    unittest.main()
