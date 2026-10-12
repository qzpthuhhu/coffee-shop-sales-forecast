import json
import unittest
from pathlib import Path

import pandas as pd

from export_dashboard_data import (
    REPORT_DIR,
    build_dashboard_payload,
    build_explanation_shards,
)


ROOT = Path(__file__).parent


class DashboardExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictions = pd.read_csv(
            REPORT_DIR / "rolling_backtest_predictions.csv"
        )
        cls.quality = pd.read_csv(REPORT_DIR / "rolling_backtest_quality.csv")
        cls.shap = pd.read_csv(REPORT_DIR / "rolling_backtest_shap.csv.gz")
        cls.run_metadata = json.loads(
            (REPORT_DIR / "rolling_backtest_run.json").read_text(
                encoding="utf-8"
            )
        )
        cls.payload = build_dashboard_payload(
            cls.predictions,
            cls.quality,
            cls.shap,
            cls.run_metadata,
            generated_at="2026-10-12T00:00:00+00:00",
        )

    def test_snapshot_reports_current_rolling_metrics(self):
        metadata = self.payload["metadata"]
        metrics = self.payload["metrics"]

        self.assertEqual(metadata["validation_start"], "2023-05-27")
        self.assertEqual(metadata["validation_end"], "2023-06-30")
        self.assertEqual(metadata["validation_days"], 35)
        self.assertEqual(metadata["backtest_windows"], 5)
        self.assertEqual(metadata["sample_count"], 8400)
        self.assertEqual(metadata["feature_count"], 44)
        self.assertEqual(
            metadata["model_name"],
            "Global LightGBM (lightgbm.LGBMRegressor)",
        )
        self.assertAlmostEqual(metrics["mae"], 2.655)
        self.assertAlmostEqual(metrics["rmse"], 3.991)
        self.assertAlmostEqual(metrics["wape"], 38.2)
        self.assertAlmostEqual(metrics["r2"], 0.592)
        self.assertAlmostEqual(metrics["interval_coverage"], 80.6)
        self.assertTrue(metrics["candidate_eligible"])

    def test_snapshot_does_not_restore_leaking_feature(self):
        feature_names = {
            row["feature"] for row in self.payload["feature_importance"]
        }
        self.assertNotIn("sku_share", feature_names)
        self.assertIn("sku_share_28d", feature_names)

    def test_snapshot_exposes_all_filter_paths_and_daily_shards(self):
        self.assertEqual(len(self.payload["hierarchy"]), 240)
        self.assertEqual(len(self.payload["dates"]), 35)
        self.assertEqual(len(self.payload["explanation_shards"]), 35)
        self.assertEqual(
            self.payload["explanation_shards"]["2023-06-27"],
            "explanations/2023-06-27.json",
        )

    def test_explanation_shard_preserves_prediction_additivity(self):
        sample_prediction = self.predictions.iloc[[0]]
        prediction_id = sample_prediction.iloc[0]["prediction_id"]
        sample_shap = self.shap[self.shap["prediction_id"] == prediction_id]
        shards = build_explanation_shards(
            sample_prediction,
            sample_shap,
            generated_at="2026-10-12T00:00:00+00:00",
        )
        record = shards["2023-05-27"]["records"][0]

        self.assertEqual(len(record["contributions"]), 11)
        reconstructed = record["base_value"] + sum(
            item["contribution"] for item in record["contributions"]
        )
        self.assertAlmostEqual(reconstructed, record["raw_prediction"], places=5)
        self.assertAlmostEqual(
            record["raw_prediction"] + record["postprocess_adjustment"],
            record["prediction"],
            places=5,
        )
        self.assertTrue(self.payload["feature_catalog"]["qty_lag_1"]["label"])
        self.assertIn(
            "不代表因果",
            self.payload["feature_catalog"]["qty_lag_1"][
                "increase_explanation"
            ],
        )

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
        self.assertIn("Global LightGBM", html)
        self.assertNotIn("HistGradientBoosting", html)
        self.assertNotIn("0.961", html)
        self.assertNotIn("69.6%", html)
        self.assertIn("模型贡献，不代表因果", html)
        self.assertIn("逐预测解释", html)


if __name__ == "__main__":
    unittest.main()
