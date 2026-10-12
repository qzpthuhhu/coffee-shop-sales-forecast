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
        cls.scenarios = pd.read_csv(
            REPORT_DIR / "rolling_backtest_scenarios.csv.gz"
        )
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
        self.assertEqual(metadata["feature_count"], 53)
        self.assertEqual(
            metadata["model_name"],
            "Global LightGBM (direct Tweedie + sparse labels)",
        )
        self.assertAlmostEqual(metrics["mae"], 2.411)
        self.assertAlmostEqual(metrics["rmse"], 3.636)
        self.assertAlmostEqual(metrics["wape"], 34.7)
        self.assertAlmostEqual(metrics["r2"], 0.661)
        self.assertAlmostEqual(metrics["interval_coverage"], 80.6)
        self.assertTrue(metrics["candidate_eligible"])

    def test_snapshot_does_not_restore_leaking_feature(self):
        feature_names = {
            row["feature"] for row in self.payload["feature_importance"]
        }
        self.assertNotIn("sku_share", feature_names)
        self.assertIn("adi", feature_names)

    def test_snapshot_exposes_all_filter_paths_and_daily_shards(self):
        self.assertEqual(len(self.payload["hierarchy"]), 240)
        self.assertEqual(len(self.payload["dates"]), 35)
        self.assertEqual(len(self.payload["explanation_shards"]), 35)
        self.assertEqual(
            self.payload["explanation_shards"]["2023-06-27"],
            "explanations/2023-06-27.json",
        )

    def test_snapshot_exposes_readable_bilingual_product_names(self):
        products = {
            row["sku"]: row
            for row in self.payload["hierarchy"]
            if row["store"] == "Astoria"
        }

        self.assertEqual(products[38]["product_name"], "Latte")
        self.assertEqual(products[38]["product_name_zh"], "拿铁")
        self.assertEqual(products[38]["product_label"], "拿铁（Latte）")
        self.assertEqual(
            products[57]["product_label"],
            "醒神辛香印度香料茶·大杯（Spicy Eye Opener Chai Lg）",
        )
        self.assertEqual(
            self.payload["labels"]["stores"]["Lower Manhattan"],
            "下曼哈顿",
        )
        self.assertEqual(
            self.payload["labels"]["categories"]["Coffee"],
            "现制咖啡",
        )
        self.assertEqual(len(self.payload["product_predictions"]), 80)
        self.assertEqual(
            self.payload["product_predictions"]["38"]["label"],
            "拿铁（Latte）",
        )
        self.assertEqual(
            len(self.payload["product_predictions"]["38"]["dates"]),
            35,
        )

    def test_diagnostics_are_bounded_and_cover_all_errors(self):
        diagnostics = self.payload["diagnostics"]

        self.assertLessEqual(len(diagnostics["scatter"]), 870)
        self.assertGreater(len(diagnostics["scatter"]), 800)
        self.assertEqual(
            sum(row["count"] for row in diagnostics["histogram"]),
            len(self.predictions),
        )
        self.assertEqual(
            diagnostics["histogram_sample_count"],
            len(self.predictions),
        )
        self.assertAlmostEqual(diagnostics["mean_error"], 0.483, places=3)
        self.assertIn("确定性抽样", diagnostics["scatter_sampling"])

    def test_demand_heatmap_covers_each_store_sku_and_date(self):
        heatmap = self.payload["demand_heatmap"]

        self.assertEqual(len(heatmap["dates"]), 35)
        self.assertEqual(len(heatmap["rows"]), 240)
        self.assertEqual(
            sum(len(row["values"]) for row in heatmap["rows"]),
            len(self.predictions),
        )
        self.assertTrue(
            all(
                len(row["values"]) == len(heatmap["dates"])
                for row in heatmap["rows"]
            )
        )
        self.assertEqual(heatmap["unit"], "杯 / SKU / 日")
        self.assertNotIn(None, heatmap["rows"][0]["values"])

    def test_risk_ranking_is_bounded_and_does_not_use_actual_sales(self):
        risk = self.payload["risk_rankings"]

        self.assertFalse(risk["uses_actual_sales"])
        self.assertEqual(len(risk["by_date"]), 35)
        self.assertEqual(
            set(risk["by_date"]["2023-06-30"]),
            {"__all__", "Astoria", "Hell's Kitchen", "Lower Manhattan"},
        )
        for scopes in risk["by_date"].values():
            for rows in scopes.values():
                self.assertLessEqual(len(rows), 10)
                self.assertEqual(
                    [row["upside_gap"] for row in rows],
                    sorted(
                        [row["upside_gap"] for row in rows],
                        reverse=True,
                    ),
                )
                self.assertTrue(
                    all(
                        {"actual", "qty", "error"}.isdisjoint(row)
                        for row in rows
                    )
                )
        top = risk["by_date"]["2023-06-30"]["__all__"][0]
        self.assertEqual(top["store"], "Hell's Kitchen")
        self.assertEqual(top["sku"], 65)
        self.assertEqual(top["product_name_zh"], "无糖香草糖浆")
        self.assertAlmostEqual(top["upside_gap"], 10.2)

    def test_explanation_shard_preserves_prediction_additivity(self):
        sample_prediction = self.predictions.iloc[[0]]
        prediction_id = sample_prediction.iloc[0]["prediction_id"]
        sample_shap = self.shap[self.shap["prediction_id"] == prediction_id]
        shards = build_explanation_shards(
            sample_prediction,
            sample_shap,
            generated_at="2026-10-12T00:00:00+00:00",
            scenarios=self.scenarios[
                self.scenarios["prediction_id"] == prediction_id
            ],
        )
        record = shards["2023-05-27"]["records"][0]

        self.assertEqual(len(record["contributions"]), 11)
        reconstructed = record["base_value"] + sum(
            item["contribution"] for item in record["contributions"]
        )
        self.assertAlmostEqual(reconstructed, record["raw_prediction"], places=5)
        self.assertAlmostEqual(
            record["raw_prediction"]
            + record["postprocess_adjustment"]
            + record["store_bias_adjustment"],
            record["prediction"],
            places=5,
        )
        self.assertAlmostEqual(
            record["raw_prediction"] + record["postprocess_adjustment"],
            record["model_prediction"],
            places=5,
        )
        self.assertTrue(self.payload["feature_catalog"]["qty_lag_1"]["label"])
        self.assertEqual(len(record["scenario_predictions"]), 27)
        self.assertTrue(record["product_name"])
        self.assertTrue(record["product_name_zh"])
        self.assertIn(record["product_name"], record["product_label"])
        self.assertEqual(
            record["scenario_inputs"]["temperature_2m_mean"],
            self.scenarios[
                (self.scenarios["prediction_id"] == prediction_id)
                & (self.scenarios["temperature_id"] == "observed")
                & (self.scenarios["rain_id"] == "observed")
                & (self.scenarios["holiday_id"] == "observed")
            ].iloc[0]["temperature_2m_mean"],
        )
        self.assertFalse(self.payload["scenario_contract"]["causal"])
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
        self.assertNotIn("HistGradientBoosting", html)
        self.assertNotIn("0.961", html)
        self.assertNotIn("69.6%", html)
        self.assertNotIn('id="m-wape"', html)
        self.assertNotIn('id="m-r2"', html)
        self.assertIn("咖啡门店销量预测助手", html)
        self.assertIn("商品每日需求热力图", html)
        self.assertIn("为什么这样预测", html)
        self.assertIn("productLabel(row)", html)
        self.assertIn("[hidden] { display:none!important; }", html)
        self.assertIn('<option value="product">按具体商品查看</option>', html)
        self.assertIn("DATA.product_predictions", html)
        self.assertIn('id="chart-context"', html)
        self.assertIn("不等于真实因果", html)
        self.assertIn("预测是否贴近实际", html)
        self.assertIn("误差分布", html)
        self.assertIn("商品每日需求热力图", html)
        self.assertIn("data-heatmap-row", html)
        self.assertIn("销量日历", html)
        self.assertIn('id="calendar-store"', html)
        self.assertIn('id="sales-calendar"', html)
        self.assertIn("function renderCalendar()", html)
        self.assertIn("模型响应模拟，不是因果推断", html)
        self.assertIn('id="scenario-temperature"', html)
        self.assertIn('id="scenario-rain"', html)
        self.assertIn('id="scenario-holiday"', html)
        self.assertIn("function renderScenario", html)
        self.assertIn("需要重点关注的商品", html)
        self.assertIn('id="risk-ranking"', html)
        self.assertIn("function renderRiskRanking", html)
        self.assertIn("不代表一定缺货", html)


if __name__ == "__main__":
    unittest.main()
