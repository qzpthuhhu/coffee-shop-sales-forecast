import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from second_round_model_experiments import (
    A_FEATURES,
    B_FEATURES,
    BASELINE_METHOD,
    C_FEATURES,
    FEATURE_BUNDLES,
    MINED_FEATURES,
    add_second_round_features,
)
from test_backtest_pipeline import make_panel


class SecondRoundExperimentTest(unittest.TestCase):
    def _source(self, days=70):
        source = make_panel(days=days)
        source["price"] = 1.0
        return source

    def test_all_feature_bundles_are_materialized(self):
        result = add_second_round_features(self._source(), horizon=4)

        self.assertTrue(
            set(A_FEATURES + B_FEATURES + C_FEATURES).issubset(result.columns)
        )
        self.assertEqual(
            FEATURE_BUNDLES["plus_a_b_c"],
            A_FEATURES + B_FEATURES + C_FEATURES,
        )
        self.assertEqual(len(MINED_FEATURES), len(set(MINED_FEATURES)))

    def test_features_ignore_labels_unknown_at_origin(self):
        source = self._source()
        target_date = pd.Timestamp("2023-03-05")
        horizon = 5
        mutated = source.copy()
        mutated.loc[
            mutated["date"] > target_date - pd.Timedelta(days=horizon),
            "qty",
        ] += 100_000

        baseline = add_second_round_features(source, horizon)
        candidate = add_second_round_features(mutated, horizon)
        columns = A_FEATURES + B_FEATURES + C_FEATURES
        baseline_row = baseline[
            baseline["date"] == target_date
        ].sort_values(["store_id", "product_id"])
        candidate_row = candidate[
            candidate["date"] == target_date
        ].sort_values(["store_id", "product_id"])

        np.testing.assert_allclose(
            baseline_row[columns],
            candidate_row[columns],
            equal_nan=True,
        )

    def test_same_weekday_lag_is_target_aligned(self):
        source = self._source()
        target_date = pd.Timestamp("2023-02-20")
        result = add_second_round_features(source, horizon=7)
        target = result[
            (result["date"] == target_date)
            & (result["product_id"] == 101)
        ].iloc[0]
        expected = source[
            (source["date"] == target_date - pd.Timedelta(days=7))
            & (source["product_id"] == 101)
        ]["qty"].iloc[0]

        self.assertEqual(target["qty_same_dow_lag_7"], expected)

    def test_hierarchy_lag_uses_last_observed_day(self):
        source = self._source()
        target_date = pd.Timestamp("2023-02-20")
        horizon = 3
        result = add_second_round_features(source, horizon)
        target = result[result["date"] == target_date].iloc[0]
        cutoff = target_date - pd.Timedelta(days=horizon)
        expected_store_total = source[source["date"] == cutoff]["qty"].sum()
        expected_product_total = source[
            (source["date"] == cutoff)
            & (source["product_id"] == target["product_id"])
        ]["qty"].sum()

        self.assertEqual(target["store_qty_lag_1"], expected_store_total)
        self.assertEqual(
            target["product_all_store_qty_lag_1"],
            expected_product_total,
        )

    def test_tracked_second_round_artifacts_are_complete(self):
        report_dir = (
            Path(__file__).parent
            / "reports"
            / "model_quality"
            / "second_round"
        )
        if not (report_dir / "second_round_run.json").exists():
            self.skipTest("Second-round full experiment has not run yet")
        predictions = pd.read_csv(
            report_dir / "second_round_predictions.csv.gz"
        )
        summary = pd.read_csv(report_dir / "second_round_summary.csv")
        metadata = json.loads(
            (report_dir / "second_round_run.json").read_text(encoding="utf-8")
        )

        self.assertEqual(predictions["method"].nunique(), 6)
        self.assertTrue(
            (predictions.groupby("method").size() == 8400).all()
        )
        self.assertEqual(
            predictions.duplicated(
                [
                    "forecast_origin",
                    "date",
                    "store_id",
                    "product_id",
                    "method",
                ]
            ).sum(),
            0,
        )
        self.assertFalse(predictions["prediction"].isna().any())
        self.assertFalse((predictions["prediction"] < 0).any())
        baseline = summary.set_index("method").loc[BASELINE_METHOD]
        self.assertAlmostEqual(baseline["mae"], 2.442179, places=5)
        self.assertFalse(metadata["production_model_changed"])
        self.assertEqual(len(metadata["training_runs"]), 25)
        self.assertTrue(
            all(
                run["fit_end"] < run["validation_start"]
                for run in metadata["training_runs"]
            )
        )


if __name__ == "__main__":
    unittest.main()
