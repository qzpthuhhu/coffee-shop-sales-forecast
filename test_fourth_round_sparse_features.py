import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fourth_round_sparse_features import (
    INTERACTION_BASE_COLS,
    _attach_sparsity_features,
    _compute_sku_sparsity,
)
from test_backtest_pipeline import make_panel


class FourthRoundSparseFeaturesTest(unittest.TestCase):
    def test_sparsity_metrics_ignore_labels_after_cutoff(self):
        source = make_panel(days=70)
        cutoff = pd.Timestamp("2023-02-12")
        mutated = source.copy()
        mutated.loc[mutated["date"] > cutoff, "qty"] += 100_000

        baseline = _compute_sku_sparsity(source, cutoff)
        candidate = _compute_sku_sparsity(mutated, cutoff)

        pd.testing.assert_frame_equal(baseline, candidate)

    def test_sparse_interactions_match_label_times_history(self):
        source = make_panel(days=14)
        source["qty_lag_1"] = np.arange(len(source), dtype=float)
        sparsity = pd.DataFrame(
            {
                "store_id": [1, 1],
                "product_id": [101, 102],
                "zero_rate": [0.75, 0.10],
                "adi": [3.0, 1.0],
                "mean_qty": [1.0, 8.0],
            }
        )

        result = _attach_sparsity_features(source, sparsity)

        sparse = result["product_id"] == 101
        dense = result["product_id"] == 102
        self.assertTrue((result.loc[sparse, "is_sparse"] == 1).all())
        self.assertTrue((result.loc[dense, "is_sparse"] == 0).all())
        self.assertTrue(
            np.array_equal(
                result.loc[sparse, "sparse_x_qty_lag_1"],
                result.loc[sparse, "qty_lag_1"],
            )
        )
        self.assertTrue(
            (result.loc[dense, "sparse_x_qty_lag_1"] == 0).all()
        )
        self.assertTrue(
            {
                f"sparse_x_{column}"
                for column in INTERACTION_BASE_COLS
                if column in source.columns
            }.issubset(result.columns)
        )

    def test_tracked_fourth_round_artifacts_are_complete(self):
        report_dir = (
            Path(__file__).parent
            / "reports"
            / "model_quality"
            / "fourth_round"
        )
        predictions = pd.read_csv(
            report_dir / "fourth_round_predictions.csv.gz"
        )
        summary = pd.read_csv(report_dir / "fourth_round_summary.csv")
        metadata = json.loads(
            (report_dir / "fourth_round_run.json").read_text(encoding="utf-8")
        )

        self.assertEqual(predictions["method"].nunique(), 3)
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
        candidate = summary.set_index("method").loc[
            "plus_sparsity_labels"
        ]
        baseline = summary.set_index("method").loc[
            "direct_tweedie_1_3"
        ]
        self.assertTrue(candidate["eligible"])
        self.assertLess(candidate["mae"], baseline["mae"])
        self.assertEqual(candidate["window_mae_wins"], 5)
        self.assertFalse(metadata["production_model_changed"])
        self.assertEqual(metadata["sparse_sku_count"], 77)
        self.assertEqual(metadata["total_sku_count"], 240)
        self.assertEqual(len(metadata["training_runs"]), 10)


if __name__ == "__main__":
    unittest.main()
