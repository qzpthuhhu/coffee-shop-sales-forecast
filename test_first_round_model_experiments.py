import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from first_round_model_experiments import (
    IDENTITY_FEATURES,
    add_direct_history_features,
    build_direct_feature_frames,
    get_direct_feature_cols,
    split_direct_horizon_frame,
)
from predict_pipeline import STORE_NAME_MAP, add_history_features
from test_backtest_pipeline import make_panel


class FirstRoundExperimentTest(unittest.TestCase):
    def test_store_names_match_source_data(self):
        source = pd.read_excel(
            Path(__file__).parent / "Coffee Shop Sales.xlsx",
            usecols=["store_id", "store_location"],
        ).drop_duplicates()

        self.assertEqual(
            source.set_index("store_id")["store_location"].to_dict(),
            STORE_NAME_MAP,
        )

    def test_tracked_first_round_artifacts_are_complete(self):
        report_dir = (
            Path(__file__).parent
            / "reports"
            / "model_quality"
            / "first_round"
        )
        predictions = pd.read_csv(
            report_dir / "first_round_predictions.csv.gz"
        )
        summary = pd.read_csv(report_dir / "first_round_summary.csv")
        metadata = json.loads(
            (report_dir / "first_round_run.json").read_text(encoding="utf-8")
        )

        self.assertEqual(predictions["method"].nunique(), 9)
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
        candidate = summary.set_index("method").loc["direct_tweedie_1_3"]
        baseline = summary.set_index("method").loc["recursive_l1"]
        self.assertTrue(candidate["eligible"])
        self.assertLess(candidate["mae"], baseline["mae"])
        self.assertLess(candidate["wape"], baseline["wape"])
        self.assertEqual(candidate["window_mae_wins"], 5)
        self.assertFalse(metadata["production_model_changed"])
        self.assertEqual(len(metadata["training_runs"]), 40)
        self.assertTrue(
            all(
                run["fit_end"] < run["validation_start"]
                for run in metadata["training_runs"]
            )
        )

    def test_horizon_one_matches_recursive_history_features(self):
        source = make_panel(days=49)

        recursive = add_history_features(source.copy())
        direct = add_direct_history_features(source.copy(), horizon=1)
        columns = [
            "qty_lag_1",
            "qty_lag_7",
            "qty_roll_mean_7d",
            "qty_roll_std_28d",
            "qty_dow_avg",
            "sku_share_28d",
        ]

        pd.testing.assert_frame_equal(
            recursive[columns],
            direct[columns],
        )

    def test_direct_features_ignore_unknown_horizon_labels(self):
        source = make_panel(days=49)
        target_date = pd.Timestamp("2023-02-12")
        horizon = 3
        mutated = source.copy()
        mutated.loc[
            mutated["date"] > target_date - pd.Timedelta(days=horizon),
            "qty",
        ] += 100_000

        baseline = add_direct_history_features(source, horizon)
        candidate = add_direct_history_features(mutated, horizon)
        columns = [
            "qty_lag_1",
            "qty_lag_2",
            "qty_lag_7",
            "qty_roll_mean_7d",
            "qty_roll_std_28d",
            "qty_dow_avg",
            "sku_share_28d",
        ]
        baseline_row = baseline[baseline["date"] == target_date].sort_values(
            ["store_id", "product_id"]
        )
        candidate_row = candidate[candidate["date"] == target_date].sort_values(
            ["store_id", "product_id"]
        )

        np.testing.assert_allclose(
            baseline_row[columns],
            candidate_row[columns],
        )

    def test_fixed_origin_split_keeps_fit_targets_before_validation_origin(self):
        source = make_panel(days=70)
        source["price"] = 1.0
        source["category_code"] = 0
        source["product_type_code"] = 0
        source["store_id_feature"] = source["store_id"]
        source["product_id_feature"] = source["product_id"]
        direct = add_direct_history_features(source, horizon=4)
        feature_cols = ["qty_lag_1", *IDENTITY_FEATURES]
        origin = pd.Timestamp("2023-03-05")

        fit, validation, calibration, test = split_direct_horizon_frame(
            direct,
            feature_cols,
            origin,
            horizon_day=4,
        )

        self.assertLess(fit["date"].max(), origin - pd.Timedelta(days=14))
        self.assertEqual(
            validation["date"].iloc[0],
            origin - pd.Timedelta(days=11),
        )
        self.assertEqual(
            calibration["date"].iloc[0],
            origin - pd.Timedelta(days=4),
        )
        self.assertEqual(
            test["date"].iloc[0],
            origin + pd.Timedelta(days=3),
        )

    def test_direct_frames_include_explicit_identity_features(self):
        source = make_panel(days=35)
        source["price"] = 1.0
        source["category_code"] = 0
        source["product_type_code"] = 0
        frames = build_direct_feature_frames(source, horizon_days=2)
        feature_cols = get_direct_feature_cols(frames[1])

        self.assertEqual(set(frames), {1, 2})
        self.assertTrue(set(IDENTITY_FEATURES).issubset(feature_cols))


if __name__ == "__main__":
    unittest.main()
