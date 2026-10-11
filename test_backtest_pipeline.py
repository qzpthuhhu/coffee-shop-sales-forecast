import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_pipeline import (
    HISTORY_FEATURE_COLS,
    add_origin_safe_baselines,
    build_quality_report,
    recursive_forecast,
    select_backtest_origins,
)
from predict_pipeline import add_history_features


def make_panel(days=49):
    rows = []
    for day_index, date in enumerate(pd.date_range("2023-01-01", periods=days)):
        for product_id in (101, 102):
            rows.append(
                {
                    "date": date,
                    "store_id": 1,
                    "store_name": "Test Store",
                    "product_id": product_id,
                    "category": "Coffee",
                    "product_type": "Brewed",
                    "qty": day_index + product_id - 100,
                }
            )
    return pd.DataFrame(rows)


class LagOneModel:
    def predict(self, features):
        return features["qty_lag_1"].fillna(0).to_numpy()


class ShiftedLagOneModel:
    def __init__(self, shift):
        self.shift = shift

    def predict(self, features):
        return features["qty_lag_1"].fillna(0).to_numpy() + self.shift


class RollingBacktestTest(unittest.TestCase):
    def test_tracked_artifacts_include_empirical_quantile_intervals(self):
        report_dir = Path(__file__).parent / "reports" / "model_quality"
        predictions = pd.read_csv(
            report_dir / "rolling_backtest_predictions.csv"
        )
        quality = pd.read_csv(report_dir / "rolling_backtest_quality.csv")
        metadata = json.loads(
            (report_dir / "rolling_backtest_run.json").read_text(encoding="utf-8")
        )

        self.assertTrue(
            {"pred_p10", "pred_p50", "pred_p90"}.issubset(predictions.columns)
        )
        self.assertFalse((predictions["pred_p10"] > predictions["pred_p50"]).any())
        self.assertFalse((predictions["pred_p50"] > predictions["pred_p90"]).any())
        overall = quality[
            (quality["level"] == "overall")
            & (quality["method"] == "lightgbm")
        ].iloc[0]
        self.assertEqual(overall["interval_status"], "empirical_p10_p90")
        self.assertTrue(np.isfinite(overall["interval_coverage"]))
        self.assertTrue(np.isfinite(overall["interval_mean_width"]))
        self.assertEqual(
            metadata["quantile_levels"],
            {"pred_p10": 0.1, "pred_p50": 0.5, "pred_p90": 0.9},
        )

    def test_origins_are_fixed_non_overlapping_windows(self):
        dates = pd.date_range("2023-01-01", periods=42)
        origins = select_backtest_origins(dates, window_count=3, horizon=7)

        self.assertEqual(
            origins,
            [
                pd.Timestamp("2023-01-22"),
                pd.Timestamp("2023-01-29"),
                pd.Timestamp("2023-02-05"),
            ],
        )

    def test_recursive_forecast_does_not_read_future_labels(self):
        origin = pd.Timestamp("2023-02-12")
        source = add_history_features(make_panel())
        candidate = source.copy()
        candidate.loc[candidate["date"] >= origin, "qty"] += 100_000

        baseline_forecast = recursive_forecast(
            LagOneModel(),
            source,
            ["qty_lag_1"],
            origin,
            horizon=7,
        )
        mutated_forecast = recursive_forecast(
            LagOneModel(),
            candidate,
            ["qty_lag_1"],
            origin,
            horizon=7,
        )

        np.testing.assert_allclose(
            baseline_forecast["pred_lightgbm"],
            mutated_forecast["pred_lightgbm"],
        )

    def test_all_history_features_are_rebuilt_for_each_forecast_day(self):
        origin = pd.Timestamp("2023-02-12")
        source = add_history_features(make_panel())
        forecast = recursive_forecast(
            LagOneModel(),
            source,
            ["qty_lag_1"],
            origin,
            horizon=2,
        )

        self.assertTrue(set(HISTORY_FEATURE_COLS).issubset(forecast.columns))
        day_one = forecast[forecast["date"] == origin].sort_values("product_id")
        day_two = forecast[
            forecast["date"] == origin + pd.Timedelta(days=1)
        ].sort_values("product_id")
        np.testing.assert_allclose(
            day_two["qty_lag_1"],
            day_one["pred_lightgbm"],
        )

    def test_quantiles_are_monotonic_and_ignore_future_labels(self):
        origin = pd.Timestamp("2023-02-12")
        source = add_history_features(make_panel())
        mutated = source.copy()
        mutated.loc[mutated["date"] >= origin, "qty"] += 100_000
        quantile_models = {
            "pred_p10": ShiftedLagOneModel(5),
            "pred_p50": ShiftedLagOneModel(0),
            "pred_p90": ShiftedLagOneModel(-5),
        }

        baseline = recursive_forecast(
            LagOneModel(),
            source,
            ["qty_lag_1"],
            origin,
            horizon=2,
            quantile_models=quantile_models,
        )
        candidate = recursive_forecast(
            LagOneModel(),
            mutated,
            ["qty_lag_1"],
            origin,
            horizon=2,
            quantile_models=quantile_models,
        )

        self.assertTrue((baseline["pred_p10"] <= baseline["pred_p50"]).all())
        self.assertTrue((baseline["pred_p50"] <= baseline["pred_p90"]).all())
        pd.testing.assert_frame_equal(
            baseline[["pred_p10", "pred_p50", "pred_p90"]],
            candidate[["pred_p10", "pred_p50", "pred_p90"]],
        )

    def test_baselines_are_unchanged_when_future_labels_change(self):
        origin = pd.Timestamp("2023-02-12")
        source = make_panel()
        forecast = source[source["date"] >= origin].head(14).copy()

        baseline = add_origin_safe_baselines(forecast, source, origin)
        mutated = source.copy()
        mutated.loc[mutated["date"] >= origin, "qty"] += 100_000
        candidate = add_origin_safe_baselines(forecast, mutated, origin)

        columns = [
            "pred_baseline_yesterday",
            "pred_baseline_last_week",
            "pred_baseline_rolling_28d",
        ]
        pd.testing.assert_frame_equal(baseline[columns], candidate[columns])

    def test_quality_report_marks_model_only_when_it_beats_baselines(self):
        frame = pd.DataFrame(
            {
                "forecast_origin": pd.to_datetime(["2023-01-08"] * 4),
                "date": pd.date_range("2023-01-08", periods=4),
                "store_id": [1] * 4,
                "store_name": ["Test Store"] * 4,
                "product_id": [101] * 4,
                "category": ["Coffee"] * 4,
                "product_type": ["Brewed"] * 4,
                "qty": [10, 12, 14, 16],
                "pred_lightgbm": [10, 12, 14, 16],
                "pred_p10": [8, 10, 12, 14],
                "pred_p50": [10, 12, 14, 16],
                "pred_p90": [12, 14, 16, 18],
                "pred_baseline_yesterday": [8, 8, 8, 8],
                "pred_baseline_last_week": [7, 7, 7, 7],
                "pred_baseline_rolling_28d": [9, 9, 9, 9],
            }
        )

        report, candidate = build_quality_report(frame)
        model_overall = report[
            (report["level"] == "overall")
            & (report["method"] == "lightgbm")
        ].iloc[0]

        self.assertTrue(candidate)
        self.assertTrue(model_overall["candidate_eligible"])
        self.assertEqual(model_overall["wape"], 0)
        self.assertEqual(model_overall["interval_coverage"], 1)
        self.assertEqual(model_overall["interval_mean_width"], 4)
        self.assertEqual(model_overall["interval_median_width"], 4)
        self.assertEqual(model_overall["interval_status"], "empirical_p10_p90")

        baseline_overall = report[
            (report["level"] == "overall")
            & (report["method"] == "baseline_yesterday")
        ].iloc[0]
        self.assertTrue(np.isnan(baseline_overall["interval_coverage"]))
        self.assertEqual(baseline_overall["interval_status"], "not_applicable")


if __name__ == "__main__":
    unittest.main()
