import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from backtest_pipeline import (
    HISTORY_FEATURE_COLS,
    INTERVAL_TARGET_COVERAGE,
    SHAP_ADDITIVITY_TOLERANCE,
    STORE_BIAS_FACTOR_MAX,
    STORE_BIAS_FACTOR_MIN,
    STORE_BIAS_SHRINKAGE,
    SCENARIO_BASELINE,
    apply_conformal_adjustment,
    apply_kalman_fusion,
    apply_store_bias_calibration,
    add_origin_safe_baselines,
    add_prediction_identity,
    build_quality_report,
    build_scenario_grid,
    calculate_conformal_adjustment,
    calculate_store_bias_factors,
    explain_tree_predictions,
    kalman_forecast,
    recursive_forecast,
    select_backtest_origins,
    select_kalman_fusion_weight,
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


class ScenarioModel:
    def predict(self, features):
        return (
            features["temperature_2m_mean"].to_numpy()
            + features["precipitation_sum"].to_numpy() * 0.1
            + features["is_long_weekend"].to_numpy() * 2
        )


class RollingBacktestTest(unittest.TestCase):
    def test_tracked_artifacts_include_conformalized_quantile_intervals(self):
        report_dir = Path(__file__).parent / "reports" / "model_quality"
        predictions = pd.read_csv(
            report_dir / "rolling_backtest_predictions.csv"
        )
        contributions = pd.read_csv(
            report_dir / "rolling_backtest_shap.csv.gz"
        )
        scenarios = pd.read_csv(
            report_dir / "rolling_backtest_scenarios.csv.gz"
        )
        quality = pd.read_csv(report_dir / "rolling_backtest_quality.csv")
        metadata = json.loads(
            (report_dir / "rolling_backtest_run.json").read_text(encoding="utf-8")
        )

        self.assertEqual(predictions["prediction_id"].nunique(), len(predictions))
        self.assertEqual(predictions["model_run_id"].nunique(), 1)
        self.assertTrue(
            {
                "pred_p10",
                "pred_p50",
                "pred_p90",
                "pred_p10_uncalibrated",
                "pred_p90_uncalibrated",
                "conformal_adjustment",
                "pred_lightgbm_model",
                "pred_lightgbm_raw",
                "pred_baseline_yesterday",
                "pred_baseline_last_week",
                "pred_baseline_rolling_28d",
                "store_bias_factor",
                "store_bias_adjustment",
            }.issubset(predictions.columns)
        )
        self.assertFalse((predictions["pred_p10"] > predictions["pred_p50"]).any())
        self.assertFalse((predictions["pred_p50"] > predictions["pred_p90"]).any())
        overall = quality[
            (quality["level"] == "overall")
            & (quality["method"] == "lightgbm")
        ].iloc[0]
        self.assertEqual(overall["interval_status"], "conformalized_p10_p90")
        self.assertTrue(np.isfinite(overall["interval_coverage"]))
        self.assertTrue(np.isfinite(overall["interval_mean_width"]))
        self.assertEqual(
            metadata["quantile_levels"],
            {"pred_p10": 0.1, "pred_p50": 0.5, "pred_p90": 0.9},
        )
        self.assertEqual(
            metadata["interval_calibration"]["target_coverage"],
            INTERVAL_TARGET_COVERAGE,
        )
        self.assertEqual(
            metadata["interval_calibration"]["method"],
            "split conformalized quantile regression",
        )
        self.assertTrue(
            all(
                "conformal_adjustment" in training_run
                for training_run in metadata["training_runs"]
            )
        )
        feature_count = metadata["feature_count"]
        self.assertEqual(len(contributions), len(predictions) * feature_count)
        self.assertEqual(
            contributions.groupby("prediction_id").size().nunique(),
            1,
        )
        self.assertEqual(
            contributions.groupby("prediction_id").size().iloc[0],
            feature_count,
        )
        self.assertTrue(
            set(contributions["prediction_id"]) == set(predictions["prediction_id"])
        )
        self.assertEqual(
            set(contributions["direction"]),
            {"increase", "decrease", "neutral"},
        )
        contribution_sums = contributions.groupby("prediction_id")[
            "shap_contribution"
        ].sum()
        persisted_sums = predictions.set_index("prediction_id")["shap_sum"]
        np.testing.assert_allclose(
            contribution_sums.sort_index(),
            persisted_sums.sort_index(),
            atol=SHAP_ADDITIVITY_TOLERANCE,
            rtol=0,
        )
        self.assertLessEqual(
            predictions["shap_additivity_error"].abs().max(),
            SHAP_ADDITIVITY_TOLERANCE,
        )
        reconstructed = (
            predictions["base_value"]
            + predictions["shap_sum"]
            + predictions["postprocess_adjustment"]
            + predictions["store_bias_adjustment"]
        )
        np.testing.assert_allclose(
            reconstructed,
            predictions["pred_lightgbm"],
            atol=SHAP_ADDITIVITY_TOLERANCE,
            rtol=0,
        )
        self.assertEqual(
            metadata["artifacts"]["shap_contributions"],
            "reports/model_quality/rolling_backtest_shap.csv.gz",
        )
        self.assertEqual(
            metadata["artifacts"]["scenario_responses"],
            "reports/model_quality/rolling_backtest_scenarios.csv.gz",
        )
        self.assertEqual(
            len(scenarios),
            len(predictions)
            * metadata["scenario_simulation"]["combination_count"],
        )
        self.assertEqual(
            scenarios.groupby("prediction_id")["scenario_index"].nunique().min(),
            27,
        )
        baseline = scenarios[
            (scenarios["temperature_id"] == SCENARIO_BASELINE[0])
            & (scenarios["rain_id"] == SCENARIO_BASELINE[1])
            & (scenarios["holiday_id"] == SCENARIO_BASELINE[2])
        ].set_index("prediction_id")
        np.testing.assert_allclose(
            baseline.loc[predictions["prediction_id"], "simulated_prediction"],
            predictions["pred_lightgbm"],
            atol=1e-9,
            rtol=0,
        )
        self.assertFalse(metadata["scenario_simulation"]["causal"])
        self.assertEqual(
            metadata["store_bias_calibration"]["shrinkage_toward_one"],
            STORE_BIAS_SHRINKAGE,
        )
        self.assertEqual(
            metadata["store_bias_calibration"]["minimum_factor"],
            STORE_BIAS_FACTOR_MIN,
        )
        self.assertEqual(
            metadata["store_bias_calibration"]["maximum_factor"],
            STORE_BIAS_FACTOR_MAX,
        )
        self.assertTrue(
            all(
                len(training_run["store_bias_factors"]) == 3
                for training_run in metadata["training_runs"]
            )
        )
        self.assertEqual(metadata["selected_point_forecast"], "pred_lightgbm")

    def test_prediction_ids_are_stable_and_entity_specific(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-06-24", "2023-06-24"]),
                "store_id": [3, 3],
                "product_id": [1, 2],
            }
        )

        first = add_prediction_identity(
            frame,
            "rolling_backtest_example",
            pd.Timestamp("2023-06-24"),
        )
        second = add_prediction_identity(
            frame,
            "rolling_backtest_example",
            pd.Timestamp("2023-06-24"),
        )

        self.assertEqual(first["prediction_id"].nunique(), 2)
        pd.testing.assert_series_equal(
            first["prediction_id"],
            second["prediction_id"],
        )

    def test_scenario_grid_scores_all_controlled_combinations(self):
        frame = pd.DataFrame(
            {
                "prediction_id": ["prediction_1"],
                "pred_lightgbm": [17.0],
                "store_bias_factor": [1.0],
                "temperature_2m_max": [22.0],
                "temperature_2m_min": [12.0],
                "temperature_2m_mean": [17.0],
                "precipitation_sum": [0.0],
                "is_rain": [0],
                "is_snow": [0],
                "is_clear": [1],
                "is_holiday_detailed": [0],
                "is_pre_holiday_1d": [0],
                "is_pre_holiday_3d": [0],
                "is_post_holiday_1d": [0],
                "is_long_weekend": [0],
                "holiday_impact_code": [0],
                "is_hot": [0],
                "is_cold": [0],
            }
        )
        feature_cols = [
            "temperature_2m_mean",
            "precipitation_sum",
            "is_long_weekend",
        ]

        scenarios = build_scenario_grid(
            ScenarioModel(),
            frame,
            feature_cols,
        )

        self.assertEqual(len(scenarios), 27)
        self.assertEqual(scenarios["scenario_index"].nunique(), 27)
        baseline = scenarios[
            (scenarios["temperature_id"] == "observed")
            & (scenarios["rain_id"] == "observed")
            & (scenarios["holiday_id"] == "observed")
        ].iloc[0]
        self.assertAlmostEqual(baseline["simulated_prediction"], 17.0)
        changed = scenarios[
            (scenarios["temperature_id"] == "warmer_5c")
            & (scenarios["rain_id"] == "heavy_20mm")
            & (scenarios["holiday_id"] == "long_weekend")
        ].iloc[0]
        self.assertAlmostEqual(changed["simulated_prediction"], 26.0)

    def test_tree_shap_preserves_signed_row_level_additivity(self):
        features = pd.DataFrame(
            {
                "lag": np.arange(40, dtype=float),
                "rain": np.tile([0.0, 1.0], 20),
            }
        )
        target = 2.5 + features["lag"] * 0.4 - features["rain"] * 1.5
        model = LGBMRegressor(
            n_estimators=30,
            num_leaves=7,
            min_child_samples=2,
            random_state=42,
            verbosity=-1,
        ).fit(features, target)
        sample = features.iloc[-6:].reset_index(drop=True)
        prediction_ids = [f"prediction_{index}" for index in range(len(sample))]

        summary, contributions = explain_tree_predictions(
            model,
            sample,
            prediction_ids,
            list(sample.columns),
        )

        self.assertEqual(len(summary), len(sample))
        self.assertEqual(len(contributions), len(sample) * len(sample.columns))
        self.assertGreater(contributions["shap_contribution"].max(), 0)
        self.assertLess(contributions["shap_contribution"].min(), 0)
        self.assertTrue(
            {"increase", "decrease"}.issubset(set(contributions["direction"]))
        )
        np.testing.assert_allclose(
            summary["base_value"] + summary["shap_sum"],
            model.predict(sample),
            atol=SHAP_ADDITIVITY_TOLERANCE,
            rtol=0,
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

    def test_store_bias_factors_are_shrunk_and_clipped(self):
        calibration = pd.DataFrame(
            {
                "store_id": [1, 1, 2, 2],
                "qty": [60.0, 60.0, 100.0, 100.0],
                "pred_lightgbm": [50.0, 50.0, 50.0, 50.0],
            }
        )

        factors = calculate_store_bias_factors(calibration)

        factor_by_store = factors.set_index("store_id")
        self.assertAlmostEqual(
            factor_by_store.loc[1, "store_bias_ratio_raw"],
            1.2,
        )
        self.assertAlmostEqual(
            factor_by_store.loc[1, "store_bias_factor"],
            1.1,
        )
        self.assertAlmostEqual(
            factor_by_store.loc[2, "store_bias_factor"],
            STORE_BIAS_FACTOR_MAX,
        )

    def test_store_bias_calibration_scales_outputs_and_defaults_unseen_store(self):
        forecast = pd.DataFrame(
            {
                "store_id": [1, 2],
                "pred_lightgbm": [10.0, 20.0],
                "pred_p10": [8.0, 18.0],
                "pred_p50": [10.0, 20.0],
                "pred_p90": [12.0, 22.0],
            }
        )
        factors = pd.DataFrame(
            {
                "store_id": [1],
                "store_bias_factor": [1.1],
            }
        )

        calibrated = apply_store_bias_calibration(forecast, factors)

        np.testing.assert_allclose(calibrated["store_bias_factor"], [1.1, 1.0])
        np.testing.assert_allclose(calibrated["pred_lightgbm_model"], [10, 20])
        np.testing.assert_allclose(calibrated["pred_lightgbm"], [11, 20])
        np.testing.assert_allclose(
            calibrated["store_bias_adjustment"],
            [1, 0],
        )
        np.testing.assert_allclose(calibrated["pred_p10"], [8.8, 18])
        np.testing.assert_allclose(calibrated["pred_p50"], [11, 20])
        np.testing.assert_allclose(calibrated["pred_p90"], [13.2, 22])
        self.assertTrue(
            (calibrated["pred_p10"] <= calibrated["pred_p50"]).all()
        )
        self.assertTrue(
            (calibrated["pred_p50"] <= calibrated["pred_p90"]).all()
        )

    def test_store_calibrated_recursive_forecast_ignores_target_labels(self):
        origin = pd.Timestamp("2023-02-12")
        source = add_history_features(make_panel())
        mutated = source.copy()
        mutated.loc[mutated["date"] >= origin, "qty"] += 100_000
        factors = pd.DataFrame(
            {"store_id": [1], "store_bias_factor": [1.1]}
        )

        baseline = recursive_forecast(
            LagOneModel(),
            source,
            ["qty_lag_1"],
            origin,
            horizon=2,
            store_bias_factors=factors,
        )
        candidate = recursive_forecast(
            LagOneModel(),
            mutated,
            ["qty_lag_1"],
            origin,
            horizon=2,
            store_bias_factors=factors,
        )

        np.testing.assert_allclose(
            baseline["pred_lightgbm"],
            candidate["pred_lightgbm"],
        )

    def test_conformal_adjustment_uses_finite_sample_quantile(self):
        calibration = pd.DataFrame(
            {
                "qty": [7, 9, 12, 17],
                "pred_p10": [8, 8, 8, 8],
                "pred_p90": [12, 12, 12, 12],
            }
        )

        adjustment = calculate_conformal_adjustment(
            calibration,
            target_coverage=0.75,
        )

        self.assertEqual(adjustment, 5)

    def test_conformal_adjustment_preserves_raw_bounds_and_order(self):
        forecast = pd.DataFrame(
            {
                "pred_p10": [1.0, 5.0],
                "pred_p50": [2.0, 6.0],
                "pred_p90": [3.0, 7.0],
            }
        )

        calibrated = apply_conformal_adjustment(forecast, adjustment=2.0)

        np.testing.assert_allclose(
            calibrated["pred_p10_uncalibrated"],
            forecast["pred_p10"],
        )
        np.testing.assert_allclose(
            calibrated["pred_p90_uncalibrated"],
            forecast["pred_p90"],
        )
        np.testing.assert_allclose(calibrated["pred_p10"], [0.0, 3.0])
        np.testing.assert_allclose(calibrated["pred_p90"], [5.0, 9.0])
        self.assertTrue((calibrated["pred_p10"] <= calibrated["pred_p50"]).all())
        self.assertTrue((calibrated["pred_p50"] <= calibrated["pred_p90"]).all())

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

    def test_kalman_forecast_ignores_future_labels(self):
        origin = pd.Timestamp("2023-02-12")
        source = make_panel()
        mutated = source.copy()
        mutated.loc[mutated["date"] >= origin, "qty"] += 100_000

        baseline = kalman_forecast(source, origin, horizon=7)
        candidate = kalman_forecast(mutated, origin, horizon=7)

        pd.testing.assert_frame_equal(baseline, candidate)

    def test_kalman_forecast_responds_to_recent_trend(self):
        source = make_panel(days=56)
        origin = pd.Timestamp("2023-02-19")
        source["qty"] = (
            source["date"] - source["date"].min()
        ).dt.days.astype(float)

        forecast = kalman_forecast(source, origin, horizon=3)
        series = forecast[
            (forecast["store_id"] == 1)
            & (forecast["product_id"] == 101)
        ].sort_values("date")

        self.assertGreater(series["pred_kalman"].iloc[0], 45)
        self.assertGreater(
            series["pred_kalman"].iloc[-1],
            series["pred_kalman"].iloc[0],
        )

    def test_kalman_fusion_weight_uses_calibration_errors(self):
        calibration = pd.DataFrame(
            {
                "qty": [10.0, 20.0, 30.0],
                "pred_lightgbm": [8.0, 18.0, 28.0],
                "pred_kalman": [10.0, 20.0, 30.0],
            }
        )

        weight, candidates = select_kalman_fusion_weight(
            calibration,
            weights=(0.0, 0.5, 1.0),
        )

        self.assertEqual(weight, 1.0)
        self.assertEqual(len(candidates), 3)

    def test_apply_kalman_fusion_preserves_component_predictions(self):
        forecast = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01"]),
                "store_id": [1],
                "product_id": [101],
                "pred_lightgbm": [10.0],
            }
        )
        kalman = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01"]),
                "store_id": [1],
                "product_id": [101],
                "pred_kalman": [14.0],
            }
        )

        blended = apply_kalman_fusion(forecast, kalman, weight=0.25)

        self.assertEqual(blended["pred_lightgbm"].iloc[0], 10.0)
        self.assertEqual(blended["pred_kalman"].iloc[0], 14.0)
        self.assertEqual(blended["pred_lightgbm_kalman"].iloc[0], 11.0)

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
                "pred_kalman": [10, 12, 14, 16],
                "pred_lightgbm_kalman": [10, 12, 14, 16],
                "pred_p10": [8, 10, 12, 14],
                "pred_p50": [10, 12, 14, 16],
                "pred_p90": [12, 14, 16, 18],
                "pred_baseline_yesterday": [8, 8, 8, 8],
                "pred_baseline_last_week": [7, 7, 7, 7],
                "pred_baseline_rolling_28d": [9, 9, 9, 9],
            }
        )

        report, candidate, fusion_candidate, fusion_window_wins = (
            build_quality_report(frame)
        )
        model_overall = report[
            (report["level"] == "overall")
            & (report["method"] == "lightgbm")
        ].iloc[0]

        self.assertTrue(candidate)
        self.assertFalse(fusion_candidate)
        self.assertEqual(fusion_window_wins, 0)
        self.assertTrue(model_overall["candidate_eligible"])
        self.assertEqual(model_overall["wape"], 0)
        self.assertEqual(model_overall["interval_coverage"], 1)
        self.assertEqual(model_overall["interval_mean_width"], 4)
        self.assertEqual(model_overall["interval_median_width"], 4)
        self.assertEqual(model_overall["interval_status"], "conformalized_p10_p90")

        baseline_overall = report[
            (report["level"] == "overall")
            & (report["method"] == "baseline_yesterday")
        ].iloc[0]
        self.assertTrue(np.isnan(baseline_overall["interval_coverage"]))
        self.assertEqual(baseline_overall["interval_status"], "not_applicable")


if __name__ == "__main__":
    unittest.main()
