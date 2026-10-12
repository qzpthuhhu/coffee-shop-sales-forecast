import unittest

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from production_sparse import (
    _build_quality,
    _build_run_id,
    _shap_contributions,
)


class ProductionSparseTest(unittest.TestCase):
    def test_tweedie_shap_is_additive_on_response_scale(self):
        features = pd.DataFrame(
            {
                "history": np.arange(1, 41, dtype=float),
                "weather": np.tile([0.0, 1.0], 20),
            }
        )
        target = 1.0 + features["history"] * 0.4 + features["weather"] * 2
        model = LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.3,
            n_estimators=30,
            num_leaves=7,
            min_child_samples=2,
            verbosity=-1,
            random_state=42,
        ).fit(features, target)
        prediction_ids = pd.Series([f"p{i}" for i in range(len(features))])

        contributions = _shap_contributions(
            model,
            features,
            list(features.columns),
            prediction_ids,
        )
        summary = contributions.groupby("prediction_id").agg(
            base_value=("base_value", "first"),
            shap_sum=("shap_sum", "first"),
            additivity_error=("shap_additivity_error", "first"),
        )
        expected = pd.Series(
            model.predict(features),
            index=prediction_ids,
        ).sort_index()

        np.testing.assert_allclose(
            (summary["base_value"] + summary["shap_sum"]).sort_index(),
            expected,
            atol=1e-6,
            rtol=0,
        )
        self.assertLessEqual(summary["additivity_error"].abs().max(), 1e-6)

    def test_quality_report_compares_model_to_origin_safe_baselines(self):
        frame = pd.DataFrame(
            {
                "forecast_origin": pd.to_datetime(["2023-01-01"] * 4),
                "date": pd.date_range("2023-01-01", periods=4),
                "store_id": [1] * 4,
                "store_name": ["Store"] * 4,
                "product_id": [101] * 4,
                "category": ["Coffee"] * 4,
                "product_type": ["Latte"] * 4,
                "qty": [10.0, 12.0, 14.0, 16.0],
                "pred_lightgbm": [10.0, 12.0, 14.0, 16.0],
                "pred_p10": [8.0, 10.0, 12.0, 14.0],
                "pred_p90": [12.0, 14.0, 16.0, 18.0],
                "pred_baseline_yesterday": [8.0] * 4,
                "pred_baseline_last_week": [7.0] * 4,
                "pred_baseline_rolling_28d": [9.0] * 4,
            }
        )

        report, candidate = _build_quality(frame)
        overall = report[report["level"] == "overall"]

        self.assertTrue(candidate)
        self.assertEqual(
            set(overall["method"]),
            {
                "lightgbm",
                "baseline_yesterday",
                "baseline_last_week",
                "baseline_rolling_28d",
            },
        )
        self.assertTrue(
            overall.loc[
                overall["method"] == "lightgbm",
                "candidate_eligible",
            ].iloc[0]
        )

    def test_run_id_is_stable_for_same_contract(self):
        origins = [
            pd.Timestamp("2023-05-27"),
            pd.Timestamp("2023-06-03"),
        ]
        first = _build_run_id({"sales": "abc"}, origins, ["a", "b"])
        second = _build_run_id({"sales": "abc"}, origins, ["a", "b"])

        self.assertEqual(first, second)
        self.assertNotEqual(
            first,
            _build_run_id({"sales": "changed"}, origins, ["a", "b"]),
        )


if __name__ == "__main__":
    unittest.main()
