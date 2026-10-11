import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from lightgbm import LGBMRegressor

from predict_pipeline import (
    LIGHTGBM_PARAMS,
    add_history_features,
    shap_explain,
    train_lightgbm,
)


def make_panel(days=42):
    rows = []
    for date_index, date in enumerate(pd.date_range('2023-01-01', periods=days)):
        for product_id in (101, 102):
            rows.append({
                'date': date,
                'store_id': 1,
                'product_id': product_id,
                'qty': date_index + (1 if product_id == 101 else 4),
            })
    return pd.DataFrame(rows)


class HistoryFeatureLeakageTest(unittest.TestCase):
    def test_current_and_future_targets_do_not_change_current_features(self):
        source = make_panel()
        target_date = pd.Timestamp('2023-01-29')
        baseline = add_history_features(source)

        mutated = source.copy()
        mutated.loc[mutated['date'] >= target_date, 'qty'] += 10_000
        candidate = add_history_features(mutated)

        history_columns = [
            column for column in baseline.columns
            if column.startswith(('qty_lag_', 'qty_roll_', 'qty_dow_', 'sku_share_'))
        ]
        baseline_row = baseline[
            (baseline['date'] == target_date) & (baseline['product_id'] == 101)
        ][history_columns].reset_index(drop=True)
        candidate_row = candidate[
            (candidate['date'] == target_date) & (candidate['product_id'] == 101)
        ][history_columns].reset_index(drop=True)

        pd.testing.assert_frame_equal(baseline_row, candidate_row)

    def test_historical_sku_share_uses_only_previous_28_days(self):
        featured = add_history_features(make_panel())
        target_date = pd.Timestamp('2023-02-05')
        target = featured[
            (featured['date'] == target_date) & (featured['product_id'] == 101)
        ].iloc[0]

        source = make_panel()
        prior = source[
            (source['date'] < target_date)
            & (source['date'] >= target_date - pd.Timedelta(days=28))
        ].copy()
        totals = prior.groupby('date')['qty'].transform('sum')
        expected = (prior.loc[prior['product_id'] == 101, 'qty'] / totals[
            prior['product_id'] == 101
        ]).mean()

        self.assertAlmostEqual(target['sku_share_28d'], expected)
        self.assertNotIn('sku_share', featured.columns)

    def test_day_of_week_average_excludes_current_target(self):
        source = make_panel()
        featured = add_history_features(source)
        target_date = pd.Timestamp('2023-02-05')

        actual = featured[
            (featured['date'] == target_date) & (featured['product_id'] == 101)
        ]['qty_dow_avg'].iloc[0]
        prior_same_weekday = source[
            (source['date'] < target_date)
            & (source['date'].dt.dayofweek == target_date.dayofweek)
            & (source['product_id'] == 101)
        ]['qty']

        self.assertEqual(actual, prior_same_weekday.mean())

    def test_lag_28_matches_observation_28_days_earlier(self):
        source = make_panel()
        featured = add_history_features(source)
        actual = featured[
            (featured['date'] == pd.Timestamp('2023-02-05'))
            & (featured['product_id'] == 102)
        ]['qty_lag_28'].reset_index(drop=True)
        expected = source[
            (source['date'] == pd.Timestamp('2023-01-08'))
            & (source['product_id'] == 102)
        ]['qty'].reset_index(drop=True)

        assert_series_equal(actual, expected, check_names=False, check_dtype=False)

    def test_first_observation_has_no_history(self):
        featured = add_history_features(make_panel())
        first = featured[
            (featured['date'] == featured['date'].min())
            & (featured['product_id'] == 101)
        ].iloc[0]

        self.assertTrue(np.isnan(first['qty_lag_1']))
        self.assertTrue(np.isnan(first['sku_share_28d']))
        self.assertTrue(np.isnan(first['qty_dow_avg']))

    def test_tree_shap_values_are_nonzero_and_additive(self):
        random = np.random.default_rng(42)
        features = pd.DataFrame({
            'history': random.normal(size=200),
            'weather': random.normal(size=200),
        })
        target = 4 + 2 * features['history'] - features['weather']
        model = LGBMRegressor(
            **{
                **LIGHTGBM_PARAMS,
                'n_estimators': 50,
                'max_depth': 3,
                'num_leaves': 7,
            },
        ).fit(features, target)

        importance, shap_values, explainer = shap_explain(
            model, features.head(40), features.columns.tolist()
        )
        base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        reconstructed = base_value + np.asarray(shap_values).sum(axis=1)

        self.assertGreater(importance['shap_importance'].max(), 0)
        np.testing.assert_allclose(
            reconstructed,
            model.predict(features.head(40)),
            atol=1e-6,
        )

    def test_training_returns_real_lightgbm_model(self):
        random = np.random.default_rng(7)
        features = pd.DataFrame({
            'history': random.normal(size=240),
            'weather': random.normal(size=240),
        })
        target = 8 + 1.5 * features['history'] - 0.5 * features['weather']
        model = train_lightgbm(
            features.iloc[:200],
            target.iloc[:200],
            features.iloc[200:],
            target.iloc[200:],
        )

        self.assertIsInstance(model, LGBMRegressor)
        self.assertEqual(model.booster_.params['objective'], 'regression_l1')
        self.assertGreater(model.best_iteration_, 0)


if __name__ == '__main__':
    unittest.main()
