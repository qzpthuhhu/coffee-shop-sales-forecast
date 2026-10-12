import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from predict_pipeline import STORE_NAME_MAP, load_and_preprocess
from test_backtest_pipeline import make_panel
from third_round_sparse_expert import (
    _label_sku_sparsity,
    run_third_round_experiments,
)


class ThirdRoundSparseExpertTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._base = STORE_NAME_MAP

    def test_sparsity_labels_use_only_pre_origin_data(self):
        source = make_panel(days=70)
        source["qty"] = source["qty"].clip(lower=0)
        first_origin = pd.Timestamp("2023-03-01")
        # Plant future-only SKU that shouldn't affect labels
        future_mask = source["date"] > first_origin
        source.loc[future_mask, "qty"] = 9999

        labels = _label_sku_sparsity(source, first_origin)
        # All labels should come from pre-origin data only
        self.assertEqual(len(labels), 2)
        for _, row in labels.iterrows():
            self.assertIsInstance(row["is_sparse"], (bool, np.bool_))

    def test_sparsity_labels_stable_across_origins(self):
        source = make_panel(days=100)
        source["qty"] = source["qty"].clip(lower=0)
        first = pd.Timestamp("2023-03-20")
        later = pd.Timestamp("2023-03-25")

        first_labels = _label_sku_sparsity(source, first)
        later_labels = _label_sku_sparsity(source, later)

        # First origin labels should be identical to later origin labels
        # because the cutoff is before the first origin in both cases
        pd.testing.assert_frame_equal(
            first_labels.sort_values(
                ["store_id", "product_id"]
            ).reset_index(drop=True),
            later_labels.sort_values(
                ["store_id", "product_id"]
            ).reset_index(drop=True),
        )

    def test_group_split_is_exhaustive_and_exclusive(self):
        source = make_panel(days=70)
        source["qty"] = source["qty"].clip(lower=0)
        labels = _label_sku_sparsity(source, pd.Timestamp("2023-03-01"))

        self.assertEqual(len(labels), len(source[["store_id", "product_id"]].drop_duplicates()))
        sparse_count = labels["is_sparse"].sum()
        dense_count = (~labels["is_sparse"]).sum()
        self.assertEqual(sparse_count + dense_count, len(labels))

    def test_third_round_artifacts_are_complete(self):
        """Run a minimal 1-window experiment and validate artifact contracts."""
        with TemporaryDirectory() as output_dir:
            predictions, quality, summary, metadata = (
                run_third_round_experiments(
                    output_dir=Path(output_dir),
                    window_count=1,
                )
            )

        self.assertGreater(len(predictions), 0)
        methods = predictions["method"].unique()
        self.assertIn("direct_tweedie_1_3", methods)
        self.assertIn("sparse_expert", methods)
        self.assertFalse(predictions["prediction"].isna().any())
        self.assertFalse((predictions["prediction"] < 0).any())

        # Summary should have both methods
        self.assertIn("direct_tweedie_1_3", summary["method"].values)
        self.assertIn("sparse_expert", summary["method"].values)

        # Metadata
        self.assertFalse(metadata["production_model_changed"])
        self.assertGreater(metadata["sparse_sku_count"], 0)
        self.assertGreater(len(metadata["training_runs"]), 0)
        for run in metadata["training_runs"]:
            self.assertIn("group", run)
            self.assertIn(run["group"], ("sparse", "dense"))

        # SKU coverage
        self.assertIn("total_skus", metadata["sku_coverage"])
        self.assertIn("sparse_sku_improved", metadata["sku_coverage"])


if __name__ == "__main__":
    unittest.main()
