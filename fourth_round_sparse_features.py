"""Fourth-round: inject sparsity labels as interaction features into global Tweedie.

Rather than splitting SKUs into separate models, we add pre-origin sparsity
labels (is_sparse, zero_rate, adi) and their interactions with history
features into a single direct_tweedie_1_3 model.  The model learns when to
use sparse-specific patterns without sacrificing dense cross-SKU learning.

Compared variants (same 5-window backtest):
  baseline      — direct_tweedie_1_3 (round-one features only)
  +sparsity     — baseline + is_sparse + zero_rate + adi + mean_qty
  +interactions — +sparsity + is_sparse × key lag/rolling features
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from backtest_pipeline import (
    BACKTEST_WINDOW_COUNT,
    ENTITY_KEYS,
    REPORT_DIR,
    select_backtest_origins,
)
from first_round_model_experiments import (
    IDENTITY_FEATURES,
    OUTPUT_DIR as FIRST_ROUND_DIR,
    _apply_store_calibration,
    _fit_lightgbm,
    _prediction_rows,
    build_direct_feature_frames,
    build_experiment_quality,
    get_direct_feature_cols,
    split_direct_horizon_frame,
)
from predict_pipeline import (
    HISTORY_LAGS,
    HISTORY_WINDOWS,
    PREDICTION_LOOKAHEAD,
    STORE_NAME_MAP,
    load_and_preprocess,
)

EXPERIMENT_VERSION = "sparse-features-v1"
OUTPUT_DIR = REPORT_DIR / "fourth_round"
BASELINE_METHOD = "direct_tweedie_1_3"
SPARSITY_LABELS_METHOD = "plus_sparsity_labels"
INTERACTIONS_METHOD = "plus_sparsity_interactions"

TWEEPIE_PARAMS = {"objective": "tweedie", "tweedie_variance_power": 1.3}

# History columns to interact with is_sparse
INTERACTION_BASE_COLS = [
    "qty_lag_1",
    "qty_lag_7",
    "qty_lag_28",
    "qty_roll_mean_7d",
    "qty_roll_mean_28d",
    "qty_dow_avg",
    "sku_share_28d",
]


# ---------- sparsity labelling (same as round 3, pre-first-origin) ----------


def _compute_sku_sparsity(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Return per-SKU sparsity metrics from data up to *cutoff*."""
    history = df[df["date"] <= cutoff].copy()
    g = history.groupby(["store_id", "product_id"])["qty"]
    zero_rate = g.apply(lambda x: (x == 0).mean())

    def _adi(series):
        s = series.reset_index(drop=True)
        nz = s[s > 0].index.values
        return np.diff(nz).mean() if len(nz) >= 2 else np.nan

    adi = g.apply(_adi)
    mean_qty = g.mean()
    return pd.DataFrame(
        {
            "zero_rate": zero_rate,
            "adi": adi.fillna(30.0),
            "mean_qty": mean_qty,
        }
    ).reset_index()


# ---------- feature injection ----------


def _attach_sparsity_features(
    frame: pd.DataFrame, sparsity: pd.DataFrame
) -> pd.DataFrame:
    """Merge precomputed sparsity labels and add interaction features."""
    result = frame.merge(
        sparsity[["store_id", "product_id", "zero_rate", "adi", "mean_qty"]],
        on=["store_id", "product_id"],
        how="left",
    )
    result["is_sparse"] = (
        (result["zero_rate"] > 0.5) | (result["adi"] > 2.0)
    ).astype(int)
    # Interaction terms
    for col in INTERACTION_BASE_COLS:
        if col in result.columns:
            result[f"sparse_x_{col}"] = result["is_sparse"] * result[col]
    return result


# ---------- main experiment ----------


def run_fourth_round_experiments(
    output_dir: str | Path = OUTPUT_DIR,
    window_count: int = BACKTEST_WINDOW_COUNT,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. load data ----
    df = load_and_preprocess()
    df["store_name"] = df["store_id"].map(STORE_NAME_MAP)

    origins = select_backtest_origins(
        df["date"].unique(),
        window_count=window_count,
        horizon=PREDICTION_LOOKAHEAD,
    )
    first_origin = min(origins)
    cutoff = first_origin - pd.Timedelta(days=1)

    # ---- 2. pre-origin sparsity labels ----
    sparsity = _compute_sku_sparsity(df, cutoff)
    sparse_mask = sparsity["zero_rate"].gt(0.5) | sparsity["adi"].gt(2.0)
    n_sparse = int(sparse_mask.sum())
    print(f"\nSparse SKUs (pre-{cutoff.date()}): {n_sparse} / {len(sparsity)}")

    # ---- 3. build direct-multi-horizon frames ----
    direct_frames = build_direct_feature_frames(df)
    base_feature_cols = get_direct_feature_cols(direct_frames[1])

    # ---- 4. load first-round baseline ----
    baseline_path = FIRST_ROUND_DIR / "first_round_predictions.csv.gz"
    origin_dates = pd.DatetimeIndex(origins)
    if baseline_path.exists():
        all_baseline = pd.read_csv(
            baseline_path, parse_dates=["date", "forecast_origin"]
        )
        baseline = all_baseline[
            (all_baseline["method"] == BASELINE_METHOD)
            & (all_baseline["forecast_origin"].isin(origin_dates))
        ].copy()
    else:
        baseline = pd.DataFrame()

    # ---- 5. define experiment bundles ----
    def _features_baseline(frame: pd.DataFrame) -> List[str]:
        return base_feature_cols

    def _features_sparsity_labels(frame: pd.DataFrame) -> List[str]:
        extra = ["is_sparse", "zero_rate", "adi", "mean_qty"]
        return list(dict.fromkeys([*base_feature_cols, *extra]))

    def _features_interactions(frame: pd.DataFrame) -> List[str]:
        extra = [
            "is_sparse",
            "zero_rate",
            "adi",
            "mean_qty",
            *[f"sparse_x_{c}" for c in INTERACTION_BASE_COLS],
        ]
        return list(dict.fromkeys([*base_feature_cols, *extra]))

    bundles: Dict[str, Tuple[callable, str]] = {
        "plus_sparsity_labels": (_features_sparsity_labels, SPARSITY_LABELS_METHOD),
        "plus_sparsity_interactions": (_features_interactions, INTERACTIONS_METHOD),
    }

    # ---- 6. per-origin training ----
    all_predictions: List[pd.DataFrame] = (
        [baseline] if len(baseline) else []
    )
    training_runs: List[dict] = []

    for origin in origins:
        print(f"\nFourth-round origin {origin.date()}")

        # Apply sparsity features to all frames (horizon-safe, labels time-invariant)
        augmented_frames = {}
        for h, direct in direct_frames.items():
            augmented_frames[h] = _attach_sparsity_features(direct, sparsity)

        # Assemble slices
        fit_rows, val_rows, cal_rows, test_rows = [], [], [], []
        for horizon_day, direct in augmented_frames.items():
            # Use base features for splitting (no leakage concern since sparsity
            # labels are static pre-origin properties)
            fit, validation, calibration, test = split_direct_horizon_frame(
                direct, base_feature_cols, origin, horizon_day
            )
            fit_rows.append(fit)
            val_rows.append(validation)
            cal_rows.append(calibration)
            test_rows.append(test)

        fit = pd.concat(fit_rows, ignore_index=True)
        validation = pd.concat(val_rows, ignore_index=True)
        calibration = pd.concat(cal_rows, ignore_index=True)
        test = pd.concat(test_rows, ignore_index=True)

        # Train each bundle
        def _train_one(name_and_fn):
            name, (fn, method) = name_and_fn
            cols = fn(test)
            model = _fit_lightgbm(fit, validation, cols, TWEEPIE_PARAMS)

            cal_pred = _prediction_rows(
                calibration,
                np.asarray(model.predict(calibration[cols]), dtype=float),
                method,
                origin,
                calibration["forecast_horizon"].to_numpy(),
            )
            test_pred = _prediction_rows(
                test,
                np.asarray(model.predict(test[cols]), dtype=float),
                method,
                origin,
                test["forecast_horizon"].to_numpy(),
            )
            calibrated, factors = _apply_store_calibration(cal_pred, test_pred)
            training_runs.append(
                {
                    "forecast_origin": origin.date().isoformat(),
                    "method": method,
                    "feature_count": len(cols),
                    "fit_end": fit["date"].max().date().isoformat(),
                    "validation_start": validation["date"].min().date().isoformat(),
                    "validation_end": validation["date"].max().date().isoformat(),
                    "best_iteration": int(model.best_iteration_),
                    "store_bias_factors": factors[
                        ["store_id", "store_bias_factor"]
                    ].to_dict(orient="records"),
                }
            )
            return calibrated

        with ThreadPoolExecutor(max_workers=2) as executor:
            for result in executor.map(_train_one, bundles.items()):
                all_predictions.append(result)

    # ---- 7. assemble predictions ----
    predictions = pd.concat(all_predictions, ignore_index=True)
    prediction_key = ["forecast_origin", "date", *ENTITY_KEYS, "method"]
    if predictions.duplicated(prediction_key).any():
        raise ValueError("Fourth-round predictions must be unique per key")
    if predictions["prediction"].isna().any():
        raise ValueError("Fourth-round predictions must be complete")
    if (predictions["prediction"] < 0).any():
        raise ValueError("Fourth-round predictions must be non-negative")

    # ---- 8. quality report ----
    quality, peak_threshold, low_threshold = build_experiment_quality(predictions)

    # ---- 9. summarize ----
    overall = quality[quality["level"] == "overall"].set_index("method")
    windows = quality[quality["level"] == "window"].pivot(
        index="entity_id", columns="method", values="mae"
    )

    summary_rows = []
    for method in [BASELINE_METHOD, SPARSITY_LABELS_METHOD, INTERACTIONS_METHOD]:
        if method not in overall.index:
            continue
        metrics = overall.loc[method]
        bl = overall.loc[BASELINE_METHOD] if BASELINE_METHOD in overall.index else None
        mae_delta = metrics["mae"] - bl["mae"] if bl is not None else np.nan
        mae_imp = (
            (bl["mae"] - metrics["mae"]) / bl["mae"] * 100
            if bl is not None and bl["mae"]
            else np.nan
        )
        window_wins = 0
        if BASELINE_METHOD in windows.columns and method in windows.columns:
            window_wins = int(
                (windows[method] < windows[BASELINE_METHOD]).sum()
            )
        eligible = bool(
            method != BASELINE_METHOD
            and bl is not None
            and metrics["mae"] < bl["mae"]
            and window_wins >= 3
        )
        summary_rows.append(
            {
                "method": method,
                "mae": metrics["mae"],
                "mae_delta": mae_delta,
                "mae_improvement_pct": mae_imp,
                "wape": metrics["wape"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "mean_error": metrics["mean_error"],
                "change_volatility_ratio": metrics["change_volatility_ratio"],
                "change_direction_accuracy": metrics["change_direction_accuracy"],
                "window_mae_wins": window_wins,
                "eligible": eligible,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mae")

    # ---- 10. sparse subset breakdown ----
    sparsity_label = sparsity[["store_id", "product_id"]].copy()
    sparsity_label["is_sparse"] = (
        (sparsity["zero_rate"] > 0.5) | (sparsity["adi"] > 2.0)
    )
    predictions_s = predictions.merge(
        sparsity_label, on=["store_id", "product_id"], how="left"
    )
    seg_rows = []
    for method in predictions["method"].unique():
        mf = predictions_s[predictions_s["method"] == method]
        for label, mask in [("sparse", mf["is_sparse"]), ("dense", ~mf["is_sparse"])]:
            sub = mf[mask]
            seg_rows.append(
                {
                    "method": method,
                    "segment": label,
                    "mae": (sub["qty"] - sub["prediction"]).abs().mean(),
                    "mbe": (sub["qty"] - sub["prediction"]).mean(),
                    "n": len(sub),
                }
            )
    segment_breakdown = pd.DataFrame(seg_rows)

    # ---- 11. write artifacts ----
    predictions_path = output_dir / "fourth_round_predictions.csv.gz"
    quality_path = output_dir / "fourth_round_quality.csv"
    summary_path = output_dir / "fourth_round_summary.csv"
    segment_path = output_dir / "fourth_round_segment_breakdown.csv"
    metadata_path = output_dir / "fourth_round_run.json"

    predictions.to_csv(
        predictions_path, index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    quality.to_csv(quality_path, index=False)
    summary.to_csv(summary_path, index=False)
    segment_breakdown.to_csv(segment_path, index=False)

    eligible = summary.loc[summary["eligible"], "method"].tolist()
    metadata = {
        "experiment_version": EXPERIMENT_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "production_model_changed": False,
        "description": (
            "Sparsity labels (is_sparse, zero_rate, adi, mean_qty) and their "
            "interactions with key history features injected into a single "
            "global direct_tweedie_1_3 — model learns to gate sparse vs dense "
            "without splitting training data."
        ),
        "origins": [o.date().isoformat() for o in origins],
        "sparse_sku_count": n_sparse,
        "total_sku_count": len(sparsity),
        "interaction_base_cols": INTERACTION_BASE_COLS,
        "eligible_methods": eligible,
        "recommended_candidate": eligible[0] if eligible else None,
        "training_runs": training_runs,
        "segment_breakdown": segment_breakdown.to_dict(orient="records"),
        "artifacts": {
            "predictions": str(predictions_path),
            "quality": str(quality_path),
            "summary": str(summary_path),
            "segment_breakdown": str(segment_path),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print("\nFourth-round aggregate summary:")
    print(summary.to_string(index=False))
    print("\nSegment breakdown:")
    print(segment_breakdown.to_string(index=False))
    return predictions, quality, summary, metadata


if __name__ == "__main__":
    run_fourth_round_experiments()
