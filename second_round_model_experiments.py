"""Second-round leakage-safe feature mining experiments.

The production model remains unchanged. This module keeps the first-round
direct Tweedie model as the baseline and evaluates incremental feature bundles
on the same five rolling origins.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_pipeline import (
    BACKTEST_WINDOW_COUNT,
    ENTITY_KEYS,
    REPORT_DIR,
    calculate_metrics,
    select_backtest_origins,
)
from first_round_model_experiments import (
    _apply_store_calibration,
    _fit_lightgbm,
    _prediction_rows,
    build_direct_feature_frames,
    build_experiment_quality,
    get_direct_feature_cols,
    split_direct_horizon_frame,
)
from predict_pipeline import (
    PREDICTION_LOOKAHEAD,
    STORE_NAME_MAP,
    load_and_preprocess,
)


EXPERIMENT_VERSION = "direct-feature-mining-v2"
OUTPUT_DIR = REPORT_DIR / "second_round"
BASELINE_METHOD = "baseline_direct_tweedie_1_3"
TWEEDIE_PARAMETERS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.3,
}

A_FEATURES = [
    "qty_ewm_3d",
    "qty_ewm_7d",
    "qty_ewm_14d",
    "qty_ewm_28d",
    "qty_slope_7d",
    "qty_momentum_7_28d",
    "qty_rate_7_28d",
    "qty_same_dow_lag_7",
    "qty_same_dow_lag_14",
    "qty_same_dow_lag_21",
    "qty_same_dow_lag_28",
    "qty_same_dow_median_4",
    "qty_same_dow_max_4",
]
B_FEATURES = [
    "zero_streak",
    "days_since_sale",
    "nonzero_rate_7d",
    "nonzero_rate_28d",
    "adi_28d",
    "cv2_28d",
]
C_FEATURES = [
    "store_qty_lag_1",
    "store_qty_mean_7d",
    "store_qty_mean_28d",
    "store_qty_momentum_7_28d",
    "store_category_qty_lag_1",
    "store_category_qty_mean_7d",
    "store_category_qty_mean_28d",
    "store_category_qty_momentum_7_28d",
    "product_all_store_qty_lag_1",
    "product_all_store_qty_mean_7d",
    "product_all_store_qty_mean_28d",
    "product_all_store_qty_momentum_7_28d",
    "sku_share_7d",
    "sku_share_trend_7_28d",
]
FEATURE_BUNDLES = {
    "plus_a": A_FEATURES,
    "plus_b": B_FEATURES,
    "plus_c": C_FEATURES,
    "plus_a_b": [*A_FEATURES, *B_FEATURES],
    "plus_a_b_c": [*A_FEATURES, *B_FEATURES, *C_FEATURES],
}
MINED_FEATURES = [*A_FEATURES, *B_FEATURES, *C_FEATURES]


def _grouped_shifted(result, values):
    return values.groupby([result[key] for key in ENTITY_KEYS])


def _zero_streak(values):
    streak = 0
    output = np.zeros(len(values), dtype=float)
    for index, value in enumerate(values):
        streak = streak + 1 if value == 0 else 0
        output[index] = streak
    return pd.Series(output, index=values.index)


def _days_since_sale(values):
    last_sale = None
    output = np.zeros(len(values), dtype=float)
    for position, (index, value) in enumerate(values.items()):
        if value > 0:
            last_sale = position
            output[position] = 0
        else:
            output[position] = (
                position - last_sale if last_sale is not None else position + 1
            )
    return pd.Series(output, index=values.index)


def add_recent_dynamics_features(frame, horizon):
    """Add bundle A using observations available at the forecast origin."""
    result = frame.sort_values([*ENTITY_KEYS, "date"]).reset_index(drop=True)
    qty_group = result.groupby(ENTITY_KEYS, sort=False)["qty"]
    shifted = qty_group.shift(horizon)
    grouped = _grouped_shifted(result, shifted)

    for span in (3, 7, 14, 28):
        result[f"qty_ewm_{span}d"] = grouped.transform(
            lambda values: values.ewm(
                span=span,
                adjust=False,
                min_periods=1,
            ).mean()
        )

    recent_mean = grouped.transform(
        lambda values: values.rolling(7, min_periods=1).mean()
    )
    long_mean = grouped.transform(
        lambda values: values.rolling(28, min_periods=1).mean()
    )
    result["qty_slope_7d"] = (
        shifted - grouped.shift(6)
    ) / 6.0
    result["qty_momentum_7_28d"] = recent_mean - long_mean
    result["qty_rate_7_28d"] = recent_mean / (long_mean + 1.0)

    same_dow_columns = []
    for lag in (7, 14, 21, 28):
        column = f"qty_same_dow_lag_{lag}"
        result[column] = qty_group.shift(lag)
        same_dow_columns.append(column)
    result["qty_same_dow_median_4"] = result[same_dow_columns].median(axis=1)
    result["qty_same_dow_max_4"] = result[same_dow_columns].max(axis=1)
    return result


def add_sparse_demand_features(frame, horizon):
    """Add bundle B for intermittency, recency, and demand regularity."""
    result = frame.sort_values([*ENTITY_KEYS, "date"]).reset_index(drop=True)
    qty_group = result.groupby(ENTITY_KEYS, sort=False)["qty"]
    shifted = qty_group.shift(horizon)
    grouped = _grouped_shifted(result, shifted)
    positive = (shifted > 0).astype(float)
    grouped_positive = _grouped_shifted(result, positive)

    streak = qty_group.transform(_zero_streak)
    recency = qty_group.transform(_days_since_sale)
    result["zero_streak"] = streak.groupby(
        [result[key] for key in ENTITY_KEYS]
    ).shift(horizon)
    result["days_since_sale"] = recency.groupby(
        [result[key] for key in ENTITY_KEYS]
    ).shift(horizon)
    result["nonzero_rate_7d"] = grouped_positive.transform(
        lambda values: values.rolling(7, min_periods=1).mean()
    )
    result["nonzero_rate_28d"] = grouped_positive.transform(
        lambda values: values.rolling(28, min_periods=1).mean()
    )
    nonzero_count = grouped_positive.transform(
        lambda values: values.rolling(28, min_periods=1).sum()
    )
    observed_count = grouped_positive.transform(
        lambda values: values.rolling(28, min_periods=1).count()
    )
    result["adi_28d"] = observed_count / nonzero_count.replace(0, np.nan)
    rolling_mean = grouped.transform(
        lambda values: values.rolling(28, min_periods=1).mean()
    )
    rolling_std = grouped.transform(
        lambda values: values.rolling(28, min_periods=2).std()
    ).fillna(0)
    result["cv2_28d"] = (
        rolling_std / rolling_mean.replace(0, np.nan)
    ).pow(2)
    result["adi_28d"] = result["adi_28d"].fillna(29.0).clip(upper=29.0)
    result["cv2_28d"] = result["cv2_28d"].fillna(0).clip(upper=100.0)
    return result


def _aggregate_history(frame, keys, prefix, horizon):
    aggregate = (
        frame.groupby([*keys, "date"], as_index=False)["qty"]
        .sum()
        .sort_values([*keys, "date"])
        .reset_index(drop=True)
    )
    grouped_qty = aggregate.groupby(keys, sort=False)["qty"]
    shifted = grouped_qty.shift(horizon)
    grouped_shifted = shifted.groupby([aggregate[key] for key in keys])
    aggregate[f"{prefix}_lag_1"] = shifted
    aggregate[f"{prefix}_mean_7d"] = grouped_shifted.transform(
        lambda values: values.rolling(7, min_periods=1).mean()
    )
    aggregate[f"{prefix}_mean_28d"] = grouped_shifted.transform(
        lambda values: values.rolling(28, min_periods=1).mean()
    )
    aggregate[f"{prefix}_momentum_7_28d"] = (
        aggregate[f"{prefix}_mean_7d"]
        - aggregate[f"{prefix}_mean_28d"]
    )
    columns = [
        *keys,
        "date",
        f"{prefix}_lag_1",
        f"{prefix}_mean_7d",
        f"{prefix}_mean_28d",
        f"{prefix}_momentum_7_28d",
    ]
    return aggregate[columns]


def add_hierarchy_features(frame, horizon):
    """Add bundle C from origin-safe store, category, and product aggregates."""
    result = frame.sort_values([*ENTITY_KEYS, "date"]).reset_index(drop=True)
    aggregates = [
        _aggregate_history(result, ["store_id"], "store_qty", horizon),
        _aggregate_history(
            result,
            ["store_id", "category"],
            "store_category_qty",
            horizon,
        ),
        _aggregate_history(
            result,
            ["product_id"],
            "product_all_store_qty",
            horizon,
        ),
    ]
    for aggregate in aggregates:
        merge_keys = [
            column
            for column in ("store_id", "product_id", "category", "date")
            if column in aggregate.columns
        ]
        result = result.merge(
            aggregate,
            on=merge_keys,
            how="left",
            validate="many_to_one",
        )

    store_day_total = result.groupby(["store_id", "date"])["qty"].transform(
        "sum"
    )
    observed_share = (
        result["qty"] / store_day_total.replace(0, np.nan)
    ).fillna(0)
    shifted_share = observed_share.groupby(
        [result[key] for key in ENTITY_KEYS]
    ).shift(horizon)
    grouped_share = _grouped_shifted(result, shifted_share)
    result["sku_share_7d"] = grouped_share.transform(
        lambda values: values.rolling(7, min_periods=1).mean()
    )
    share_28d = grouped_share.transform(
        lambda values: values.rolling(28, min_periods=1).mean()
    )
    result["sku_share_trend_7_28d"] = (
        result["sku_share_7d"] - share_28d
    )
    return result


def add_second_round_features(frame, horizon):
    """Materialize all three feature bundles for one direct horizon."""
    result = add_recent_dynamics_features(frame, horizon)
    result = add_sparse_demand_features(result, horizon)
    return add_hierarchy_features(result, horizon)


def build_second_round_feature_frames(df, horizon_days=PREDICTION_LOOKAHEAD):
    """Build direct frames with all second-round feature candidates."""
    direct_frames = build_direct_feature_frames(df, horizon_days=horizon_days)
    return {
        horizon: add_second_round_features(frame, horizon)
        for horizon, frame in direct_frames.items()
    }


def _load_first_round_baseline(origins):
    path = REPORT_DIR / "first_round" / "first_round_predictions.csv.gz"
    baseline = pd.read_csv(path, parse_dates=["date", "forecast_origin"])
    baseline = baseline[
        (baseline["method"] == "direct_tweedie_1_3")
        & baseline["forecast_origin"].isin(origins)
    ].copy()
    baseline["method"] = BASELINE_METHOD
    baseline["store_name"] = baseline["store_id"].map(STORE_NAME_MAP)
    return baseline


def _coverage_rows(predictions):
    levels = {
        "window": ["forecast_origin"],
        "horizon": ["horizon_day"],
        "store": ["store_id"],
        "category": ["category"],
        "product_type": ["product_type"],
        "sku": ["product_id"],
    }
    rows = []
    baseline = predictions[predictions["method"] == BASELINE_METHOD]
    for method in FEATURE_BUNDLES:
        candidate = predictions[predictions["method"] == method]
        for level, keys in levels.items():
            baseline_groups = baseline.groupby(keys, dropna=False)
            candidate_groups = candidate.groupby(keys, dropna=False)
            wins = 0
            total = 0
            for entity, baseline_frame in baseline_groups:
                try:
                    candidate_frame = candidate_groups.get_group(entity)
                except KeyError:
                    continue
                baseline_mae = calculate_metrics(
                    baseline_frame["qty"],
                    baseline_frame["prediction"],
                )["mae"]
                candidate_mae = calculate_metrics(
                    candidate_frame["qty"],
                    candidate_frame["prediction"],
                )["mae"]
                wins += int(candidate_mae < baseline_mae)
                total += 1
            rows.append(
                {
                    "method": method,
                    "level": level,
                    "mae_wins": wins,
                    "entity_count": total,
                    "win_rate": wins / total if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_second_round(quality, coverage):
    """Screen mined feature bundles against the first-round candidate."""
    overall = quality[quality["level"] == "overall"].set_index("method")
    peak = quality[
        (quality["level"] == "demand_segment")
        & (quality["entity_id"] == "peak_p90")
    ].set_index("method")
    baseline = overall.loc[BASELINE_METHOD]
    baseline_peak = peak.loc[BASELINE_METHOD]
    window_coverage = coverage[coverage["level"] == "window"].set_index(
        "method"
    )
    rows = []
    for method, metrics in overall.iterrows():
        window_wins = (
            0
            if method == BASELINE_METHOD
            else int(window_coverage.loc[method, "mae_wins"])
        )
        rows.append(
            {
                "method": method,
                "mae": metrics["mae"],
                "mae_delta": metrics["mae"] - baseline["mae"],
                "mae_improvement_pct": (
                    (baseline["mae"] - metrics["mae"])
                    / baseline["mae"]
                    * 100
                ),
                "wape": metrics["wape"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "mean_error": metrics["mean_error"],
                "change_volatility_ratio": metrics[
                    "change_volatility_ratio"
                ],
                "change_direction_accuracy": metrics[
                    "change_direction_accuracy"
                ],
                "peak_mae": peak.loc[method, "mae"],
                "peak_mean_error": peak.loc[method, "mean_error"],
                "window_mae_wins": window_wins,
                "eligible": bool(
                    method != BASELINE_METHOD
                    and metrics["mae"] < baseline["mae"]
                    and metrics["wape"] < baseline["wape"]
                    and peak.loc[method, "mae"] < baseline_peak["mae"]
                    and window_wins >= 3
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["eligible", "mae"],
        ascending=[False, True],
    )


def run_second_round_experiments(
    output_dir=OUTPUT_DIR,
    window_count=BACKTEST_WINDOW_COUNT,
):
    df = load_and_preprocess()
    df["store_name"] = df["store_id"].map(STORE_NAME_MAP)
    frames = build_second_round_feature_frames(df)
    base_features = [
        feature
        for feature in get_direct_feature_cols(frames[1])
        if feature not in MINED_FEATURES
    ]
    origins = select_backtest_origins(
        df["date"].unique(),
        window_count=window_count,
        horizon=PREDICTION_LOOKAHEAD,
    )
    all_predictions = [_load_first_round_baseline(origins)]
    training_runs = []

    for origin in origins:
        print(f"\nSecond-round origin {origin.date()}")

        def train_bundle(item):
            method, incremental_features = item
            feature_cols = [*base_features, *incremental_features]
            fit_rows = []
            validation_rows = []
            calibration_rows = []
            test_rows = []
            for horizon_day, frame in frames.items():
                split = split_direct_horizon_frame(
                    frame,
                    feature_cols,
                    origin,
                    horizon_day,
                )
                fit_rows.append(split[0])
                validation_rows.append(split[1])
                calibration_rows.append(split[2])
                test_rows.append(split[3])
            fit = pd.concat(fit_rows, ignore_index=True)
            validation = pd.concat(validation_rows, ignore_index=True)
            calibration = pd.concat(calibration_rows, ignore_index=True)
            test = pd.concat(test_rows, ignore_index=True)
            model = _fit_lightgbm(
                fit,
                validation,
                feature_cols,
                TWEEDIE_PARAMETERS,
            )
            calibration_prediction = _prediction_rows(
                calibration,
                model.predict(calibration[feature_cols]),
                method,
                origin,
                calibration["forecast_horizon"].to_numpy(),
            )
            test_prediction = _prediction_rows(
                test,
                model.predict(test[feature_cols]),
                method,
                origin,
                test["forecast_horizon"].to_numpy(),
            )
            calibrated, factors = _apply_store_calibration(
                calibration_prediction,
                test_prediction,
            )
            run = {
                "forecast_origin": origin.date().isoformat(),
                "method": method,
                "feature_count": len(feature_cols),
                "incremental_features": incremental_features,
                "fit_end": fit["date"].max().date().isoformat(),
                "validation_start": validation[
                    "date"
                ].min().date().isoformat(),
                "validation_end": validation[
                    "date"
                ].max().date().isoformat(),
                "best_iteration": int(model.best_iteration_),
                "store_bias_factors": factors[
                    ["store_id", "store_bias_factor"]
                ].to_dict(orient="records"),
            }
            return calibrated, run

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(train_bundle, FEATURE_BUNDLES.items()))
        for predictions, run in results:
            all_predictions.append(predictions)
            training_runs.append(run)

    predictions = pd.concat(all_predictions, ignore_index=True)
    prediction_key = [
        "forecast_origin",
        "date",
        *ENTITY_KEYS,
        "method",
    ]
    if predictions.duplicated(prediction_key).any():
        raise ValueError("Second-round predictions must be unique")
    if predictions["prediction"].isna().any():
        raise ValueError("Second-round predictions must be complete")
    if (predictions["prediction"] < 0).any():
        raise ValueError("Second-round predictions must be nonnegative")

    quality, peak_threshold, low_threshold = build_experiment_quality(
        predictions
    )
    coverage = _coverage_rows(predictions)
    summary = summarize_second_round(quality, coverage)
    eligible_methods = summary.loc[summary["eligible"], "method"].tolist()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "second_round" if window_count == BACKTEST_WINDOW_COUNT else "smoke"
    predictions_path = output_dir / f"{prefix}_predictions.csv.gz"
    quality_path = output_dir / f"{prefix}_quality.csv"
    coverage_path = output_dir / f"{prefix}_coverage.csv"
    summary_path = output_dir / f"{prefix}_summary.csv"
    metadata_path = output_dir / f"{prefix}_run.json"
    predictions.to_csv(
        predictions_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    quality.to_csv(quality_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata = {
        "experiment_version": EXPERIMENT_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat(),
        "production_model_changed": False,
        "baseline_method": BASELINE_METHOD,
        "origins": [origin.date().isoformat() for origin in origins],
        "forecast_horizon_days": PREDICTION_LOOKAHEAD,
        "base_feature_count": len(base_features),
        "feature_bundles": FEATURE_BUNDLES,
        "tweedie_parameters": TWEEDIE_PARAMETERS,
        "peak_threshold": peak_threshold,
        "low_threshold": low_threshold,
        "promotion_rule": (
            "Candidate must improve aggregate MAE, WAPE, and peak-segment "
            "MAE and beat the first-round baseline in at least three windows."
        ),
        "eligible_methods": eligible_methods,
        "recommended_candidate": (
            eligible_methods[0] if eligible_methods else None
        ),
        "training_runs": training_runs,
        "artifacts": {
            "predictions": str(predictions_path),
            "quality": str(quality_path),
            "coverage": str(coverage_path),
            "summary": str(summary_path),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nSecond-round aggregate summary:")
    print(summary.to_string(index=False))
    return predictions, quality, coverage, summary, metadata


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--window-count",
        type=int,
        default=BACKTEST_WINDOW_COUNT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_second_round_experiments(
        output_dir=args.output_dir,
        window_count=args.window_count,
    )
