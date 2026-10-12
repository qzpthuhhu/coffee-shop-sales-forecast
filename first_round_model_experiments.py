"""First-round direct multi-horizon model experiments.

This module keeps the production recursive model unchanged. It benchmarks
direct T+1 through T+7 models on the same rolling origins and pre-origin store
calibration used by the production backtest.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from backtest_pipeline import (
    BACKTEST_WINDOW_COUNT,
    ENTITY_KEYS,
    REPORT_DIR,
    calculate_metrics,
    calculate_store_bias_factors,
    select_backtest_origins,
)
from predict_pipeline import (
    HISTORY_LAGS,
    HISTORY_WINDOWS,
    LIGHTGBM_PARAMS,
    PREDICTION_LOOKAHEAD,
    STORE_NAME_MAP,
    get_feature_cols,
    load_and_preprocess,
)


EXPERIMENT_VERSION = "direct-multi-horizon-v1"
OUTPUT_DIR = REPORT_DIR / "first_round"
IDENTITY_FEATURES = [
    "store_id_feature",
    "product_id_feature",
    "category_code",
    "product_type_code",
]
LIGHTGBM_EXPERIMENTS = {
    "direct_l1": {"objective": "regression_l1"},
    "direct_l2": {"objective": "regression"},
    "direct_huber": {"objective": "huber", "alpha": 0.90},
    "direct_poisson": {"objective": "poisson"},
    "direct_tweedie_1_1": {
        "objective": "tweedie",
        "tweedie_variance_power": 1.1,
    },
    "direct_tweedie_1_3": {
        "objective": "tweedie",
        "tweedie_variance_power": 1.3,
    },
    "direct_tweedie_1_5": {
        "objective": "tweedie",
        "tweedie_variance_power": 1.5,
    },
}
CATBOOST_PARAMS = {
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "iterations": 1000,
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "thread_count": 4,
    "verbose": False,
    "allow_writing_files": False,
}
def add_direct_history_features(frame, horizon):
    """Build target features available before a fixed-origin horizon."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    result = frame.sort_values([*ENTITY_KEYS, "date"]).reset_index(drop=True)
    qty_group = result.groupby(ENTITY_KEYS)["qty"]

    for lag in HISTORY_LAGS:
        result[f"qty_lag_{lag}"] = qty_group.shift(horizon + lag - 1)

    shifted = qty_group.transform(lambda values: values.shift(horizon))
    grouped_shifted = shifted.groupby(
        [result[key] for key in ENTITY_KEYS]
    )
    for window in HISTORY_WINDOWS:
        result[f"qty_roll_mean_{window}d"] = grouped_shifted.transform(
            lambda values: values.rolling(window, min_periods=1).mean()
        )
        result[f"qty_roll_std_{window}d"] = grouped_shifted.transform(
            lambda values: values.rolling(
                window,
                min_periods=1,
            ).std().fillna(0)
        )

    result["dow"] = result["date"].dt.dayofweek
    result["qty_dow_avg"] = result.groupby(
        [*ENTITY_KEYS, "dow"]
    )["qty"].transform(
        lambda values: values.shift(1).expanding(min_periods=1).mean()
    )

    store_day_total = result.groupby(["store_id", "date"])["qty"].transform(
        "sum"
    )
    result["_observed_sku_share"] = (
        result["qty"] / store_day_total.replace(0, np.nan)
    ).fillna(0)
    result["sku_share_28d"] = result.groupby(
        ENTITY_KEYS
    )["_observed_sku_share"].transform(
        lambda values: values.shift(horizon).rolling(
            28,
            min_periods=1,
        ).mean()
    )
    return result.drop(columns="_observed_sku_share")


def build_direct_feature_frames(df, horizon_days=PREDICTION_LOOKAHEAD):
    """Materialize one leakage-safe feature frame for each forecast horizon."""
    category_values = sorted(df["category"].dropna().unique())
    product_type_values = sorted(df["product_type"].dropna().unique())
    category_codes = {
        value: index for index, value in enumerate(category_values)
    }
    product_type_codes = {
        value: index for index, value in enumerate(product_type_values)
    }
    frames = {}
    for horizon in range(1, horizon_days + 1):
        direct = add_direct_history_features(df.copy(), horizon)
        direct["store_id_feature"] = direct["store_id"].astype(int)
        direct["product_id_feature"] = direct["product_id"].astype(int)
        direct["category_code"] = direct["category"].map(category_codes).astype(
            int
        )
        direct["product_type_code"] = direct["product_type"].map(
            product_type_codes
        ).astype(int)
        direct["forecast_horizon"] = horizon
        frames[horizon] = direct
    return frames


def get_direct_feature_cols(frame):
    """Return production covariates plus explicit series identity features."""
    return list(dict.fromkeys([*get_feature_cols(frame), *IDENTITY_FEATURES]))


def split_direct_horizon_frame(
    frame,
    feature_cols,
    origin,
    horizon_day,
    forecast_horizon=PREDICTION_LOOKAHEAD,
):
    """Create fixed-origin fit, validation, calibration, and test slices."""
    calibration_origin = origin - pd.Timedelta(days=forecast_horizon)
    validation_origin = calibration_origin - pd.Timedelta(
        days=forecast_horizon
    )
    validation_date = validation_origin + pd.Timedelta(days=horizon_day - 1)
    calibration_date = calibration_origin + pd.Timedelta(days=horizon_day - 1)
    test_date = origin + pd.Timedelta(days=horizon_day - 1)

    fit = frame[frame["date"] < validation_origin].dropna(
        subset=[*feature_cols, "qty"]
    )
    validation = frame[frame["date"] == validation_date].dropna(
        subset=[*feature_cols, "qty"]
    )
    calibration = frame[frame["date"] == calibration_date].dropna(
        subset=[*feature_cols, "qty"]
    )
    test = frame[frame["date"] == test_date].dropna(
        subset=[*feature_cols, "qty"]
    )
    if min(map(len, (fit, validation, calibration, test))) == 0:
        raise ValueError(
            f"Insufficient direct data for {origin.date()} T+{horizon_day}"
        )
    return fit, validation, calibration, test


def _fit_lightgbm(fit, validation, feature_cols, parameters):
    model = LGBMRegressor(
        **{
            **LIGHTGBM_PARAMS,
            **parameters,
        }
    )
    model.fit(
        fit[feature_cols],
        fit["qty"],
        eval_X=validation[feature_cols],
        eval_y=validation["qty"],
        eval_metric="mae",
        callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
    )
    return model


def _fit_catboost(fit, validation, feature_cols):
    categorical_indices = [
        feature_cols.index(column) for column in IDENTITY_FEATURES
    ]
    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(
        fit[feature_cols],
        fit["qty"],
        eval_set=(validation[feature_cols], validation["qty"]),
        cat_features=categorical_indices,
        early_stopping_rounds=50,
        verbose=False,
    )
    return model


def _prediction_rows(source, prediction, method, origin, horizon_day):
    result = source[
        [
            "date",
            "store_id",
            "store_name",
            "product_id",
            "category",
            "product_type",
            "qty",
        ]
    ].copy()
    result["store_name"] = result["store_id"].map(STORE_NAME_MAP)
    result["forecast_origin"] = origin
    result["horizon_day"] = horizon_day
    result["method"] = method
    result["prediction_raw"] = np.asarray(prediction, dtype=float)
    result["prediction"] = np.clip(
        result["prediction_raw"],
        a_min=0,
        a_max=None,
    )
    return result


def _apply_store_calibration(calibration, test):
    calibration_for_factor = calibration.rename(
        columns={"prediction": "pred_lightgbm"}
    )
    factors = calculate_store_bias_factors(calibration_for_factor)
    factor_map = factors.set_index("store_id")["store_bias_factor"]
    result = test.copy()
    result["store_bias_factor"] = (
        result["store_id"].map(factor_map).fillna(1.0)
    )
    result["prediction_model"] = result["prediction"]
    result["prediction"] = (
        result["prediction_model"] * result["store_bias_factor"]
    )
    return result, factors


def _load_recursive_baseline(origins):
    path = REPORT_DIR / "rolling_backtest_predictions.csv"
    baseline = pd.read_csv(
        path,
        parse_dates=["date", "forecast_origin"],
    )
    baseline = baseline[baseline["forecast_origin"].isin(origins)].copy()
    baseline["store_name"] = baseline["store_id"].map(STORE_NAME_MAP)
    baseline["method"] = "recursive_l1"
    baseline["prediction_raw"] = baseline["pred_lightgbm_raw"]
    baseline["prediction_model"] = baseline["pred_lightgbm_model"]
    baseline["prediction"] = baseline["pred_lightgbm"]
    return baseline[
        [
            "date",
            "store_id",
            "store_name",
            "product_id",
            "category",
            "product_type",
            "qty",
            "forecast_origin",
            "horizon_day",
            "method",
            "prediction_raw",
            "prediction_model",
            "prediction",
            "store_bias_factor",
        ]
    ]


def _change_metrics(frame):
    ordered = frame.sort_values(
        ["forecast_origin", *ENTITY_KEYS, "date"]
    )
    groups = ordered.groupby(
        ["forecast_origin", *ENTITY_KEYS],
        sort=False,
    )
    actual_change = groups["qty"].diff()
    prediction_change = groups["prediction"].diff()
    valid = actual_change.notna() & prediction_change.notna()
    actual_change = actual_change[valid].to_numpy(dtype=float)
    prediction_change = prediction_change[valid].to_numpy(dtype=float)
    if len(actual_change) == 0:
        return {
            "change_volatility_ratio": np.nan,
            "change_direction_accuracy": np.nan,
        }
    actual_volatility = float(np.std(actual_change))
    nonzero = actual_change != 0
    return {
        "change_volatility_ratio": (
            float(np.std(prediction_change)) / actual_volatility
            if actual_volatility
            else np.nan
        ),
        "change_direction_accuracy": (
            float(
                (
                    np.sign(actual_change[nonzero])
                    == np.sign(prediction_change[nonzero])
                ).mean()
            )
            if nonzero.any()
            else np.nan
        ),
    }


def build_experiment_quality(predictions):
    """Build overall, window, horizon, demand-segment, and SKU metrics."""
    peak_threshold = float(predictions["qty"].quantile(0.90))
    low_threshold = float(predictions["qty"].quantile(0.25))
    rows = []

    def append(frame, method, level, entity_id, entity_name=None):
        metrics = calculate_metrics(frame["qty"], frame["prediction"])
        error = frame["qty"] - frame["prediction"]
        rows.append(
            {
                "level": level,
                "entity_id": str(entity_id),
                "entity_name": (
                    str(entity_name) if entity_name is not None else str(entity_id)
                ),
                "method": method,
                "validation_start": frame["date"].min().date().isoformat(),
                "validation_end": frame["date"].max().date().isoformat(),
                **metrics,
                **_change_metrics(frame),
                "underprediction_rate": float((error > 0).mean()),
            }
        )

    for method, method_frame in predictions.groupby("method", sort=True):
        append(method_frame, method, "overall", "all", "全部")
        for origin, frame in method_frame.groupby(
            "forecast_origin",
            sort=True,
        ):
            append(
                frame,
                method,
                "window",
                origin.date().isoformat(),
                origin.date().isoformat(),
            )
        for horizon_day, frame in method_frame.groupby(
            "horizon_day",
            sort=True,
        ):
            append(frame, method, "horizon", horizon_day, f"T+{horizon_day}")
        append(
            method_frame[method_frame["qty"] >= peak_threshold],
            method,
            "demand_segment",
            "peak_p90",
            f"实际销量 >= {peak_threshold:g}",
        )
        append(
            method_frame[method_frame["qty"] <= low_threshold],
            method,
            "demand_segment",
            "low_p25",
            f"实际销量 <= {low_threshold:g}",
        )
        for product_id, frame in method_frame.groupby(
            "product_id",
            sort=True,
        ):
            append(frame, method, "sku", product_id, product_id)

    quality = pd.DataFrame(rows)
    return quality, peak_threshold, low_threshold


def summarize_candidates(quality):
    """Apply a conservative promotion screen against recursive LightGBM."""
    overall = quality[quality["level"] == "overall"].set_index("method")
    windows = quality[quality["level"] == "window"].pivot(
        index="entity_id",
        columns="method",
        values="mae",
    )
    peak = quality[
        (quality["level"] == "demand_segment")
        & (quality["entity_id"] == "peak_p90")
    ].set_index("method")
    baseline = overall.loc["recursive_l1"]
    baseline_peak = peak.loc["recursive_l1"]
    rows = []
    for method, metrics in overall.iterrows():
        window_wins = int(
            (windows[method] < windows["recursive_l1"]).sum()
        )
        rows.append(
            {
                "method": method,
                "mae": metrics["mae"],
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
                    method != "recursive_l1"
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


def run_first_round_experiments(output_dir=OUTPUT_DIR):
    df = load_and_preprocess()
    df["store_name"] = df["store_id"].map(STORE_NAME_MAP)
    direct_frames = build_direct_feature_frames(df)
    feature_cols = get_direct_feature_cols(direct_frames[1])
    origins = select_backtest_origins(
        df["date"].unique(),
        window_count=BACKTEST_WINDOW_COUNT,
        horizon=PREDICTION_LOOKAHEAD,
    )
    all_predictions = [_load_recursive_baseline(origins)]
    training_runs = []

    for origin in origins:
        print(f"\nFirst-round origin {origin.date()}")
        fit_rows = []
        validation_rows = []
        calibration_rows = []
        test_rows = []
        for horizon_day, direct in direct_frames.items():
            fit, validation, calibration, test = split_direct_horizon_frame(
                direct,
                feature_cols,
                origin,
                horizon_day,
            )
            fit_rows.append(fit)
            validation_rows.append(validation)
            calibration_rows.append(calibration)
            test_rows.append(test)

        fit = pd.concat(fit_rows, ignore_index=True)
        validation = pd.concat(validation_rows, ignore_index=True)
        calibration = pd.concat(calibration_rows, ignore_index=True)
        test = pd.concat(test_rows, ignore_index=True)

        def fit_lightgbm_experiment(item):
            method, parameters = item
            return method, _fit_lightgbm(
                fit,
                validation,
                feature_cols,
                parameters,
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            origin_models = dict(
                executor.map(
                    fit_lightgbm_experiment,
                    LIGHTGBM_EXPERIMENTS.items(),
                )
            )
        origin_models["direct_catboost"] = _fit_catboost(
            fit,
            validation,
            feature_cols,
        )

        for method, model in origin_models.items():
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
            all_predictions.append(calibrated)
            best_iteration = (
                model.get_best_iteration()
                if method == "direct_catboost"
                else model.best_iteration_
            )
            training_runs.append(
                {
                    "forecast_origin": origin.date().isoformat(),
                    "method": method,
                    "fit_end": fit["date"].max().date().isoformat(),
                    "validation_start": validation[
                        "date"
                    ].min().date().isoformat(),
                    "validation_end": validation[
                        "date"
                    ].max().date().isoformat(),
                    "best_iteration": int(best_iteration),
                    "store_bias_factors": factors[
                        ["store_id", "store_bias_factor"]
                    ].to_dict(orient="records"),
                }
            )

    predictions = pd.concat(all_predictions, ignore_index=True)
    prediction_key = [
        "forecast_origin",
        "date",
        *ENTITY_KEYS,
        "method",
    ]
    if predictions.duplicated(prediction_key).any():
        raise ValueError("First-round predictions must be unique")
    if predictions["prediction"].isna().any():
        raise ValueError("First-round predictions must be complete")
    if (predictions["prediction"] < 0).any():
        raise ValueError("First-round predictions must be nonnegative")
    quality, peak_threshold, low_threshold = build_experiment_quality(
        predictions
    )
    summary = summarize_candidates(quality)
    eligible_methods = summary.loc[summary["eligible"], "method"].tolist()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "first_round_predictions.csv.gz"
    quality_path = output_dir / "first_round_quality.csv"
    summary_path = output_dir / "first_round_summary.csv"
    metadata_path = output_dir / "first_round_run.json"
    predictions.to_csv(
        predictions_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    quality.to_csv(quality_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata = {
        "experiment_version": EXPERIMENT_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat(),
        "production_model_changed": False,
        "forecast_strategy": (
            "Stacked direct multi-horizon models with shared parameters and "
            "an explicit forecast_horizon feature"
        ),
        "origins": [origin.date().isoformat() for origin in origins],
        "forecast_horizon_days": PREDICTION_LOOKAHEAD,
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "identity_features": IDENTITY_FEATURES,
        "lightgbm_experiments": LIGHTGBM_EXPERIMENTS,
        "catboost_parameters": CATBOOST_PARAMS,
        "peak_threshold": peak_threshold,
        "low_threshold": low_threshold,
        "promotion_rule": (
            "Candidate must improve aggregate MAE, WAPE, and peak-segment "
            "MAE and beat recursive LightGBM MAE in at least three windows."
        ),
        "eligible_methods": eligible_methods,
        "recommended_candidate": (
            eligible_methods[0] if eligible_methods else None
        ),
        "training_runs": training_runs,
        "artifacts": {
            "predictions": str(predictions_path),
            "quality": str(quality_path),
            "summary": str(summary_path),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nFirst-round aggregate summary:")
    print(summary.to_string(index=False))
    return predictions, quality, summary, metadata


if __name__ == "__main__":
    run_first_round_experiments()
