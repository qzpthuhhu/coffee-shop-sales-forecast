"""Leakage-safe rolling-origin backtests for the global LightGBM model."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from predict_pipeline import (
    DATA_DIR,
    HISTORY_LAGS,
    HISTORY_WINDOWS,
    LIGHTGBM_PARAMS,
    PREDICTION_LOOKAHEAD,
    SALES_FILE,
    WEATHER_FILE,
    add_history_features,
    get_feature_cols,
    load_and_preprocess,
)


REPORT_DIR = DATA_DIR / "reports" / "model_quality"
BACKTEST_WINDOW_COUNT = 5
PIPELINE_VERSION = "rolling-backtest-v7-kalman-candidate"
ENTITY_KEYS = ["store_id", "product_id"]
HISTORY_FEATURE_COLS = (
    [f"qty_lag_{lag}" for lag in HISTORY_LAGS]
    + [
        f"qty_roll_{stat}_{window}d"
        for window in HISTORY_WINDOWS
        for stat in ("mean", "std")
    ]
    + ["qty_dow_avg", "sku_share_28d"]
)
PREDICTION_COLUMNS = {
    "lightgbm": "pred_lightgbm",
    "kalman": "pred_kalman",
    "lightgbm_kalman": "pred_lightgbm_kalman",
    "baseline_yesterday": "pred_baseline_yesterday",
    "baseline_last_week": "pred_baseline_last_week",
    "baseline_rolling_28d": "pred_baseline_rolling_28d",
}
KALMAN_CONFIG = {
    "history_days": 56,
    "seasonal_shrinkage": 0.50,
    "level_process_ratio": 0.15,
    "trend_process_ratio": 0.01,
    "trend_damping": 0.90,
    "minimum_variance": 1.0,
}
KALMAN_FUSION_WEIGHTS = tuple(np.linspace(0.0, 1.0, 11))
QUANTILE_LEVELS = {
    "pred_p10": 0.10,
    "pred_p50": 0.50,
    "pred_p90": 0.90,
}
INTERVAL_TARGET_COVERAGE = 0.80
SHAP_ADDITIVITY_TOLERANCE = 1e-6
STORE_BIAS_SHRINKAGE = 0.50
STORE_BIAS_FACTOR_MIN = 0.85
STORE_BIAS_FACTOR_MAX = 1.20
SCENARIO_TEMPERATURES = (
    {"id": "cooler_5c", "label": "降温 5°C", "delta_c": -5.0},
    {"id": "observed", "label": "原始天气", "delta_c": 0.0},
    {"id": "warmer_5c", "label": "升温 5°C", "delta_c": 5.0},
)
SCENARIO_RAIN = (
    {"id": "dry", "label": "无降雨", "precipitation_mm": 0.0},
    {"id": "observed", "label": "原始降雨", "precipitation_mm": None},
    {"id": "heavy_20mm", "label": "强降雨 20 mm", "precipitation_mm": 20.0},
)
SCENARIO_HOLIDAYS = (
    {"id": "ordinary_day", "label": "普通日"},
    {"id": "observed", "label": "原始节假日"},
    {"id": "long_weekend", "label": "长周末"},
)
SCENARIO_BASELINE = ("observed", "observed", "observed")


def build_model_run_id(
    source_hashes,
    origins,
    horizon,
    feature_cols,
):
    """Build a stable model-run identifier from reproducibility inputs."""
    run_contract = {
        "pipeline_version": PIPELINE_VERSION,
        "source_hashes": source_hashes,
        "origins": [pd.Timestamp(origin).date().isoformat() for origin in origins],
        "horizon": horizon,
        "features": feature_cols,
        "parameters": LIGHTGBM_PARAMS,
        "quantile_levels": QUANTILE_LEVELS,
        "interval_target_coverage": INTERVAL_TARGET_COVERAGE,
        "kalman": {
            **KALMAN_CONFIG,
            "fusion_weights": KALMAN_FUSION_WEIGHTS,
        },
        "store_bias_calibration": {
            "shrinkage": STORE_BIAS_SHRINKAGE,
            "minimum_factor": STORE_BIAS_FACTOR_MIN,
            "maximum_factor": STORE_BIAS_FACTOR_MAX,
        },
    }
    digest = hashlib.sha256(
        json.dumps(run_contract, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"rolling_backtest_{digest}"


def add_prediction_identity(frame, model_run_id, origin):
    """Attach stable identifiers for each store-SKU-date prediction."""
    result = frame.copy()
    origin_key = pd.Timestamp(origin).strftime("%Y%m%d")
    date_keys = result["date"].dt.strftime("%Y%m%d")
    result["model_run_id"] = model_run_id
    result["prediction_id"] = (
        model_run_id
        + "_o"
        + origin_key
        + "_d"
        + date_keys
        + "_s"
        + result["store_id"].astype(str)
        + "_sku"
        + result["product_id"].astype(str)
    )
    if result["prediction_id"].duplicated().any():
        raise ValueError("Prediction identifiers must be unique within a forecast")
    return result


def explain_tree_predictions(model, feature_frame, prediction_ids, feature_cols):
    """Return row-level TreeSHAP summaries and signed long-form contributions."""
    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(feature_frame), dtype=float)
    if shap_values.ndim != 2 or shap_values.shape != feature_frame.shape:
        raise ValueError(
            "Expected one TreeSHAP contribution per prediction and feature"
        )

    expected_value = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    if expected_value.size != 1:
        raise ValueError("Expected a single base value for the regression model")
    base_values = np.full(len(feature_frame), expected_value.item())
    model_outputs = np.asarray(model.predict(feature_frame), dtype=float)
    shap_sums = shap_values.sum(axis=1)
    additivity_errors = base_values + shap_sums - model_outputs
    max_error = float(np.max(np.abs(additivity_errors), initial=0))
    if max_error > SHAP_ADDITIVITY_TOLERANCE:
        raise ValueError(
            "TreeSHAP additivity validation failed: "
            f"max error={max_error:.8f}"
        )

    ids = pd.Series(prediction_ids, dtype="object").reset_index(drop=True)
    summary = pd.DataFrame(
        {
            "prediction_id": ids,
            "base_value": base_values,
            "shap_sum": shap_sums,
            "shap_additivity_error": additivity_errors,
        }
    )
    contributions = pd.DataFrame(
        {
            "prediction_id": np.repeat(ids.to_numpy(), len(feature_cols)),
            "feature": np.tile(np.asarray(feature_cols, dtype=object), len(ids)),
            "feature_value": feature_frame[feature_cols].to_numpy().reshape(-1),
            "shap_contribution": shap_values.reshape(-1),
        }
    )
    contributions["direction"] = np.select(
        [
            contributions["shap_contribution"] > 0,
            contributions["shap_contribution"] < 0,
        ],
        ["increase", "decrease"],
        default="neutral",
    )
    return summary, contributions


def build_scenario_grid(model, current, feature_cols):
    """Score controlled weather and holiday combinations for each prediction."""
    required = {
        "prediction_id",
        "pred_lightgbm",
        "store_bias_factor",
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "is_rain",
        "is_snow",
        "is_clear",
        "is_holiday_detailed",
        "is_pre_holiday_1d",
        "is_pre_holiday_3d",
        "is_post_holiday_1d",
        "is_long_weekend",
        "holiday_impact_code",
        "is_hot",
        "is_cold",
    }
    missing = required.difference(current.columns)
    if missing:
        raise ValueError(f"Missing scenario columns: {sorted(missing)}")

    scenario_rows = []
    scenario_index = 0
    for temperature in SCENARIO_TEMPERATURES:
        for rain in SCENARIO_RAIN:
            for holiday in SCENARIO_HOLIDAYS:
                scenario = current.copy()
                temperature_delta = temperature["delta_c"]
                for column in (
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "temperature_2m_mean",
                ):
                    scenario[column] = scenario[column] + temperature_delta
                scenario["is_hot"] = (
                    scenario["temperature_2m_max"] > 25
                ).astype(int)
                scenario["is_cold"] = (
                    scenario["temperature_2m_max"] < 5
                ).astype(int)

                precipitation = rain["precipitation_mm"]
                if precipitation is not None:
                    scenario["precipitation_sum"] = precipitation
                    scenario["is_rain"] = int(precipitation > 0)
                    scenario["is_snow"] = 0
                    scenario["is_clear"] = int(precipitation == 0)

                if holiday["id"] != "observed":
                    for column in (
                        "is_holiday_detailed",
                        "is_pre_holiday_1d",
                        "is_pre_holiday_3d",
                        "is_post_holiday_1d",
                        "is_long_weekend",
                    ):
                        scenario[column] = 0
                    scenario["holiday_impact_code"] = 0
                    if holiday["id"] == "long_weekend":
                        scenario["is_holiday_detailed"] = 1
                        scenario["is_long_weekend"] = 1
                        scenario["holiday_impact_code"] = 2

                raw_prediction = np.asarray(
                    model.predict(scenario[feature_cols]),
                    dtype=float,
                )
                simulated = (
                    np.clip(raw_prediction, a_min=0, a_max=None)
                    * scenario["store_bias_factor"].to_numpy(dtype=float)
                )
                scenario_rows.append(
                    pd.DataFrame(
                        {
                            "prediction_id": scenario["prediction_id"],
                            "scenario_index": scenario_index,
                            "temperature_id": temperature["id"],
                            "rain_id": rain["id"],
                            "holiday_id": holiday["id"],
                            "temperature_2m_mean": scenario[
                                "temperature_2m_mean"
                            ],
                            "precipitation_sum": scenario[
                                "precipitation_sum"
                            ],
                            "is_holiday_detailed": scenario[
                                "is_holiday_detailed"
                            ],
                            "is_long_weekend": scenario["is_long_weekend"],
                            "simulated_prediction": simulated,
                        }
                    )
                )
                scenario_index += 1

    result = pd.concat(scenario_rows, ignore_index=True)
    baseline = result[
        (result["temperature_id"] == SCENARIO_BASELINE[0])
        & (result["rain_id"] == SCENARIO_BASELINE[1])
        & (result["holiday_id"] == SCENARIO_BASELINE[2])
    ].sort_values("prediction_id")
    expected = current.sort_values("prediction_id")
    np.testing.assert_allclose(
        baseline["simulated_prediction"],
        expected["pred_lightgbm"],
        atol=1e-9,
        rtol=0,
    )
    return result


def select_backtest_origins(
    dates,
    window_count=BACKTEST_WINDOW_COUNT,
    horizon=PREDICTION_LOOKAHEAD,
):
    """Select the latest fixed, non-overlapping rolling origins."""
    normalized = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values().unique()
    required_days = window_count * horizon
    if len(normalized) < required_days:
        raise ValueError(
            f"Need at least {required_days} dates for {window_count} backtest windows"
        )
    return [
        pd.Timestamp(normalized[-required_days + offset * horizon])
        for offset in range(window_count)
    ]


def _fit_model_for_origin(df, feature_cols, origin, horizon):
    """Fit models with disjoint training, validation, and calibration windows."""
    calibration_start = origin - pd.Timedelta(days=horizon)
    internal_validation_start = calibration_start - pd.Timedelta(days=horizon)
    fit_df = df[df["date"] < internal_validation_start].dropna(
        subset=feature_cols + ["qty"]
    )
    validation_df = df[
        (df["date"] >= internal_validation_start)
        & (df["date"] < calibration_start)
    ].dropna(subset=feature_cols + ["qty"])
    calibration_df = df[
        (df["date"] >= calibration_start) & (df["date"] < origin)
    ].dropna(subset=feature_cols + ["qty"])
    if fit_df.empty or validation_df.empty or calibration_df.empty:
        raise ValueError(f"Insufficient training history for origin {origin.date()}")

    model = LGBMRegressor(**LIGHTGBM_PARAMS)
    model.fit(
        fit_df[feature_cols],
        fit_df["qty"],
        eval_X=validation_df[feature_cols],
        eval_y=validation_df["qty"],
        eval_metric="mae",
        callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
    )
    quantile_models = {}
    for prediction_column, alpha in QUANTILE_LEVELS.items():
        quantile_model = LGBMRegressor(
            **{
                **LIGHTGBM_PARAMS,
                "objective": "quantile",
                "alpha": alpha,
            }
        )
        quantile_model.fit(
            fit_df[feature_cols],
            fit_df["qty"],
            eval_X=validation_df[feature_cols],
            eval_y=validation_df["qty"],
            eval_metric="quantile",
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        quantile_models[prediction_column] = quantile_model
    return model, quantile_models, fit_df, validation_df, calibration_df


def recursive_forecast(
    model,
    df,
    feature_cols,
    origin,
    horizon,
    quantile_models=None,
    store_bias_factors=None,
    model_run_id=None,
    include_explanations=False,
    include_scenarios=False,
):
    """Forecast one day at a time without reading labels at or after origin."""
    if (include_explanations or include_scenarios) and not model_run_id:
        raise ValueError(
            "model_run_id is required when explanations or scenarios are enabled"
        )

    forecast_dates = pd.date_range(origin, periods=horizon, freq="D")
    state = df[["date", *ENTITY_KEYS, "qty"]].copy()
    state.loc[state["date"] >= origin, "qty"] = np.nan
    forecast_rows = []
    explanation_rows = []
    scenario_rows = []

    for forecast_date in forecast_dates:
        history = add_history_features(state[state["date"] <= forecast_date].copy())
        history_today = history[history["date"] == forecast_date][
            [*ENTITY_KEYS, *HISTORY_FEATURE_COLS]
        ]
        current = df[df["date"] == forecast_date].copy()
        current = current.drop(columns=HISTORY_FEATURE_COLS).merge(
            history_today,
            on=ENTITY_KEYS,
            how="left",
            validate="one_to_one",
        )
        current = current.sort_values(ENTITY_KEYS).reset_index(drop=True)
        current["pred_lightgbm_raw"] = model.predict(current[feature_cols])
        current["pred_lightgbm"] = np.clip(
            current["pred_lightgbm_raw"],
            a_min=0,
            a_max=None,
        )
        current["postprocess_adjustment"] = (
            current["pred_lightgbm"] - current["pred_lightgbm_raw"]
        )
        if model_run_id:
            current = add_prediction_identity(current, model_run_id, origin)
        if include_explanations:
            explanation_summary, contributions = explain_tree_predictions(
                model,
                current[feature_cols],
                current["prediction_id"],
                feature_cols,
            )
            current = current.merge(
                explanation_summary,
                on="prediction_id",
                how="left",
                validate="one_to_one",
            )
            explanation_rows.append(contributions)
        if quantile_models:
            quantiles = np.column_stack(
                [
                    np.clip(
                        quantile_models[column].predict(current[feature_cols]),
                        a_min=0,
                        a_max=None,
                    )
                    for column in QUANTILE_LEVELS
                ]
            )
            quantiles.sort(axis=1)
            current[list(QUANTILE_LEVELS)] = quantiles
        if store_bias_factors is not None:
            current = apply_store_bias_calibration(
                current,
                store_bias_factors,
            )
        if include_scenarios:
            scenario_rows.append(
                build_scenario_grid(model, current, feature_cols)
            )
        forecast_rows.append(current)

        predicted = current[[*ENTITY_KEYS, "pred_lightgbm"]].rename(
            columns={"pred_lightgbm": "qty"}
        )
        update_index = state["date"] == forecast_date
        state_today = state.loc[update_index, [*ENTITY_KEYS]].merge(
            predicted,
            on=ENTITY_KEYS,
            how="left",
            validate="one_to_one",
        )
        state.loc[update_index, "qty"] = state_today["qty"].to_numpy()

    forecasts = pd.concat(forecast_rows, ignore_index=True)
    if include_explanations and include_scenarios:
        return (
            forecasts,
            pd.concat(explanation_rows, ignore_index=True),
            pd.concat(scenario_rows, ignore_index=True),
        )
    if include_explanations:
        return forecasts, pd.concat(explanation_rows, ignore_index=True)
    if include_scenarios:
        return forecasts, pd.concat(scenario_rows, ignore_index=True)
    return forecasts


def calculate_store_bias_factors(
    calibration_forecast,
    shrinkage=STORE_BIAS_SHRINKAGE,
    minimum_factor=STORE_BIAS_FACTOR_MIN,
    maximum_factor=STORE_BIAS_FACTOR_MAX,
):
    """Estimate leakage-safe store factors from a pre-origin calibration window."""
    required_columns = {"store_id", "qty", "pred_lightgbm"}
    missing = required_columns.difference(calibration_forecast.columns)
    if missing:
        raise ValueError(f"Missing store calibration columns: {sorted(missing)}")
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be between 0 and 1")
    if minimum_factor <= 0 or maximum_factor < minimum_factor:
        raise ValueError("Store factor bounds must be positive and ordered")

    factors = (
        calibration_forecast.groupby("store_id", as_index=False)
        .agg(
            calibration_actual=("qty", "sum"),
            calibration_prediction=("pred_lightgbm", "sum"),
            calibration_sample_count=("qty", "size"),
        )
        .sort_values("store_id")
        .reset_index(drop=True)
    )
    valid_prediction = (
        np.isfinite(factors["calibration_prediction"])
        & (factors["calibration_prediction"] > 0)
        & np.isfinite(factors["calibration_actual"])
    )
    factors["store_bias_ratio_raw"] = 1.0
    factors.loc[valid_prediction, "store_bias_ratio_raw"] = (
        factors.loc[valid_prediction, "calibration_actual"]
        / factors.loc[valid_prediction, "calibration_prediction"]
    )
    shrunk = 1 + shrinkage * (factors["store_bias_ratio_raw"] - 1)
    factors["store_bias_factor"] = shrunk.clip(
        lower=minimum_factor,
        upper=maximum_factor,
    )
    return factors


def apply_store_bias_calibration(forecast, store_bias_factors):
    """Apply store factors to point and quantile forecasts with audit columns."""
    required_columns = {"store_id", "pred_lightgbm"}
    missing = required_columns.difference(forecast.columns)
    if missing:
        raise ValueError(f"Missing forecast columns: {sorted(missing)}")
    factor_columns = {"store_id", "store_bias_factor"}
    missing_factors = factor_columns.difference(store_bias_factors.columns)
    if missing_factors:
        raise ValueError(f"Missing store factor columns: {sorted(missing_factors)}")
    if store_bias_factors["store_id"].duplicated().any():
        raise ValueError("Store bias factors must be unique by store_id")

    result = forecast.merge(
        store_bias_factors[["store_id", "store_bias_factor"]],
        on="store_id",
        how="left",
        validate="many_to_one",
    )
    result["store_bias_factor"] = result["store_bias_factor"].fillna(1.0)
    if (
        ~np.isfinite(result["store_bias_factor"])
        | (result["store_bias_factor"] <= 0)
    ).any():
        raise ValueError("Store bias factors must be finite and positive")

    result["pred_lightgbm_model"] = result["pred_lightgbm"]
    result["pred_lightgbm"] = (
        result["pred_lightgbm_model"] * result["store_bias_factor"]
    )
    result["store_bias_adjustment"] = (
        result["pred_lightgbm"] - result["pred_lightgbm_model"]
    )
    quantile_columns = [
        column for column in QUANTILE_LEVELS if column in result.columns
    ]
    if quantile_columns:
        scaled_quantiles = (
            result[quantile_columns].to_numpy(dtype=float)
            * result["store_bias_factor"].to_numpy(dtype=float)[:, None]
        )
        scaled_quantiles.sort(axis=1)
        result[quantile_columns] = scaled_quantiles
    return result


def calculate_conformal_adjustment(
    calibration_forecast,
    target_coverage=INTERVAL_TARGET_COVERAGE,
):
    """Calculate the finite-sample CQR correction from pre-origin forecasts."""
    required_columns = {"qty", "pred_p10", "pred_p90"}
    missing = required_columns.difference(calibration_forecast.columns)
    if missing:
        raise ValueError(f"Missing conformal calibration columns: {sorted(missing)}")
    if not 0 < target_coverage < 1:
        raise ValueError("target_coverage must be between 0 and 1")

    scores = np.maximum(
        calibration_forecast["pred_p10"].to_numpy(dtype=float)
        - calibration_forecast["qty"].to_numpy(dtype=float),
        calibration_forecast["qty"].to_numpy(dtype=float)
        - calibration_forecast["pred_p90"].to_numpy(dtype=float),
    )
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        raise ValueError("Conformal calibration requires finite scores")

    quantile_level = min(
        1.0,
        np.ceil((len(scores) + 1) * target_coverage) / len(scores),
    )
    return float(np.quantile(scores, quantile_level, method="higher"))


def apply_conformal_adjustment(forecast, adjustment):
    """Apply a scalar CQR correction while preserving ordered nonnegative bounds."""
    result = forecast.copy()
    result["pred_p10_uncalibrated"] = result["pred_p10"]
    result["pred_p90_uncalibrated"] = result["pred_p90"]
    calibrated_lower = np.clip(
        result["pred_p10_uncalibrated"] - adjustment,
        a_min=0,
        a_max=None,
    )
    calibrated_upper = np.clip(
        result["pred_p90_uncalibrated"] + adjustment,
        a_min=0,
        a_max=None,
    )
    result["pred_p10"] = np.minimum(calibrated_lower, result["pred_p50"])
    result["pred_p90"] = np.maximum(calibrated_upper, result["pred_p50"])
    result["conformal_adjustment"] = adjustment
    return result


def add_origin_safe_baselines(forecast_df, df, origin):
    """Add baselines that use only observations available at forecast origin."""
    history = df[df["date"] < origin][["date", *ENTITY_KEYS, "qty"]].copy()
    latest = (
        history.sort_values("date")
        .groupby(ENTITY_KEYS, as_index=False)
        .tail(1)[[*ENTITY_KEYS, "qty"]]
        .rename(columns={"qty": "pred_baseline_yesterday"})
    )
    rolling_28 = (
        history.sort_values("date")
        .groupby(ENTITY_KEYS, group_keys=False)
        .tail(28)
        .groupby(ENTITY_KEYS, as_index=False)["qty"]
        .mean()
        .rename(columns={"qty": "pred_baseline_rolling_28d"})
    )
    last_week = history.rename(
        columns={
            "date": "history_date",
            "qty": "pred_baseline_last_week",
        }
    )

    result = forecast_df.merge(latest, on=ENTITY_KEYS, how="left")
    result = result.merge(rolling_28, on=ENTITY_KEYS, how="left")
    result["history_date"] = result["date"] - pd.Timedelta(days=7)
    result = result.merge(
        last_week,
        on=["history_date", *ENTITY_KEYS],
        how="left",
        validate="many_to_one",
    ).drop(columns="history_date")
    return result


def _kalman_local_trend(values, dates, forecast_dates, config=None):
    """Forecast one series with a damped local-trend Kalman filter."""
    parameters = {**KALMAN_CONFIG, **(config or {})}
    values = np.asarray(values, dtype=float)
    dates = pd.DatetimeIndex(dates)
    forecast_dates = pd.DatetimeIndex(forecast_dates)
    finite = np.isfinite(values)
    values = values[finite]
    dates = dates[finite]
    if len(values) == 0:
        return np.zeros(len(forecast_dates), dtype=float)

    overall_mean = float(values.mean())
    seasonal_offsets = {}
    for weekday in range(7):
        weekday_values = values[dates.dayofweek == weekday]
        raw_offset = (
            float(weekday_values.mean() - overall_mean)
            if len(weekday_values)
            else 0.0
        )
        seasonal_offsets[weekday] = (
            parameters["seasonal_shrinkage"] * raw_offset
        )
    adjusted = values - np.array(
        [seasonal_offsets[weekday] for weekday in dates.dayofweek],
        dtype=float,
    )

    if len(adjusted) > 7:
        noise_sample = adjusted[7:] - adjusted[:-7]
    else:
        noise_sample = np.diff(adjusted)
    observation_variance = max(
        float(np.nanvar(noise_sample) / 2) if len(noise_sample) else 0.0,
        parameters["minimum_variance"],
    )
    recent_differences = np.diff(adjusted[-14:])
    initial_trend = (
        float(np.nanmedian(recent_differences))
        if len(recent_differences)
        else 0.0
    )
    state = np.array([adjusted[0], initial_trend], dtype=float)
    covariance = np.diag(
        [observation_variance * 10, observation_variance]
    )
    transition = np.array(
        [[1.0, parameters["trend_damping"]], [0.0, parameters["trend_damping"]]],
        dtype=float,
    )
    observation = np.array([[1.0, 0.0]], dtype=float)
    process_noise = np.diag(
        [
            observation_variance * parameters["level_process_ratio"],
            observation_variance * parameters["trend_process_ratio"],
        ]
    )
    identity = np.eye(2)

    for measurement in adjusted:
        predicted_state = transition @ state
        predicted_covariance = (
            transition @ covariance @ transition.T + process_noise
        )
        innovation = measurement - (observation @ predicted_state).item()
        innovation_variance = (
            observation @ predicted_covariance @ observation.T
            + observation_variance
        ).item()
        gain = predicted_covariance @ observation.T / innovation_variance
        state = predicted_state + gain[:, 0] * innovation
        covariance = (identity - gain @ observation) @ predicted_covariance

    predictions = []
    for forecast_date in forecast_dates:
        state = transition @ state
        covariance = transition @ covariance @ transition.T + process_noise
        prediction = state[0] + seasonal_offsets[forecast_date.dayofweek]
        predictions.append(max(0.0, float(prediction)))
    return np.asarray(predictions)


def kalman_forecast(df, origin, horizon, config=None):
    """Create store-SKU Kalman forecasts using only observations before origin."""
    parameters = {**KALMAN_CONFIG, **(config or {})}
    history_start = origin - pd.Timedelta(days=parameters["history_days"])
    history = df[
        (df["date"] < origin) & (df["date"] >= history_start)
    ][["date", *ENTITY_KEYS, "qty"]].copy()
    forecast_dates = pd.date_range(origin, periods=horizon, freq="D")
    rows = []
    for entity_values, entity_history in history.groupby(
        ENTITY_KEYS, sort=True
    ):
        entity_history = entity_history.sort_values("date")
        predictions = _kalman_local_trend(
            entity_history["qty"].to_numpy(dtype=float),
            entity_history["date"],
            forecast_dates,
            config=parameters,
        )
        store_id, product_id = entity_values
        rows.append(
            pd.DataFrame(
                {
                    "date": forecast_dates,
                    "store_id": store_id,
                    "product_id": product_id,
                    "pred_kalman": predictions,
                }
            )
        )
    if not rows:
        raise ValueError(f"Insufficient Kalman history for origin {origin.date()}")
    return pd.concat(rows, ignore_index=True)


def apply_kalman_fusion(forecast, kalman, weight):
    """Merge Kalman predictions and blend them with calibrated LightGBM."""
    if not 0 <= weight <= 1:
        raise ValueError("Kalman fusion weight must be between 0 and 1")
    result = forecast.merge(
        kalman,
        on=["date", *ENTITY_KEYS],
        how="left",
        validate="one_to_one",
    )
    if result["pred_kalman"].isna().any():
        raise ValueError("Kalman predictions must cover every forecast row")
    result["kalman_fusion_weight"] = float(weight)
    result["pred_lightgbm_kalman"] = (
        (1 - weight) * result["pred_lightgbm"]
        + weight * result["pred_kalman"]
    ).clip(lower=0)
    return result


def select_kalman_fusion_weight(
    calibration_forecast,
    weights=KALMAN_FUSION_WEIGHTS,
):
    """Select a global blend weight from the pre-origin calibration window."""
    required = {"qty", "pred_lightgbm", "pred_kalman"}
    missing = required.difference(calibration_forecast.columns)
    if missing:
        raise ValueError(f"Missing fusion calibration columns: {sorted(missing)}")
    actual = calibration_forecast["qty"].to_numpy(dtype=float)
    lightgbm = calibration_forecast["pred_lightgbm"].to_numpy(dtype=float)
    kalman = calibration_forecast["pred_kalman"].to_numpy(dtype=float)
    denominator = np.abs(actual).sum()
    candidates = []
    for weight in weights:
        prediction = (1 - weight) * lightgbm + weight * kalman
        absolute_error = np.abs(actual - prediction)
        candidates.append(
            {
                "weight": float(weight),
                "mae": float(absolute_error.mean()),
                "wape": (
                    float(absolute_error.sum() / denominator)
                    if denominator
                    else np.nan
                ),
            }
        )
    selection = min(candidates, key=lambda row: (row["mae"], row["weight"]))
    return selection["weight"], candidates


def calculate_metrics(actual, predicted, lower=None, upper=None):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = actual - predicted
    denominator = np.abs(actual).sum()
    interval_coverage = np.nan
    interval_mean_width = np.nan
    interval_median_width = np.nan
    if lower is not None and upper is not None:
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        interval_width = upper - lower
        interval_coverage = float(((actual >= lower) & (actual <= upper)).mean())
        interval_mean_width = float(interval_width.mean())
        interval_median_width = float(np.median(interval_width))
    return {
        "sample_count": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "wape": float(np.abs(error).sum() / denominator) if denominator else np.nan,
        "mean_error": float(error.mean()),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else np.nan,
        "within_10pct": float((np.abs(error / (actual + 1)) < 0.10).mean()),
        "within_20pct": float((np.abs(error / (actual + 1)) < 0.20).mean()),
        "interval_coverage": interval_coverage,
        "interval_mean_width": interval_mean_width,
        "interval_median_width": interval_median_width,
    }


def calculate_change_metrics(frame, prediction_column):
    """Measure day-to-day amplitude and direction within each forecast series."""
    ordered = frame.sort_values(
        ["forecast_origin", *ENTITY_KEYS, "date"]
    ).copy()
    groups = ordered.groupby(
        ["forecast_origin", *ENTITY_KEYS], sort=False
    )
    actual_change = groups["qty"].diff()
    predicted_change = groups[prediction_column].diff()
    valid = actual_change.notna() & predicted_change.notna()
    actual_change = actual_change[valid].to_numpy(dtype=float)
    predicted_change = predicted_change[valid].to_numpy(dtype=float)
    if len(actual_change) == 0:
        return {
            "change_sample_count": 0,
            "change_volatility_ratio": np.nan,
            "change_direction_accuracy": np.nan,
        }
    actual_volatility = float(np.std(actual_change))
    predicted_volatility = float(np.std(predicted_change))
    nonzero = actual_change != 0
    direction_accuracy = (
        float(
            (
                np.sign(actual_change[nonzero])
                == np.sign(predicted_change[nonzero])
            ).mean()
        )
        if nonzero.any()
        else np.nan
    )
    return {
        "change_sample_count": int(len(actual_change)),
        "change_volatility_ratio": (
            predicted_volatility / actual_volatility
            if actual_volatility
            else np.nan
        ),
        "change_direction_accuracy": direction_accuracy,
    }


def build_quality_report(predictions):
    """Build comparable model and baseline metrics across business levels."""
    rows = []

    def append_metrics(frame, level, entity_id, entity_name=None):
        for method, prediction_column in PREDICTION_COLUMNS.items():
            interval_available = method == "lightgbm"
            metrics = calculate_metrics(
                frame["qty"],
                frame[prediction_column],
                lower=frame["pred_p10"] if interval_available else None,
                upper=frame["pred_p90"] if interval_available else None,
            )
            change_metrics = calculate_change_metrics(
                frame,
                prediction_column,
            )
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
                    **change_metrics,
                    "interval_status": (
                        "conformalized_p10_p90"
                        if interval_available
                        else "not_applicable"
                    ),
                }
            )

    append_metrics(predictions, "overall", "all", "全部")
    for origin, frame in predictions.groupby("forecast_origin", sort=True):
        append_metrics(frame, "window", origin.date().isoformat(), origin.date().isoformat())

    levels = [
        ("store", "store_id", "store_name"),
        ("category", "category", "category"),
        ("product_type", "product_type", "product_type"),
        ("sku", "product_id", "product_id"),
    ]
    for level, id_column, name_column in levels:
        for entity_id, frame in predictions.groupby(id_column, sort=True):
            entity_name = frame[name_column].iloc[0]
            append_metrics(frame, level, entity_id, entity_name)

    report = pd.DataFrame(rows)
    overall = report[report["level"] == "overall"].set_index("method")
    baseline_methods = [
        method for method in overall.index if method.startswith("baseline_")
    ]
    best_baseline_mae = overall.loc[baseline_methods, "mae"].min()
    best_baseline_wape = overall.loc[baseline_methods, "wape"].min()
    model_is_candidate = bool(
        overall.loc["lightgbm", "mae"] < best_baseline_mae
        and overall.loc["lightgbm", "wape"] < best_baseline_wape
    )
    report["candidate_eligible"] = False
    report.loc[
        (report["level"] == "overall") & (report["method"] == "lightgbm"),
        "candidate_eligible",
    ] = model_is_candidate
    lightgbm_metrics = overall.loc["lightgbm"]
    fusion_metrics = overall.loc["lightgbm_kalman"]
    window_metrics = report[report["level"] == "window"].pivot(
        index="entity_id",
        columns="method",
        values="mae",
    )
    fusion_window_wins = int(
        (
            window_metrics["lightgbm_kalman"]
            < window_metrics["lightgbm"]
        ).sum()
    )
    fusion_is_candidate = bool(
        fusion_metrics["mae"] < lightgbm_metrics["mae"]
        and fusion_metrics["wape"] < lightgbm_metrics["wape"]
        and abs(fusion_metrics["mean_error"])
        <= abs(lightgbm_metrics["mean_error"])
        and fusion_window_wins >= 3
    )
    report.loc[
        (report["level"] == "overall")
        & (report["method"] == "lightgbm_kalman"),
        "candidate_eligible",
    ] = fusion_is_candidate
    return report, model_is_candidate, fusion_is_candidate, fusion_window_wins


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_rolling_backtest(
    output_dir=REPORT_DIR,
    window_count=BACKTEST_WINDOW_COUNT,
    horizon=PREDICTION_LOOKAHEAD,
):
    df = load_and_preprocess()
    feature_cols = get_feature_cols(df)
    origins = select_backtest_origins(
        df["date"].unique(),
        window_count=window_count,
        horizon=horizon,
    )
    source_hashes = {
        SALES_FILE.name: _sha256(SALES_FILE),
        WEATHER_FILE.name: _sha256(WEATHER_FILE),
    }
    run_id = build_model_run_id(
        source_hashes,
        origins,
        horizon,
        feature_cols,
    )
    all_predictions = []
    all_shap_contributions = []
    all_scenarios = []
    training_runs = []

    for origin in origins:
        print(f"\nBacktest origin {origin.date()} ({horizon} days)")
        (
            model,
            quantile_models,
            fit_df,
            validation_df,
            calibration_df,
        ) = _fit_model_for_origin(
            df,
            feature_cols,
            origin,
            horizon,
        )
        calibration_start = calibration_df["date"].min()
        calibration_forecast_raw = recursive_forecast(
            model,
            df,
            feature_cols,
            calibration_start,
            horizon,
            quantile_models=quantile_models,
        )
        store_bias_factors = calculate_store_bias_factors(
            calibration_forecast_raw
        )
        calibration_forecast = recursive_forecast(
            model,
            df,
            feature_cols,
            calibration_start,
            horizon,
            quantile_models=quantile_models,
            store_bias_factors=store_bias_factors,
        )
        calibration_kalman = kalman_forecast(
            df,
            calibration_start,
            horizon,
        )
        calibration_forecast = apply_kalman_fusion(
            calibration_forecast,
            calibration_kalman,
            weight=0,
        )
        (
            kalman_fusion_weight,
            kalman_fusion_candidates,
        ) = select_kalman_fusion_weight(calibration_forecast)
        calibration_forecast = apply_kalman_fusion(
            calibration_forecast.drop(
                columns=[
                    "pred_kalman",
                    "kalman_fusion_weight",
                    "pred_lightgbm_kalman",
                ]
            ),
            calibration_kalman,
            weight=kalman_fusion_weight,
        )
        conformal_adjustment = calculate_conformal_adjustment(calibration_forecast)
        forecast, shap_contributions, scenarios = recursive_forecast(
            model,
            df,
            feature_cols,
            origin,
            horizon,
            quantile_models=quantile_models,
            store_bias_factors=store_bias_factors,
            model_run_id=run_id,
            include_explanations=True,
            include_scenarios=True,
        )
        forecast = apply_kalman_fusion(
            forecast,
            kalman_forecast(df, origin, horizon),
            weight=kalman_fusion_weight,
        )
        forecast = apply_conformal_adjustment(forecast, conformal_adjustment)
        forecast = add_origin_safe_baselines(forecast, df, origin)
        forecast["forecast_origin"] = origin
        forecast["horizon_day"] = (forecast["date"] - origin).dt.days + 1
        forecast["prediction_error"] = forecast["qty"] - forecast["pred_lightgbm"]
        forecast["absolute_error"] = forecast["prediction_error"].abs()
        all_predictions.append(forecast)
        all_shap_contributions.append(shap_contributions)
        all_scenarios.append(scenarios)
        training_runs.append(
            {
                "forecast_origin": origin.date().isoformat(),
                "fit_start": fit_df["date"].min().date().isoformat(),
                "fit_end": fit_df["date"].max().date().isoformat(),
                "early_stopping_start": validation_df["date"].min().date().isoformat(),
                "early_stopping_end": validation_df["date"].max().date().isoformat(),
                "calibration_start": calibration_df["date"].min().date().isoformat(),
                "calibration_end": calibration_df["date"].max().date().isoformat(),
                "calibration_sample_count": len(calibration_forecast),
                "store_bias_factors": store_bias_factors.to_dict(
                    orient="records"
                ),
                "kalman_fusion_weight": kalman_fusion_weight,
                "kalman_fusion_candidates": kalman_fusion_candidates,
                "conformal_adjustment": conformal_adjustment,
                "best_iteration": int(model.best_iteration_),
                "quantile_best_iterations": {
                    column: int(quantile_model.best_iteration_)
                    for column, quantile_model in quantile_models.items()
                },
            }
        )

    predictions = pd.concat(all_predictions, ignore_index=True)
    shap_contributions = pd.concat(all_shap_contributions, ignore_index=True)
    scenarios = pd.concat(all_scenarios, ignore_index=True)
    (
        quality,
        model_is_candidate,
        fusion_is_candidate,
        fusion_window_wins,
    ) = build_quality_report(predictions)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "rolling_backtest_predictions.csv"
    shap_path = output_dir / "rolling_backtest_shap.csv.gz"
    scenario_path = output_dir / "rolling_backtest_scenarios.csv.gz"
    quality_path = output_dir / "rolling_backtest_quality.csv"
    metadata_path = output_dir / "rolling_backtest_run.json"
    prediction_columns = [
        "prediction_id",
        "model_run_id",
        "forecast_origin",
        "horizon_day",
        "date",
        "store_id",
        "store_name",
        "product_id",
        "category",
        "product_type",
        "qty",
        "prediction_error",
        "absolute_error",
        "pred_lightgbm_raw",
        "pred_lightgbm_model",
        *PREDICTION_COLUMNS.values(),
        "kalman_fusion_weight",
        "postprocess_adjustment",
        "store_bias_factor",
        "store_bias_adjustment",
        *QUANTILE_LEVELS,
        "pred_p10_uncalibrated",
        "pred_p90_uncalibrated",
        "conformal_adjustment",
        "base_value",
        "shap_sum",
        "shap_additivity_error",
    ]
    predictions[prediction_columns].to_csv(prediction_path, index=False)
    shap_contributions.to_csv(
        shap_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    scenarios.to_csv(
        scenario_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    quality.to_csv(quality_path, index=False)

    metadata = {
        "run_id": run_id,
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": generated_at,
        "model_name": "Global LightGBM (lightgbm.LGBMRegressor)",
        "candidate_status": "candidate" if model_is_candidate else "rejected",
        "candidate_rule": (
            "Aggregate LightGBM MAE and WAPE must both beat the best origin-safe baseline"
        ),
        "selected_point_forecast": (
            "pred_lightgbm_kalman"
            if fusion_is_candidate
            else "pred_lightgbm"
        ),
        "kalman_candidate_status": (
            "promoted" if fusion_is_candidate else "experimental"
        ),
        "kalman_promotion_rule": (
            "Fusion must improve aggregate MAE and WAPE, not increase absolute "
            "mean error, and beat LightGBM MAE in at least three of five windows"
        ),
        "kalman_fusion_window_wins": fusion_window_wins,
        "kalman": {
            "method": (
                "Per-store-SKU damped local-linear-trend Kalman filter with "
                "shrunk weekday seasonal offsets"
            ),
            "parameters": KALMAN_CONFIG,
            "candidate_weights": KALMAN_FUSION_WEIGHTS,
            "weight_selection": (
                "Minimum MAE on the seven-day pre-origin calibration window; "
                "ties prefer the lower Kalman weight"
            ),
            "future_label_access": False,
        },
        "data_range": {
            "start": df["date"].min().date().isoformat(),
            "end": df["date"].max().date().isoformat(),
        },
        "forecast_horizon_days": horizon,
        "backtest_window_count": window_count,
        "origins": [origin.date().isoformat() for origin in origins],
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "parameters": LIGHTGBM_PARAMS,
        "quantile_levels": QUANTILE_LEVELS,
        "store_bias_calibration": {
            "method": "store aggregate actual-to-predicted ratio",
            "calibration_window_days": horizon,
            "shrinkage_toward_one": STORE_BIAS_SHRINKAGE,
            "minimum_factor": STORE_BIAS_FACTOR_MIN,
            "maximum_factor": STORE_BIAS_FACTOR_MAX,
            "fallback_for_unseen_store": 1.0,
            "applies_to": [
                "pred_lightgbm",
                *QUANTILE_LEVELS.keys(),
            ],
        },
        "interval_calibration": {
            "method": "split conformalized quantile regression",
            "target_coverage": INTERVAL_TARGET_COVERAGE,
            "calibration_window_days": horizon,
            "score": "max(pred_p10 - actual, actual - pred_p90)",
            "finite_sample_correction": True,
        },
        "quantile_parameters": {
            "objective": "quantile",
            "base_parameters": {
                key: value
                for key, value in LIGHTGBM_PARAMS.items()
                if key != "objective"
            },
        },
        "training_runs": training_runs,
        "source_hashes": source_hashes,
        "prediction_explanations": {
            "method": "TreeSHAP",
            "additivity_target": "pred_lightgbm_raw",
            "additivity_tolerance": SHAP_ADDITIVITY_TOLERANCE,
            "row_count": len(predictions),
            "contribution_count": len(shap_contributions),
            "features_per_prediction": len(feature_cols),
            "maximum_absolute_additivity_error": float(
                predictions["shap_additivity_error"].abs().max()
            ),
            "postprocessing": (
                "pred_lightgbm_model = max(0, pred_lightgbm_raw); "
                "pred_lightgbm = pred_lightgbm_model * the pre-origin store "
                "bias factor; "
                "postprocess_adjustment and store_bias_adjustment are stored "
                "separately from SHAP"
            ),
        },
        "scenario_simulation": {
            "method": "controlled LightGBM re-scoring",
            "baseline": {
                "temperature_id": SCENARIO_BASELINE[0],
                "rain_id": SCENARIO_BASELINE[1],
                "holiday_id": SCENARIO_BASELINE[2],
            },
            "temperature_options": SCENARIO_TEMPERATURES,
            "rain_options": SCENARIO_RAIN,
            "holiday_options": SCENARIO_HOLIDAYS,
            "combination_count": (
                len(SCENARIO_TEMPERATURES)
                * len(SCENARIO_RAIN)
                * len(SCENARIO_HOLIDAYS)
            ),
            "row_count": len(scenarios),
            "scope": (
                "Re-scores the selected prediction while holding historical "
                "demand, price, other weather inputs, and store calibration fixed."
            ),
            "causal": False,
        },
        "artifacts": {
            "predictions": str(prediction_path.relative_to(DATA_DIR)),
            "shap_contributions": str(shap_path.relative_to(DATA_DIR)),
            "scenario_responses": str(scenario_path.relative_to(DATA_DIR)),
            "quality_report": str(quality_path.relative_to(DATA_DIR)),
        },
        "limitations": [
            "P10/P90 use split conformal calibration for marginal 80% coverage; P50 remains the raw median quantile.",
            "Conformal coverage is marginal over the calibration population and is not guaranteed for each store, SKU, or horizon day.",
            "Store bias factors use only the seven days before each forecast origin, are shrunk toward one, and may not persist under abrupt demand shifts.",
            "Kalman parameters are fixed; only the LightGBM/Kalman blend weight is selected on each pre-origin calibration window.",
            "Kalman forecasts are observational state estimates, not causal demand estimates.",
            "This retrospective backtest covers five fixed seven-day origins only.",
            "TreeSHAP explains model output and feature association, not causal business impact.",
            "What-if results are controlled model responses, not causal estimates or operational guarantees.",
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    overall = quality[quality["level"] == "overall"][
        [
            "method",
            "mae",
            "rmse",
            "wape",
            "mean_error",
            "r2",
            "within_10pct",
            "within_20pct",
            "change_volatility_ratio",
            "change_direction_accuracy",
            "interval_coverage",
            "interval_mean_width",
        ]
    ]
    print("\nAggregate rolling-backtest quality:")
    print(overall.to_string(index=False))
    print(f"\nModel candidate status: {metadata['candidate_status']}")
    return predictions, quality, metadata


if __name__ == "__main__":
    run_rolling_backtest()
