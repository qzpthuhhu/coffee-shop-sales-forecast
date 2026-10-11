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
PIPELINE_VERSION = "rolling-backtest-v3-prediction-shap"
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
    "baseline_yesterday": "pred_baseline_yesterday",
    "baseline_last_week": "pred_baseline_last_week",
    "baseline_rolling_28d": "pred_baseline_rolling_28d",
}
QUANTILE_LEVELS = {
    "pred_p10": 0.10,
    "pred_p50": 0.50,
    "pred_p90": 0.90,
}
SHAP_ADDITIVITY_TOLERANCE = 1e-6


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
    """Fit point and quantile models using data strictly before origin."""
    internal_validation_start = origin - pd.Timedelta(days=horizon)
    fit_df = df[df["date"] < internal_validation_start].dropna(
        subset=feature_cols + ["qty"]
    )
    validation_df = df[
        (df["date"] >= internal_validation_start) & (df["date"] < origin)
    ].dropna(subset=feature_cols + ["qty"])
    if fit_df.empty or validation_df.empty:
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
    return model, quantile_models, fit_df, validation_df


def recursive_forecast(
    model,
    df,
    feature_cols,
    origin,
    horizon,
    quantile_models=None,
    model_run_id=None,
    include_explanations=False,
):
    """Forecast one day at a time without reading labels at or after origin."""
    if include_explanations and not model_run_id:
        raise ValueError("model_run_id is required when explanations are enabled")

    forecast_dates = pd.date_range(origin, periods=horizon, freq="D")
    state = df[["date", *ENTITY_KEYS, "qty"]].copy()
    state.loc[state["date"] >= origin, "qty"] = np.nan
    forecast_rows = []
    explanation_rows = []

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
    if include_explanations:
        return forecasts, pd.concat(explanation_rows, ignore_index=True)
    return forecasts


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
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else np.nan,
        "within_10pct": float((np.abs(error / (actual + 1)) < 0.10).mean()),
        "within_20pct": float((np.abs(error / (actual + 1)) < 0.20).mean()),
        "interval_coverage": interval_coverage,
        "interval_mean_width": interval_mean_width,
        "interval_median_width": interval_median_width,
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
                    "interval_status": (
                        "empirical_p10_p90" if interval_available else "not_applicable"
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
    best_baseline_mae = overall.loc[
        overall.index != "lightgbm", "mae"
    ].min()
    best_baseline_wape = overall.loc[
        overall.index != "lightgbm", "wape"
    ].min()
    model_is_candidate = bool(
        overall.loc["lightgbm", "mae"] < best_baseline_mae
        and overall.loc["lightgbm", "wape"] < best_baseline_wape
    )
    report["candidate_eligible"] = False
    report.loc[
        (report["level"] == "overall") & (report["method"] == "lightgbm"),
        "candidate_eligible",
    ] = model_is_candidate
    return report, model_is_candidate


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
    training_runs = []

    for origin in origins:
        print(f"\nBacktest origin {origin.date()} ({horizon} days)")
        model, quantile_models, fit_df, validation_df = _fit_model_for_origin(
            df,
            feature_cols,
            origin,
            horizon,
        )
        forecast, shap_contributions = recursive_forecast(
            model,
            df,
            feature_cols,
            origin,
            horizon,
            quantile_models=quantile_models,
            model_run_id=run_id,
            include_explanations=True,
        )
        forecast = add_origin_safe_baselines(forecast, df, origin)
        forecast["forecast_origin"] = origin
        forecast["horizon_day"] = (forecast["date"] - origin).dt.days + 1
        forecast["prediction_error"] = forecast["qty"] - forecast["pred_lightgbm"]
        forecast["absolute_error"] = forecast["prediction_error"].abs()
        all_predictions.append(forecast)
        all_shap_contributions.append(shap_contributions)
        training_runs.append(
            {
                "forecast_origin": origin.date().isoformat(),
                "fit_start": fit_df["date"].min().date().isoformat(),
                "fit_end": fit_df["date"].max().date().isoformat(),
                "early_stopping_start": validation_df["date"].min().date().isoformat(),
                "early_stopping_end": validation_df["date"].max().date().isoformat(),
                "best_iteration": int(model.best_iteration_),
                "quantile_best_iterations": {
                    column: int(quantile_model.best_iteration_)
                    for column, quantile_model in quantile_models.items()
                },
            }
        )

    predictions = pd.concat(all_predictions, ignore_index=True)
    shap_contributions = pd.concat(all_shap_contributions, ignore_index=True)
    quality, model_is_candidate = build_quality_report(predictions)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "rolling_backtest_predictions.csv"
    shap_path = output_dir / "rolling_backtest_shap.csv.gz"
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
        *PREDICTION_COLUMNS.values(),
        "postprocess_adjustment",
        *QUANTILE_LEVELS,
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
                "pred_lightgbm = max(0, pred_lightgbm_raw); "
                "postprocess_adjustment is stored separately from SHAP"
            ),
        },
        "artifacts": {
            "predictions": str(prediction_path.relative_to(DATA_DIR)),
            "shap_contributions": str(shap_path.relative_to(DATA_DIR)),
            "quality_report": str(quality_path.relative_to(DATA_DIR)),
        },
        "limitations": [
            "P10/P50/P90 are empirical LightGBM quantiles and are not causal or conformally calibrated.",
            "This retrospective backtest covers five fixed seven-day origins only.",
            "TreeSHAP explains model output and feature association, not causal business impact.",
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
            "r2",
            "within_10pct",
            "within_20pct",
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
