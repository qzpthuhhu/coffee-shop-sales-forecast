"""Production pipeline — direct multi-step Tweedie 1.3 + sparse labels.

Provides quantile intervals, TreeSHAP explanations, and what-if scenarios.
Outputs the same five files that export_dashboard_data.py reads, enabling
a drop-in production switch from the recursive LightGBM pipeline.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from backtest_pipeline import (
    BACKTEST_WINDOW_COUNT,
    ENTITY_KEYS,
    QUANTILE_LEVELS,
    REPORT_DIR,
    SCENARIO_TEMPERATURES,
    SCENARIO_RAIN,
    SCENARIO_HOLIDAYS,
    SCENARIO_BASELINE,
    SHAP_ADDITIVITY_TOLERANCE,
    STORE_BIAS_FACTOR_MAX,
    STORE_BIAS_FACTOR_MIN,
    STORE_BIAS_SHRINKAGE,
    add_prediction_identity,
    add_origin_safe_baselines,
    apply_conformal_adjustment,
    apply_store_bias_calibration,
    calculate_change_metrics,
    calculate_conformal_adjustment,
    calculate_metrics,
    calculate_store_bias_factors,
    select_backtest_origins,
)
from first_round_model_experiments import (
    _fit_lightgbm,
    build_direct_feature_frames,
    get_direct_feature_cols,
    split_direct_horizon_frame,
)
from fourth_round_sparse_features import (
    _attach_sparsity_features,
    _compute_sku_sparsity,
)
from predict_pipeline import (
    DATA_DIR,
    LIGHTGBM_PARAMS,
    PREDICTION_LOOKAHEAD,
    STORE_NAME_MAP,
    load_and_preprocess,
)


PRODUCTION_METHOD = "direct_tweedie_1_3_sparse"
PIPELINE_VERSION = "direct-tweedie-sparse-v1"
TWEEDIE_PARAMS = {"objective": "tweedie", "tweedie_variance_power": 1.3}
QUALITY_METHODS = {
    "lightgbm": "pred_lightgbm",
    "baseline_yesterday": "pred_baseline_yesterday",
    "baseline_last_week": "pred_baseline_last_week",
    "baseline_rolling_28d": "pred_baseline_rolling_28d",
}


def _train_models_for_origin(fit, validation, feature_cols):
    """Train Tweedie point model + 3 quantile models."""
    model = _fit_lightgbm(fit, validation, feature_cols, TWEEDIE_PARAMS)

    quantile_models = {}
    for label, alpha in QUANTILE_LEVELS.items():
        qm = LGBMRegressor(
            objective="quantile", alpha=alpha,
            n_estimators=1000, learning_rate=0.03, num_leaves=63,
            max_depth=-1, min_child_samples=20, subsample=0.9,
            subsample_freq=1, colsample_bytree=0.9, reg_alpha=0.05,
            reg_lambda=0.1, random_state=42, n_jobs=1, verbosity=-1,
        )
        qm.fit(
            fit[feature_cols], fit["qty"],
            eval_X=validation[feature_cols], eval_y=validation["qty"],
            eval_metric="quantile",
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        quantile_models[label] = qm
    return model, quantile_models


def _shap_contributions(model, frame, feature_cols, prediction_ids):
    """Return signed TreeSHAP contributions on the response scale."""
    explainer = shap.TreeExplainer(model)
    vals = np.asarray(explainer.shap_values(frame[feature_cols]), dtype=float)
    if vals.ndim != 2 or vals.shape != (len(frame), len(feature_cols)):
        raise ValueError("SHAP shape mismatch")

    exp = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    if exp.size != 1:
        raise ValueError("Expected scalar base value")
    raw_base = exp.item()

    raw_scores = np.asarray(
        model.predict(frame[feature_cols], raw_score=True),
        dtype=float,
    )
    response = np.asarray(model.predict(frame[feature_cols]), dtype=float)
    raw_delta = raw_scores - raw_base
    base = float(np.exp(raw_base))
    scale = np.full(len(raw_delta), base, dtype=float)
    nonzero = np.abs(raw_delta) > 1e-12
    scale[nonzero] = (
        base * np.expm1(raw_delta[nonzero]) / raw_delta[nonzero]
    )
    vals = vals * scale[:, None]
    shap_sums = vals.sum(axis=1)
    additivity_errors = base + shap_sums - response
    max_error = float(np.max(np.abs(additivity_errors), initial=0))
    if max_error > SHAP_ADDITIVITY_TOLERANCE:
        raise ValueError(
            f"TreeSHAP additivity failed: max error={max_error:.8f}"
        )

    ids = prediction_ids.reset_index(drop=True)
    n = len(ids)
    contrib = pd.DataFrame({
        "prediction_id": np.repeat(ids.to_numpy(), len(feature_cols)),
        "feature": np.tile(np.asarray(feature_cols, dtype=object), n),
        "feature_value": frame[feature_cols].to_numpy().reshape(-1),
        "shap_contribution": vals.reshape(-1),
    })
    contrib["direction"] = np.select(
        [contrib["shap_contribution"] > 0, contrib["shap_contribution"] < 0],
        ["increase", "decrease"], default="neutral",
    )
    summary = pd.DataFrame({
        "prediction_id": ids,
        "base_value": np.full(n, base),
        "shap_sum": shap_sums,
        "shap_additivity_error": additivity_errors,
    })
    return contrib.merge(summary, on="prediction_id", how="left")


def _scenarios(model, current, feature_cols):
    """Weather + holiday counterfactuals for every prediction row."""
    required = {
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "precipitation_sum", "is_rain", "is_snow", "is_clear",
        "is_holiday_detailed", "is_pre_holiday_1d", "is_pre_holiday_3d",
        "is_post_holiday_1d", "is_long_weekend", "holiday_impact_code",
        "is_hot", "is_cold",
    }
    missing = required.difference(current.columns)
    if missing:
        raise ValueError(f"Missing scenario columns: {sorted(missing)}")

    rows = []
    idx = 0
    for temp in SCENARIO_TEMPERATURES:
        for rain in SCENARIO_RAIN:
            for hol in SCENARIO_HOLIDAYS:
                s = current.copy()
                dt = temp["delta_c"]
                for col in ("temperature_2m_max", "temperature_2m_min", "temperature_2m_mean"):
                    s[col] = s[col] + dt
                s["is_hot"] = (s["temperature_2m_max"] > 25).astype(int)
                s["is_cold"] = (s["temperature_2m_max"] < 5).astype(int)

                precip = rain["precipitation_mm"]
                if precip is not None:
                    s["precipitation_sum"] = precip
                    s["is_rain"] = int(precip > 0)
                    s["is_snow"] = 0
                    s["is_clear"] = int(precip == 0)

                if hol["id"] != "observed":
                    for col in ("is_holiday_detailed", "is_pre_holiday_1d",
                                "is_pre_holiday_3d", "is_post_holiday_1d",
                                "is_long_weekend"):
                        s[col] = 0
                    s["holiday_impact_code"] = 0
                    if hol["id"] == "long_weekend":
                        s["is_holiday_detailed"] = 1
                        s["is_long_weekend"] = 1
                        s["holiday_impact_code"] = 2

                raw = np.asarray(model.predict(s[feature_cols]), dtype=float)
                sim = np.clip(raw, a_min=0, a_max=None) * s.get(
                    "store_bias_factor", pd.Series(1.0, index=s.index)
                ).to_numpy(dtype=float)

                rows.append(pd.DataFrame({
                    "prediction_id": s["prediction_id"],
                    "scenario_index": idx,
                    "temperature_id": temp["id"],
                    "rain_id": rain["id"],
                    "holiday_id": hol["id"],
                    "temperature_2m_mean": s["temperature_2m_mean"],
                    "precipitation_sum": s["precipitation_sum"],
                    "is_holiday_detailed": s["is_holiday_detailed"],
                    "is_long_weekend": s["is_long_weekend"],
                    "simulated_prediction": sim,
                }))
                idx += 1

    result = pd.concat(rows, ignore_index=True)
    baseline = result[
        (result["temperature_id"] == SCENARIO_BASELINE[0])
        & (result["rain_id"] == SCENARIO_BASELINE[1])
        & (result["holiday_id"] == SCENARIO_BASELINE[2])
    ].sort_values("prediction_id")
    expected = current.sort_values("prediction_id")
    np.testing.assert_allclose(
        baseline["simulated_prediction"], expected["pred_lightgbm"],
        atol=1e-6, rtol=0,
    )
    return result


def _prediction_rows(test, raw_pred, method, origin, feature_cols, horizon_col="forecast_horizon"):
    """Assemble forecast rows with all feature columns kept."""
    core = ["date", "store_id", "store_name", "product_id", "category", "product_type", "qty"]
    extra = [c for c in test.columns if c in feature_cols and c not in core]
    result = test[core + extra].copy()
    result["store_name"] = result["store_id"].map(STORE_NAME_MAP)
    result["forecast_origin"] = origin
    result["horizon_day"] = test[horizon_col].to_numpy(dtype=int)
    result["method"] = method
    result["pred_lightgbm_raw"] = np.asarray(raw_pred, dtype=float)
    result["pred_lightgbm_model"] = np.clip(result["pred_lightgbm_raw"], a_min=0, a_max=None)
    result["postprocess_adjustment"] = result["pred_lightgbm_model"] - result["pred_lightgbm_raw"]
    result["pred_lightgbm"] = result["pred_lightgbm_model"]
    return result


def _build_quality(predictions):
    """Build the dashboard-compatible report without obsolete model rows."""
    rows = []

    def append_metrics(frame, level, entity_id, entity_name=None):
        for method, prediction_column in QUALITY_METHODS.items():
            has_interval = method == "lightgbm"
            rows.append({
                "level": level,
                "entity_id": str(entity_id),
                "entity_name": str(
                    entity_name if entity_name is not None else entity_id
                ),
                "method": method,
                "validation_start": frame["date"].min().date().isoformat(),
                "validation_end": frame["date"].max().date().isoformat(),
                **calculate_metrics(
                    frame["qty"],
                    frame[prediction_column],
                    lower=frame["pred_p10"] if has_interval else None,
                    upper=frame["pred_p90"] if has_interval else None,
                ),
                **calculate_change_metrics(frame, prediction_column),
                "interval_status": (
                    "conformalized_p10_p90"
                    if has_interval
                    else "not_applicable"
                ),
            })

    append_metrics(predictions, "overall", "all", "全部")
    for origin, frame in predictions.groupby("forecast_origin", sort=True):
        label = pd.Timestamp(origin).date().isoformat()
        append_metrics(frame, "window", label, label)
    for level, id_column, name_column in (
        ("store", "store_id", "store_name"),
        ("category", "category", "category"),
        ("product_type", "product_type", "product_type"),
        ("sku", "product_id", "product_id"),
    ):
        for entity_id, frame in predictions.groupby(id_column, sort=True):
            append_metrics(
                frame,
                level,
                entity_id,
                frame[name_column].iloc[0],
            )

    report = pd.DataFrame(rows)
    overall = report[report["level"] == "overall"].set_index("method")
    baselines = overall.loc[
        [method for method in overall.index if method.startswith("baseline_")]
    ]
    candidate = bool(
        overall.loc["lightgbm", "mae"] < baselines["mae"].min()
        and overall.loc["lightgbm", "wape"] < baselines["wape"].min()
    )
    report["candidate_eligible"] = False
    report.loc[
        (report["level"] == "overall")
        & (report["method"] == "lightgbm"),
        "candidate_eligible",
    ] = candidate
    return report, candidate


def _build_run_id(source_hashes, origins, feature_cols):
    contract = {
        "pipeline_version": PIPELINE_VERSION,
        "source_hashes": source_hashes,
        "origins": [origin.date().isoformat() for origin in origins],
        "horizon": PREDICTION_LOOKAHEAD,
        "features": feature_cols,
        "point_parameters": {**LIGHTGBM_PARAMS, **TWEEDIE_PARAMS},
        "quantile_levels": QUANTILE_LEVELS,
    }
    digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"sparse_production_{digest}"


def run_production_pipeline():
    print("=" * 60)
    print("Production Pipeline - Direct Tweedie + Sparse Labels")
    print("=" * 60)

    df = load_and_preprocess()
    df["store_name"] = df["store_id"].map(STORE_NAME_MAP)
    origins = select_backtest_origins(df["date"].unique())
    cutoff = min(origins) - pd.Timedelta(days=1)
    print(f"\nOrigins: {[o.date() for o in origins]}, cutoff: {cutoff.date()}")

    sparsity = _compute_sku_sparsity(df, cutoff)
    n_sparse = int((sparsity["zero_rate"].gt(0.5) | sparsity["adi"].gt(2.0)).sum())
    print(f"Sparse SKUs: {n_sparse} / {len(sparsity)}")

    direct_frames = build_direct_feature_frames(df)
    base_cols = get_direct_feature_cols(direct_frames[1])
    sparse_cols = ["is_sparse", "zero_rate", "adi", "mean_qty"]
    feature_cols = list(dict.fromkeys([*base_cols, *sparse_cols]))
    print(f"Features: {len(feature_cols)} ({len(base_cols)} + {len(sparse_cols)})")
    augmented = {
        horizon: _attach_sparsity_features(frame, sparsity)
        for horizon, frame in direct_frames.items()
    }
    source_files = [
        DATA_DIR / "Coffee Shop Sales.xlsx",
        DATA_DIR / "nyc_weather_2023_h1.csv",
    ]
    source_hashes = {
        source.name: hashlib.sha256(source.read_bytes()).hexdigest()
        for source in source_files
        if source.exists()
    }
    run_id = _build_run_id(source_hashes, origins, feature_cols)

    all_forecasts, all_shap, all_scenarios = [], [], []
    training_runs = []

    for origin in origins:
        print(f"\nOrigin {origin.date()}")
        fit_rows, val_rows, cal_rows, test_rows = [], [], [], []
        for h, aug in augmented.items():
            fit, val, cal, test = split_direct_horizon_frame(aug, base_cols, origin, h)
            fit_rows.append(fit)
            val_rows.append(val)
            cal_rows.append(cal)
            test_rows.append(test)

        fit = pd.concat(fit_rows, ignore_index=True)
        val = pd.concat(val_rows, ignore_index=True)
        cal = pd.concat(cal_rows, ignore_index=True)
        test = pd.concat(test_rows, ignore_index=True)

        print(f"  Training (fit={len(fit)}, val={len(val)})...")
        model, quantile_models = _train_models_for_origin(fit, val, feature_cols)

        cal_pred = _prediction_rows(cal, model.predict(cal[feature_cols]),
                                    PRODUCTION_METHOD, origin, feature_cols)
        for column, quantile_model in quantile_models.items():
            cal_pred[column] = np.clip(
                quantile_model.predict(cal[feature_cols]),
                a_min=0,
                a_max=None,
            )
        cal_pred[list(QUANTILE_LEVELS)] = np.sort(
            cal_pred[list(QUANTILE_LEVELS)].to_numpy(dtype=float),
            axis=1,
        )
        factors = calculate_store_bias_factors(cal_pred)
        cal_pred = apply_store_bias_calibration(cal_pred, factors)
        conformal_adjustment = calculate_conformal_adjustment(cal_pred)

        test_pred = _prediction_rows(test, model.predict(test[feature_cols]),
                                     PRODUCTION_METHOD, origin, feature_cols)
        for column, quantile_model in quantile_models.items():
            test_pred[column] = np.clip(
                quantile_model.predict(test[feature_cols]),
                a_min=0,
                a_max=None,
            )
        test_pred[list(QUANTILE_LEVELS)] = np.sort(
            test_pred[list(QUANTILE_LEVELS)].to_numpy(dtype=float),
            axis=1,
        )
        test_pred = apply_store_bias_calibration(test_pred, factors)
        test_pred = apply_conformal_adjustment(
            test_pred,
            conformal_adjustment,
        )
        test_pred = add_origin_safe_baselines(test_pred, df, origin)
        test_pred = add_prediction_identity(test_pred, run_id, origin)

        print(f"  SHAP ({len(test_pred)} preds x {len(feature_cols)} features)...")
        shap_contribs = _shap_contributions(model, test_pred, feature_cols, test_pred["prediction_id"])
        shap_summary = (
            shap_contribs.groupby("prediction_id")
            .agg(base_value=("base_value", "first"),
                 shap_sum=("shap_sum", "first"),
                 shap_additivity_error=("shap_additivity_error", "first"))
            .reset_index()
        )
        test_pred = test_pred.merge(shap_summary, on="prediction_id", how="left")
        all_shap.append(shap_contribs)

        print(f"  Scenarios...")
        all_scenarios.append(_scenarios(model, test_pred, feature_cols))

        all_forecasts.append(test_pred)
        training_runs.append({
            "forecast_origin": origin.date().isoformat(),
            "method": PRODUCTION_METHOD,
            "feature_count": len(feature_cols),
            "best_iteration": int(model.best_iteration_),
            "quantile_best_iterations": {
                column: int(quantile_model.best_iteration_)
                for column, quantile_model in quantile_models.items()
            },
            "conformal_adjustment": float(conformal_adjustment),
            "store_bias_factors": factors[["store_id", "store_bias_factor"]].to_dict(orient="records"),
        })

    forecasts = pd.concat(all_forecasts, ignore_index=True)
    forecasts["prediction_error"] = forecasts["qty"] - forecasts["pred_lightgbm"]
    forecasts["absolute_error"] = forecasts["prediction_error"].abs()
    shap_contributions = pd.concat(all_shap, ignore_index=True)
    scenarios = pd.concat(all_scenarios, ignore_index=True)
    quality, model_is_candidate = _build_quality(forecasts)

    q_row = quality[
        (quality["level"] == "overall")
        & (quality["method"] == "lightgbm")
    ].iloc[0]
    print(f"\nProduction: MAE={q_row['mae']:.4f}  WAPE={q_row['wape']:.4f}  "
          f"R2={q_row['r2']:.4f}  Coverage={q_row['interval_coverage']:.4f}")

    meta = {
        "run_id": run_id,
        "pipeline_version": PIPELINE_VERSION,
        "pipeline": "production_sparse",
        "method": PRODUCTION_METHOD,
        "model_name": "Global LightGBM (direct Tweedie + sparse labels)",
        "candidate_status": "candidate" if model_is_candidate else "rejected",
        "candidate_rule": (
            "Aggregate model MAE and WAPE must both beat the best "
            "origin-safe baseline"
        ),
        "selected_point_forecast": "pred_lightgbm",
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "sparse_sku_count": n_sparse,
        "data_range": {
            "start": df["date"].min().date().isoformat(),
            "end": df["date"].max().date().isoformat(),
        },
        "forecast_horizon_days": PREDICTION_LOOKAHEAD,
        "origins": [o.date().isoformat() for o in origins],
        "backtest_window_count": len(origins),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prediction_count": len(forecasts),
        "overall": {"mae": q_row["mae"], "wape": q_row["wape"],
                    "rmse": q_row["rmse"], "r2": q_row["r2"]},
        "coverage_p10_p90": q_row["interval_coverage"],
        "training_runs": training_runs,
        "source_hashes": source_hashes,
        "quantile_levels": QUANTILE_LEVELS,
        "parameters": {**LIGHTGBM_PARAMS, **TWEEDIE_PARAMS},
        "store_bias_calibration": {
            "method": "store aggregate actual-to-predicted ratio",
            "calibration_window_days": PREDICTION_LOOKAHEAD,
            "shrinkage_toward_one": STORE_BIAS_SHRINKAGE,
            "minimum_factor": STORE_BIAS_FACTOR_MIN,
            "maximum_factor": STORE_BIAS_FACTOR_MAX,
            "fallback_for_unseen_store": 1.0,
            "applies_to": ["pred_lightgbm", *QUANTILE_LEVELS],
        },
        "interval_calibration": {
            "method": "split conformalized quantile regression",
            "target_coverage": 0.80,
            "calibration_window_days": PREDICTION_LOOKAHEAD,
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
        "prediction_explanations": {
            "method": "TreeSHAP with exact Tweedie log-link transformation",
            "additivity_target": "pred_lightgbm_raw",
            "unit": "cups",
            "additivity_tolerance": SHAP_ADDITIVITY_TOLERANCE,
            "row_count": len(forecasts),
            "contribution_count": len(shap_contributions),
            "features_per_prediction": len(feature_cols),
            "maximum_absolute_additivity_error": float(
                forecasts["shap_additivity_error"].abs().max()
            ),
            "postprocessing": (
                "pred_lightgbm_model = max(0, pred_lightgbm_raw); "
                "pred_lightgbm = pred_lightgbm_model * the pre-origin store "
                "bias factor; adjustments are stored separately from SHAP"
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
                "Re-scores one prediction while holding historical demand, "
                "price, sparse labels, other weather inputs, and store "
                "calibration fixed."
            ),
            "causal": False,
        },
        "artifacts": {
            "predictions": "reports/model_quality/rolling_backtest_predictions.csv",
            "shap_contributions": "reports/model_quality/rolling_backtest_shap.csv.gz",
            "scenario_responses": "reports/model_quality/rolling_backtest_scenarios.csv.gz",
            "quality_report": "reports/model_quality/rolling_backtest_quality.csv",
        },
        "limitations": [
            "P10/P90 use split conformal calibration for marginal 80% coverage.",
            "Coverage is not guaranteed for each store, SKU, or horizon day.",
            "Sparse labels use only data before the first backtest origin.",
            "The retrospective backtest covers five fixed seven-day origins.",
            "TreeSHAP explains model association, not causal business impact.",
            "What-if outputs are controlled model responses, not causal estimates.",
        ],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(REPORT_DIR / "rolling_backtest_predictions.csv", index=False)
    quality.to_csv(REPORT_DIR / "rolling_backtest_quality.csv", index=False)
    gzip_options = {"method": "gzip", "compresslevel": 6, "mtime": 0}
    shap_contributions.to_csv(
        REPORT_DIR / "rolling_backtest_shap.csv.gz",
        index=False,
        compression=gzip_options,
    )
    scenarios.to_csv(
        REPORT_DIR / "rolling_backtest_scenarios.csv.gz",
        index=False,
        compression=gzip_options,
    )
    (REPORT_DIR / "rolling_backtest_run.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print(f"\nWrote 5 artifacts to {REPORT_DIR}")
    print(f"  predictions:   {len(forecasts):,} rows")
    print(f"  shap:          {len(shap_contributions):,} contributions")
    print(f"  scenarios:     {len(scenarios):,} rows")
    return forecasts, shap_contributions, scenarios, meta


if __name__ == "__main__":
    run_production_pipeline()
