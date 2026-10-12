"""Third-round sparse SKU expert model — segment-and-conquer with group-specific Tweedie.

Each (store_id, product_id) is labelled sparse or dense using only data
available before the first forecast origin.  For every rolling origin we
train two separate direct_tweedie_1_3 LightGBM models — one on sparse
SKUs and one on dense SKUs — and route test predictions accordingly.

The baseline is the single-model direct_tweedie_1_3 from round one.
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
    calculate_metrics,
    calculate_store_bias_factors,
    select_backtest_origins,
)
from first_round_model_experiments import (
    IDENTITY_FEATURES,
    OUTPUT_DIR as FIRST_ROUND_DIR,
    _apply_store_calibration,
    _change_metrics,
    _fit_lightgbm,
    _prediction_rows,
    add_direct_history_features,
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

EXPERIMENT_VERSION = "sparse-expert-v1"
OUTPUT_DIR = REPORT_DIR / "third_round"
BASELINE_METHOD = "direct_tweedie_1_3"
SPARSE_METHOD = "sparse_expert"

TWEEPIE_PARAMS = {"objective": "tweedie", "tweedie_variance_power": 1.3}

# ---------- sparsity labelling (strictly before first origin) ----------


def _label_sku_sparsity(
    df: pd.DataFrame,
    first_origin: pd.Timestamp,
    zero_rate_threshold: float = 0.5,
    adi_threshold: float = 2.0,
) -> pd.DataFrame:
    """Return a per-SKU sparsity frame computed from *df* up to *first_origin*-1 day."""
    cutoff = first_origin - pd.Timedelta(days=1)
    history = df[df["date"] <= cutoff].copy()
    g = history.groupby(["store_id", "product_id"])["qty"]

    zero_rate = g.apply(lambda x: (x == 0).mean())

    def _adi(series):
        nonzero_idx = series.reset_index(drop=True)
        nonzero_dates = nonzero_idx[nonzero_idx > 0].index.values
        if len(nonzero_dates) < 2:
            return np.nan
        return np.diff(nonzero_dates).mean()

    adi = g.apply(_adi)
    sparse = pd.DataFrame({"zero_rate": zero_rate, "adi": adi}).reset_index()
    sparse["is_sparse"] = (sparse["zero_rate"] > zero_rate_threshold) | (
        sparse["adi"] > adi_threshold
    )
    return sparse[["store_id", "product_id", "is_sparse", "zero_rate", "adi"]]


# ---------- training helpers ----------


def _train_group_model(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    feature_cols: List[str],
    method: str,
) -> object:
    """Train a direct_tweedie_1_3 model; skip if fit has too few rows."""
    if len(fit) < 10:
        return None
    return _fit_lightgbm(fit, validation, feature_cols, TWEEPIE_PARAMS)


def _predict_with_model(
    model: object,
    frame: pd.DataFrame,
    feature_cols: List[str],
) -> np.ndarray:
    if model is None:
        return np.zeros(len(frame))
    return model.predict(frame[feature_cols])


# ---------- main experiment ----------


def run_third_round_experiments(
    output_dir: str | Path = OUTPUT_DIR,
    window_count: int = BACKTEST_WINDOW_COUNT,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Train sparse/dense expert models and compare against single-model baseline."""

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

    # ---- 2. SKU sparsity labels (from pre-first-origin data only) ----
    sparsity = _label_sku_sparsity(df, first_origin)
    sparse_skus = set(
        zip(
            sparsity[sparsity["is_sparse"]]["store_id"],
            sparsity[sparsity["is_sparse"]]["product_id"],
        )
    )
    n_sparse = len(sparse_skus)
    n_total = len(sparsity)
    print(f"\nSparse SKUs: {n_sparse} / {n_total}")
    print(f"  zero_rate > 0.5: {(sparsity['zero_rate'] > 0.5).sum()}")
    print(f"  ADI > 2:         {(sparsity['adi'] > 2).sum()}")

    # ---- 3. build direct-multi-horizon frames ----
    direct_frames = build_direct_feature_frames(df)
    feature_cols = get_direct_feature_cols(direct_frames[1])

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

    # ---- 5. per-origin training ----
    all_predictions: List[pd.DataFrame] = (
        [baseline] if len(baseline) else []
    )
    training_runs: List[dict] = []

    for origin in origins:
        print(f"\nThird-round origin {origin.date()}")

        # Assemble fit/validation/calibration/test across horizons
        fit_rows: List[pd.DataFrame] = []
        val_rows: List[pd.DataFrame] = []
        cal_rows: List[pd.DataFrame] = []
        test_rows: List[pd.DataFrame] = []
        for horizon_day, direct in direct_frames.items():
            fit, validation, calibration, test = (
                split_direct_horizon_frame(
                    direct, feature_cols, origin, horizon_day
                )
            )
            fit_rows.append(fit)
            val_rows.append(validation)
            cal_rows.append(calibration)
            test_rows.append(test)

        fit = pd.concat(fit_rows, ignore_index=True)
        validation = pd.concat(val_rows, ignore_index=True)
        calibration = pd.concat(cal_rows, ignore_index=True)
        test = pd.concat(test_rows, ignore_index=True)

        # Tag each row with sparsity group
        for frame in (fit, validation, calibration, test):
            frame["_is_sparse"] = frame.apply(
                lambda r: (r["store_id"], r["product_id"]) in sparse_skus,
                axis=1,
            )

        # Split into sparse / dense subsets
        fit_sparse = fit[fit["_is_sparse"]].copy()
        fit_dense = fit[~fit["_is_sparse"]].copy()
        val_sparse = validation[validation["_is_sparse"]].copy()
        val_dense = validation[~validation["_is_sparse"]].copy()

        # Train group-specific models (parallel)
        tasks: List[Tuple[str, pd.DataFrame, pd.DataFrame]] = []
        if len(fit_sparse) >= 10:
            tasks.append(("sparse", fit_sparse, val_sparse))
        if len(fit_dense) >= 10:
            tasks.append(("dense", fit_dense, val_dense))

        with ThreadPoolExecutor(max_workers=2) as executor:
            group_models = dict(
                executor.map(
                    lambda t: (
                        t[0],
                        _train_group_model(t[1], t[2], feature_cols, t[0]),
                    ),
                    tasks,
                )
            )

        # Route predictions
        cal_sparse = calibration[calibration["_is_sparse"]]
        cal_dense = calibration[~calibration["_is_sparse"]]
        test_sparse = test[test["_is_sparse"]]
        test_dense = test[~test["_is_sparse"]]

        cal_preds = []
        test_preds = []

        for group, cal_data, test_data in [
            ("sparse", cal_sparse, test_sparse),
            ("dense", cal_dense, test_dense),
        ]:
            model = group_models.get(group)
            if model is None:
                continue

            cal_pred = _prediction_rows(
                cal_data,
                _predict_with_model(model, cal_data, feature_cols),
                SPARSE_METHOD,
                origin,
                cal_data["forecast_horizon"].to_numpy(),
            )
            test_pred = _prediction_rows(
                test_data,
                _predict_with_model(model, test_data, feature_cols),
                SPARSE_METHOD,
                origin,
                test_data["forecast_horizon"].to_numpy(),
            )
            cal_preds.append(cal_pred)
            test_preds.append(test_pred)

        if cal_preds and test_preds:
            cal_combined = pd.concat(cal_preds, ignore_index=True)
            test_combined = pd.concat(test_preds, ignore_index=True)
            calibrated, factors = _apply_store_calibration(
                cal_combined, test_combined
            )
            all_predictions.append(calibrated)

            for group, model in group_models.items():
                training_runs.append(
                    {
                        "forecast_origin": origin.date().isoformat(),
                        "method": SPARSE_METHOD,
                        "group": group,
                        "fit_end": fit["date"].max().date().isoformat(),
                        "validation_start": validation["date"]
                        .min()
                        .date()
                        .isoformat(),
                        "validation_end": validation["date"]
                        .max()
                        .date()
                        .isoformat(),
                        "best_iteration": (
                            int(model.best_iteration_)
                            if model and hasattr(model, "best_iteration_")
                            else None
                        ),
                        "store_bias_factors": factors[
                            ["store_id", "store_bias_factor"]
                        ].to_dict(orient="records"),
                        "sparse_fit_rows": len(fit_sparse),
                        "dense_fit_rows": len(fit_dense),
                        "sparse_test_rows": len(test_sparse),
                        "dense_test_rows": len(test_dense),
                    }
                )

    # ---- 6. assemble predictions ----
    predictions = pd.concat(all_predictions, ignore_index=True)
    prediction_key = [
        "forecast_origin",
        "date",
        *ENTITY_KEYS,
        "method",
    ]
    if predictions.duplicated(prediction_key).any():
        raise ValueError("Third-round predictions must be unique per key")
    if predictions["prediction"].isna().any():
        raise ValueError("Third-round predictions must be complete")
    if (predictions["prediction"] < 0).any():
        raise ValueError("Third-round predictions must be non-negative")

    # ---- 7. quality report ----
    quality, peak_threshold, low_threshold = (
        build_experiment_quality(predictions)
    )

    # ---- 8. summarize ----
    overall = quality[quality["level"] == "overall"].set_index("method")
    windows = quality[quality["level"] == "window"].pivot(
        index="entity_id", columns="method", values="mae"
    )

    summary_rows = []
    for method in [BASELINE_METHOD, SPARSE_METHOD]:
        if method not in overall.index:
            continue
        metrics = overall.loc[method]
        baseline_metrics = (
            overall.loc[BASELINE_METHOD]
            if BASELINE_METHOD in overall.index
            else None
        )
        mae_delta = (
            metrics["mae"] - baseline_metrics["mae"]
            if baseline_metrics is not None
            else np.nan
        )
        mae_improvement_pct = (
            (baseline_metrics["mae"] - metrics["mae"])
            / baseline_metrics["mae"]
            * 100
            if baseline_metrics is not None and baseline_metrics["mae"]
            else np.nan
        )
        window_wins = 0
        if (
            BASELINE_METHOD in windows.columns
            and method in windows.columns
        ):
            window_wins = int(
                (windows[method] < windows[BASELINE_METHOD]).sum()
            )

        eligible = bool(
            method != BASELINE_METHOD
            and baseline_metrics is not None
            and metrics["mae"] < baseline_metrics["mae"]
            and window_wins >= 3
        )
        summary_rows.append(
            {
                "method": method,
                "mae": metrics["mae"],
                "mae_delta": mae_delta,
                "mae_improvement_pct": mae_improvement_pct,
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
                "window_mae_wins": window_wins,
                "eligible": eligible,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mae")

    # ---- 9. SKU-level coverage (how many SKUs improved vs baseline?) ----
    sku_mae = (
        predictions.groupby(
            ["store_id", "product_id", "method"]
        )["prediction"]
        .apply(
            lambda p: (
                predictions.loc[p.index, "qty"] - p
            ).abs().mean()
        )
        .unstack("method")
    )
    if BASELINE_METHOD in sku_mae.columns and SPARSE_METHOD in sku_mae.columns:
        sku_coverage = {
            "total_skus": len(sku_mae),
            "improved_skus": int(
                (
                    sku_mae[SPARSE_METHOD] < sku_mae[BASELINE_METHOD]
                ).sum()
            ),
            "sparse_sdku_improved": 0,
            "dense_sku_improved": 0,
        }
        if "is_sparse" in sparsity.columns:
            sku_mae_w_sparsity = sku_mae.join(
                sparsity.set_index(["store_id", "product_id"])[
                    "is_sparse"
                ],
                how="left",
            )
            sparse_mask = sku_mae_w_sparsity["is_sparse"]
            sku_coverage["sparse_sku_improved"] = int(
                (
                    sku_mae_w_sparsity.loc[
                        sparse_mask, SPARSE_METHOD
                    ]
                    < sku_mae_w_sparsity.loc[
                        sparse_mask, BASELINE_METHOD
                    ]
                ).sum()
            )
            sku_coverage["dense_sku_improved"] = int(
                (
                    ~sparse_mask
                    & (
                        sku_mae_w_sparsity[SPARSE_METHOD]
                        < sku_mae_w_sparsity[BASELINE_METHOD]
                    )
                ).sum()
            )
    else:
        sku_coverage = {}

    # ---- 10. write artifacts ----
    predictions_path = output_dir / "third_round_predictions.csv.gz"
    quality_path = output_dir / "third_round_quality.csv"
    summary_path = output_dir / "third_round_summary.csv"
    metadata_path = output_dir / "third_round_run.json"

    predictions.to_csv(
        predictions_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    quality.to_csv(quality_path, index=False)
    summary.to_csv(summary_path, index=False)

    metadata = {
        "experiment_version": EXPERIMENT_VERSION,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "production_model_changed": False,
        "description": (
            "Separate direct_tweedie_1_3 models for sparse and dense SKUs; "
            "routing by pre-first-origin zero_rate and ADI."
        ),
        "origins": [o.date().isoformat() for o in origins],
        "forecast_horizon_days": PREDICTION_LOOKAHEAD,
        "sparse_sku_count": n_sparse,
        "total_sku_count": n_total,
        "feature_count": len(feature_cols),
        "baseline_method": BASELINE_METHOD,
        "expert_method": SPARSE_METHOD,
        "sku_coverage": sku_coverage,
        "eligible_methods": (
            summary.loc[summary["eligible"], "method"].tolist()
        ),
        "recommended_candidate": (
            summary.loc[summary["eligible"], "method"].iloc[0]
            if summary["eligible"].any()
            else None
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

    print("\nThird-round aggregate summary:")
    print(summary.to_string(index=False))
    return predictions, quality, summary, metadata


if __name__ == "__main__":
    run_third_round_experiments()