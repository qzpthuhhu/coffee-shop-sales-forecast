"""Build the static dashboard snapshot from the latest trusted model artifacts."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
OUTPUT_FILE = ROOT / "dashboard" / "data.json"
PREDICTION_COLUMN = "pred_ensemble"


def _round(value, digits=4):
    return round(float(value), digits)


def _feature_category(feature):
    if feature.startswith(("qty_", "sku_share_")):
        return "时序"
    if feature.startswith(("temperature_", "precipitation", "is_rain", "is_snow",
                           "is_clear", "sunshine_", "wind_", "relative_humidity",
                           "temp_range", "is_hot", "is_cold")):
        return "天气"
    if "holiday" in feature or feature == "is_long_weekend":
        return "节假日"
    if feature in {"price", "price_vs_category_avg"}:
        return "价格"
    return "日历"


def _series_by_entity(predictions, entity_column):
    grouped = (
        predictions.groupby([entity_column, "date"], dropna=False)
        .agg(actual=("qty", "sum"), pred=(PREDICTION_COLUMN, "sum"))
        .reset_index()
        .sort_values([entity_column, "date"])
    )
    result = {}
    for entity, rows in grouped.groupby(entity_column, dropna=False):
        result[str(entity)] = {
            "dates": rows["date"].dt.strftime("%Y-%m-%d").tolist(),
            "actual": [_round(value, 1) for value in rows["actual"]],
            "pred": [_round(value, 1) for value in rows["pred"]],
        }
    return result


def build_dashboard_payload(predictions, daily, importance, generated_at=None):
    predictions = predictions.copy()
    daily = daily.copy()
    predictions["date"] = pd.to_datetime(predictions["date"])
    daily["date"] = pd.to_datetime(daily["date"])

    actual = predictions["qty"].to_numpy()
    predicted = predictions[PREDICTION_COLUMN].to_numpy()
    errors = actual - predicted
    relative_errors = np.abs(errors / (actual + 1))

    metrics = {
        "mae": _round(mean_absolute_error(actual, predicted), 2),
        "rmse": _round(np.sqrt(mean_squared_error(actual, predicted)), 2),
        "r2": _round(r2_score(actual, predicted), 3),
        "within_10pct": _round((relative_errors < 0.10).mean() * 100, 1),
        "within_20pct": _round((relative_errors < 0.20).mean() * 100, 1),
    }

    daily_rows = []
    for row in daily.sort_values("date").itertuples():
        daily_rows.append({
            "date": row.date.strftime("%Y-%m-%d"),
            "actual": _round(row.actual, 1),
            "pred": _round(row.predicted, 1),
            "error": _round(row.error, 1),
            "error_pct": _round(row.error_pct * 100, 1),
        })

    feature_importance = [
        {
            "feature": row.feature,
            "importance": _round(row.shap_importance, 6),
            "category": _feature_category(row.feature),
        }
        for row in importance.head(15).itertuples()
    ]

    error_rows = predictions.assign(abs_error=np.abs(errors)).nlargest(10, "abs_error")
    top_errors = [
        {
            "date": row.date.strftime("%Y-%m-%d"),
            "store": row.store_name,
            "category": row.category,
            "type": row.product_type,
            "sku": int(row.product_id),
            "actual": _round(row.qty, 1),
            "pred": _round(row.pred_ensemble, 1),
            "error": _round(row.error, 1),
        }
        for row in error_rows.itertuples()
    ]

    first_by_date = (
        predictions.sort_values(["date", "store_id", "product_id"])
        .drop_duplicates("date")
        .set_index("date")
    )
    weather_info = {}
    for date, row in first_by_date.iterrows():
        weather_info[date.strftime("%Y-%m-%d")] = {
            "temp_mean": _round(row["temperature_2m_mean"], 1),
            "precipitation": _round(row["precipitation_sum"], 1),
            "is_rain": bool(row["is_rain"]),
            "is_holiday": bool(row["is_holiday_detailed"]),
            "is_long_weekend": bool(row["is_long_weekend"]),
        }

    feature_columns = [
        column for column in predictions.columns
        if column not in {
            "date", "store_id", "store_name", "product_id", "category",
            "product_type", "qty", "dow", "holiday_impact", "pred_hgb",
            "pred_gb", "pred_ensemble", "error", "error_pct",
        }
    ]
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    start = predictions["date"].min().strftime("%Y-%m-%d")
    end = predictions["date"].max().strftime("%Y-%m-%d")

    return {
        "metadata": {
            "model_name": "Global HistGradientBoosting + GradientBoosting ensemble",
            "validation_start": start,
            "validation_end": end,
            "validation_days": int(predictions["date"].nunique()),
            "sample_count": int(len(predictions)),
            "feature_count": len(feature_columns),
            "generated_at": generated_at,
            "limitation": "Single 7-day holdout; not production accuracy.",
        },
        "metrics": metrics,
        "daily_predictions": daily_rows,
        "feature_importance": feature_importance,
        "store_predictions": _series_by_entity(predictions, "store_name"),
        "category_predictions": _series_by_entity(predictions, "category"),
        "type_predictions": _series_by_entity(predictions, "product_type"),
        "weather_info": weather_info,
        "test_dates": sorted(predictions["date"].dt.strftime("%Y-%m-%d").unique()),
        "top_errors": top_errors,
        "shap_note": "TreeSHAP mean absolute contribution in cups; direction is not retained in this aggregate.",
    }


def export_dashboard_data(output_file=OUTPUT_FILE):
    payload = build_dashboard_payload(
        pd.read_csv(MODEL_DIR / "test_predictions.csv"),
        pd.read_csv(MODEL_DIR / "daily_predictions.csv"),
        pd.read_csv(MODEL_DIR / "shap_importance.csv"),
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    exported = export_dashboard_data()
    print(
        f"Exported {len(exported['daily_predictions'])} validation days "
        f"to {OUTPUT_FILE}"
    )
