"""Build the static dashboard snapshot from trusted rolling-backtest artifacts."""

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent
REPORT_DIR = ROOT / "reports" / "model_quality"
OUTPUT_FILE = ROOT / "dashboard" / "data.json"
EXPLANATION_DIR = ROOT / "dashboard" / "explanations"
SCENARIO_FILE = REPORT_DIR / "rolling_backtest_scenarios.csv.gz"
PREDICTION_COLUMN = "pred_lightgbm"
TOP_CONTRIBUTIONS = 10

FEATURE_LABELS = {
    "price": "商品单价",
    "sku_share_28d": "历史 SKU 销量占比（28 天）",
    "day_of_week": "星期",
    "is_weekend": "是否周末",
    "month": "月份",
    "day_of_month": "月内日期",
    "week_of_month": "月内周次",
    "is_month_start": "是否月初",
    "is_month_end": "是否月末",
    "season_code": "季节",
    "week_of_season": "季内周次",
    "temperature_2m_max": "最高气温",
    "temperature_2m_min": "最低气温",
    "temperature_2m_mean": "平均气温",
    "precipitation_sum": "降水量",
    "is_rain": "是否下雨",
    "is_snow": "是否下雪",
    "is_clear": "是否晴朗",
    "sunshine_duration": "日照时长",
    "wind_speed_10m_max": "最大风速",
    "relative_humidity_2m_mean": "平均湿度",
    "is_holiday_detailed": "节假日",
    "is_pre_holiday_1d": "节前 1 天",
    "is_pre_holiday_3d": "节前 3 天",
    "is_post_holiday_1d": "节后 1 天",
    "is_long_weekend": "是否长周末",
    "holiday_impact_code": "节假日影响编码",
    "temp_range": "昼夜温差",
    "is_hot": "是否高温",
    "is_cold": "是否低温",
    "price_vs_category_avg": "相对品类均价",
}


def _round(value, digits=4):
    return round(float(value), digits)


def _feature_category(feature):
    if feature.startswith(("qty_", "sku_share_")):
        return "历史销量"
    if feature.startswith((
        "temperature_", "precipitation", "is_rain", "is_snow", "is_clear",
        "sunshine_", "wind_", "relative_humidity", "temp_range", "is_hot",
        "is_cold",
    )):
        return "天气"
    if "holiday" in feature or feature == "is_long_weekend":
        return "节假日"
    if feature in {"price", "price_vs_category_avg"}:
        return "商品"
    return "日历"


def _feature_label(feature):
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    if feature.startswith("qty_lag_"):
        return f"{feature.removeprefix('qty_lag_')} 天前销量"
    if feature.startswith("qty_roll_mean_"):
        return f"历史 {feature.removeprefix('qty_roll_mean_').removesuffix('d')} 天平均销量"
    if feature.startswith("qty_roll_std_"):
        return f"历史 {feature.removeprefix('qty_roll_std_').removesuffix('d')} 天销量波动"
    if feature == "qty_dow_avg":
        return "历史同星期平均销量"
    return feature


def _business_explanation(feature, contribution):
    category = _feature_category(feature)
    direction = "推高" if contribution >= 0 else "压低"
    context = {
        "历史销量": "历史需求状态",
        "天气": "预测日天气输入",
        "节假日": "节假日及前后窗口",
        "商品": "商品价格与品类位置",
        "日历": "预测日日历位置",
    }[category]
    return (
        f"模型根据{context}，将该因素与基准值相比识别为{direction}销量的信号；"
        "这是模型贡献，不代表因果关系。"
    )


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


def _metric_payload(quality):
    row = quality[
        (quality["level"] == "overall") & (quality["method"] == "lightgbm")
    ].iloc[0]
    return {
        "mae": _round(row["mae"], 3),
        "rmse": _round(row["rmse"], 3),
        "wape": _round(row["wape"] * 100, 1),
        "r2": _round(row["r2"], 3),
        "within_10pct": _round(row["within_10pct"] * 100, 1),
        "within_20pct": _round(row["within_20pct"] * 100, 1),
        "interval_coverage": _round(row["interval_coverage"] * 100, 1),
        "interval_mean_width": _round(row["interval_mean_width"], 2),
        "candidate_eligible": bool(row["candidate_eligible"]),
    }


def _diagnostic_payload(predictions):
    sampled = (
        predictions.assign(
            _sample_key=predictions["prediction_id"].astype(str)
        )
        .sort_values(["date", "store_name", "_sample_key"])
        .groupby(["date", "store_name"], as_index=False)
        .head(8)
    )
    extremes = predictions.nlargest(30, "absolute_error")
    sampled = (
        pd.concat([sampled, extremes], ignore_index=True)
        .drop_duplicates("prediction_id")
        .sort_values(["date", "store_name", "product_id"])
    )
    scatter = [
        {
            "x": _round(row.qty, 3),
            "y": _round(row.pred_lightgbm, 3),
            "date": row.date.strftime("%Y-%m-%d"),
            "store": row.store_name,
            "sku": int(row.product_id),
            "error": _round(row.prediction_error, 3),
        }
        for row in sampled.itertuples()
    ]

    bin_width = 2
    minimum = math.floor(predictions["prediction_error"].min() / bin_width) * bin_width
    maximum = math.ceil(predictions["prediction_error"].max() / bin_width) * bin_width
    histogram = []
    for lower in range(minimum, maximum, bin_width):
        upper = lower + bin_width
        if upper == maximum:
            count = int(
                predictions["prediction_error"].between(
                    lower, upper, inclusive="both"
                ).sum()
            )
        else:
            count = int(
                (
                    (predictions["prediction_error"] >= lower)
                    & (predictions["prediction_error"] < upper)
                ).sum()
            )
        histogram.append({
            "lower": lower,
            "upper": upper,
            "count": count,
        })

    return {
        "scatter": scatter,
        "scatter_sampling": (
            "每个门店每日确定性抽样 8 条，并保留绝对误差最大的 30 条预测。"
        ),
        "histogram": histogram,
        "histogram_sample_count": int(len(predictions)),
        "mean_error": _round(predictions["prediction_error"].mean(), 3),
        "underprediction_rate": _round(
            (predictions["prediction_error"] > 0).mean() * 100, 1
        ),
    }


def _demand_heatmap_payload(predictions):
    dates = sorted(predictions["date"].dt.strftime("%Y-%m-%d").unique())
    rows = []
    for keys, group in predictions.groupby(
        ["store_name", "category", "product_type", "product_id"], sort=True
    ):
        store, category, product_type, sku = keys
        values_by_date = {
            row.date.strftime("%Y-%m-%d"): _round(row.pred_lightgbm, 1)
            for row in group.itertuples()
        }
        rows.append({
            "store": store,
            "category": category,
            "type": product_type,
            "sku": int(sku),
            "values": [values_by_date.get(date) for date in dates],
        })
    return {
        "dates": dates,
        "rows": rows,
        "unit": "杯 / SKU / 日",
        "value": "Global LightGBM 点预测",
    }


def build_dashboard_payload(
    predictions, quality, shap_contributions, run_metadata, generated_at=None
):
    predictions = predictions.copy()
    predictions["date"] = pd.to_datetime(predictions["date"])
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()

    daily = (
        predictions.groupby("date")
        .agg(actual=("qty", "sum"), pred=(PREDICTION_COLUMN, "sum"))
        .reset_index()
        .sort_values("date")
    )
    daily_rows = [
        {
            "date": row.date.strftime("%Y-%m-%d"),
            "actual": _round(row.actual, 1),
            "pred": _round(row.pred, 1),
            "error": _round(row.actual - row.pred, 1),
        }
        for row in daily.itertuples()
    ]

    importance = (
        shap_contributions.assign(
            absolute_contribution=shap_contributions["shap_contribution"].abs()
        )
        .groupby("feature", as_index=False)["absolute_contribution"]
        .mean()
        .sort_values("absolute_contribution", ascending=False)
        .head(15)
    )
    feature_importance = [
        {
            "feature": row.feature,
            "label": _feature_label(row.feature),
            "importance": _round(row.absolute_contribution, 6),
            "category": _feature_category(row.feature),
        }
        for row in importance.itertuples()
    ]

    top_errors = [
        {
            "prediction_id": row.prediction_id,
            "date": row.date.strftime("%Y-%m-%d"),
            "store": row.store_name,
            "category": row.category,
            "type": row.product_type,
            "sku": int(row.product_id),
            "actual": _round(row.qty, 1),
            "pred": _round(row.pred_lightgbm, 1),
            "error": _round(row.prediction_error, 1),
        }
        for row in predictions.nlargest(10, "absolute_error").itertuples()
    ]

    hierarchy = (
        predictions[
            ["store_id", "store_name", "category", "product_type", "product_id"]
        ]
        .drop_duplicates()
        .sort_values(["store_name", "category", "product_type", "product_id"])
    )
    hierarchy_rows = [
        {
            "store_id": int(row.store_id),
            "store": row.store_name,
            "category": row.category,
            "type": row.product_type,
            "sku": int(row.product_id),
        }
        for row in hierarchy.itertuples()
    ]

    start = predictions["date"].min().strftime("%Y-%m-%d")
    end = predictions["date"].max().strftime("%Y-%m-%d")
    dates = sorted(predictions["date"].dt.strftime("%Y-%m-%d").unique())
    feature_catalog = {
        feature: {
            "label": _feature_label(feature),
            "category": _feature_category(feature),
            "increase_explanation": _business_explanation(feature, 1),
            "decrease_explanation": _business_explanation(feature, -1),
        }
        for feature in sorted(shap_contributions["feature"].unique())
    }
    feature_catalog["other_features"] = {
        "label": f"其他 {run_metadata['feature_count'] - TOP_CONTRIBUTIONS} 项因素",
        "category": "其他",
        "increase_explanation": (
            "为保证瀑布图与原始模型输出可核对，其余较小贡献合并展示；"
            "完整 44 项明细保留在回测 SHAP 产物中。"
        ),
        "decrease_explanation": (
            "为保证瀑布图与原始模型输出可核对，其余较小贡献合并展示；"
            "完整 44 项明细保留在回测 SHAP 产物中。"
        ),
    }
    return {
        "metadata": {
            "model_run_id": run_metadata["run_id"],
            "model_name": run_metadata["model_name"],
            "candidate_status": run_metadata["candidate_status"],
            "validation_start": start,
            "validation_end": end,
            "validation_days": int(predictions["date"].nunique()),
            "backtest_windows": int(run_metadata["backtest_window_count"]),
            "sample_count": int(len(predictions)),
            "feature_count": int(run_metadata["feature_count"]),
            "generated_at": generated_at,
            "artifact_generated_at": run_metadata["generated_at"],
            "limitations": run_metadata["limitations"],
        },
        "metrics": _metric_payload(quality),
        "diagnostics": _diagnostic_payload(predictions),
        "demand_heatmap": _demand_heatmap_payload(predictions),
        "daily_predictions": daily_rows,
        "feature_importance": feature_importance,
        "store_predictions": _series_by_entity(predictions, "store_name"),
        "category_predictions": _series_by_entity(predictions, "category"),
        "type_predictions": _series_by_entity(predictions, "product_type"),
        "dates": dates,
        "hierarchy": hierarchy_rows,
        "feature_catalog": feature_catalog,
        "scenario_contract": run_metadata["scenario_simulation"],
        "explanation_shards": {
            date: f"explanations/{date}.json" for date in dates
        },
        "top_errors": top_errors,
        "shap_note": (
            "TreeSHAP 全局平均绝对贡献，单位为杯；逐预测解释保留方向并通过加法校验。"
        ),
    }


def build_explanation_shards(
    predictions,
    shap_contributions,
    generated_at,
    scenarios=None,
):
    predictions = predictions.copy()
    predictions["date"] = pd.to_datetime(predictions["date"])
    shap_groups = {
        prediction_id: rows
        for prediction_id, rows in shap_contributions.groupby("prediction_id")
    }
    scenario_groups = {}
    if scenarios is not None:
        scenario_groups = {
            prediction_id: rows.sort_values("scenario_index")
            for prediction_id, rows in scenarios.groupby("prediction_id")
        }
    shards = {}
    for date, date_rows in predictions.groupby("date", sort=True):
        records = []
        for row in date_rows.sort_values(["store_name", "product_id"]).itertuples():
            contributions = shap_groups[row.prediction_id].copy()
            contributions["absolute"] = contributions["shap_contribution"].abs()
            contributions = contributions.sort_values("absolute", ascending=False)
            top = contributions.head(TOP_CONTRIBUTIONS)
            detail = [
                {
                    "feature": item.feature,
                    "value": _round(item.feature_value, 4),
                    "contribution": _round(item.shap_contribution, 6),
                    "direction": item.direction,
                }
                for item in top.itertuples()
            ]
            other = contributions.iloc[TOP_CONTRIBUTIONS:]["shap_contribution"].sum()
            detail.append({
                "feature": "other_features",
                "value": None,
                "contribution": _round(other, 6),
                "direction": "increase" if other >= 0 else "decrease",
            })
            record = {
                "prediction_id": row.prediction_id,
                "model_run_id": row.model_run_id,
                "forecast_origin": row.forecast_origin,
                "horizon_day": int(row.horizon_day),
                "date": row.date.strftime("%Y-%m-%d"),
                "store_id": int(row.store_id),
                "store": row.store_name,
                "category": row.category,
                "type": row.product_type,
                "sku": int(row.product_id),
                "actual": _round(row.qty, 3),
                "prediction": _round(row.pred_lightgbm, 6),
                "raw_prediction": _round(row.pred_lightgbm_raw, 6),
                "model_prediction": _round(row.pred_lightgbm_model, 6),
                "p10": _round(row.pred_p10, 6),
                "p50": _round(row.pred_p50, 6),
                "p90": _round(row.pred_p90, 6),
                "error": _round(row.prediction_error, 6),
                "base_value": _round(row.base_value, 6),
                "postprocess_adjustment": _round(row.postprocess_adjustment, 6),
                "store_bias_factor": _round(row.store_bias_factor, 6),
                "store_bias_adjustment": _round(
                    row.store_bias_adjustment,
                    6,
                ),
                "additivity_error": _round(row.shap_additivity_error, 12),
                "contributions": detail,
            }
            if scenario_groups:
                scenario_rows = scenario_groups[row.prediction_id]
                baseline = scenario_rows[
                    (scenario_rows["temperature_id"] == "observed")
                    & (scenario_rows["rain_id"] == "observed")
                    & (scenario_rows["holiday_id"] == "observed")
                ].iloc[0]
                record["scenario_inputs"] = {
                    "temperature_2m_mean": _round(
                        baseline.temperature_2m_mean,
                        1,
                    ),
                    "precipitation_sum": _round(
                        baseline.precipitation_sum,
                        1,
                    ),
                    "is_holiday_detailed": int(
                        baseline.is_holiday_detailed
                    ),
                    "is_long_weekend": int(baseline.is_long_weekend),
                }
                record["scenario_predictions"] = [
                    _round(value, 4)
                    for value in scenario_rows["simulated_prediction"]
                ]
            records.append(record)
        date_key = date.strftime("%Y-%m-%d")
        shards[date_key] = {
            "date": date_key,
            "generated_at": generated_at,
            "records": records,
        }
    return shards


def export_dashboard_data(output_file=OUTPUT_FILE):
    predictions = pd.read_csv(REPORT_DIR / "rolling_backtest_predictions.csv")
    quality = pd.read_csv(REPORT_DIR / "rolling_backtest_quality.csv")
    shap_contributions = pd.read_csv(
        REPORT_DIR / "rolling_backtest_shap.csv.gz"
    )
    scenarios = pd.read_csv(SCENARIO_FILE)
    run_metadata = json.loads(
        (REPORT_DIR / "rolling_backtest_run.json").read_text(encoding="utf-8")
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = build_dashboard_payload(
        predictions,
        quality,
        shap_contributions,
        run_metadata,
        generated_at=generated_at,
    )
    shards = build_explanation_shards(
        predictions,
        shap_contributions,
        generated_at,
        scenarios=scenarios,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if EXPLANATION_DIR.exists():
        shutil.rmtree(EXPLANATION_DIR)
    EXPLANATION_DIR.mkdir(parents=True)
    output_file.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    for date, shard in shards.items():
        (EXPLANATION_DIR / f"{date}.json").write_text(
            json.dumps(shard, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
    return payload


if __name__ == "__main__":
    exported = export_dashboard_data()
    print(
        f"Exported {len(exported['dates'])} rolling-backtest days and "
        f"{len(exported['hierarchy'])} store-SKU combinations to {OUTPUT_FILE}"
    )
