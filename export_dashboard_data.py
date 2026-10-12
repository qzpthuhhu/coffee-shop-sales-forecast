"""Build the static dashboard snapshot from trusted rolling-backtest artifacts."""

import json
import math
import shutil
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent
REPORT_DIR = ROOT / "reports" / "model_quality"
OUTPUT_FILE = ROOT / "dashboard" / "data.json"
EXPLANATION_DIR = ROOT / "dashboard" / "explanations"
SCENARIO_FILE = REPORT_DIR / "rolling_backtest_scenarios.csv.gz"
SALES_FILE = ROOT / "Coffee Shop Sales.xlsx"
PREDICTION_COLUMN = "pred_lightgbm"
TOP_CONTRIBUTIONS = 10
RISK_RANKING_LIMIT = 10

STORE_LABELS = {
    "Astoria": "阿斯托利亚",
    "Hell's Kitchen": "地狱厨房",
    "Lower Manhattan": "下曼哈顿",
}

CATEGORY_LABELS = {
    "Bakery": "烘焙食品",
    "Branded": "品牌周边",
    "Coffee": "现制咖啡",
    "Coffee beans": "咖啡豆",
    "Drinking Chocolate": "热巧克力原料",
    "Flavours": "风味糖浆",
    "Loose Tea": "散装茶叶",
    "Packaged Chocolate": "包装巧克力",
    "Tea": "现制茶饮",
}

TYPE_LABELS = {
    "Barista Espresso": "意式咖啡",
    "Biscotti": "意式脆饼",
    "Black tea": "红茶",
    "Brewed Black tea": "现泡红茶",
    "Brewed Chai tea": "现泡印度香料茶",
    "Brewed Green tea": "现泡绿茶",
    "Brewed herbal tea": "现泡花草茶",
    "Chai tea": "印度香料茶",
    "Clothing": "服装",
    "Drinking Chocolate": "冲饮巧克力",
    "Drip coffee": "滴滤咖啡",
    "Espresso Beans": "意式咖啡豆",
    "Gourmet Beans": "精品咖啡豆",
    "Gourmet brewed coffee": "精品手冲咖啡",
    "Green beans": "生咖啡豆",
    "Green tea": "绿茶",
    "Herbal tea": "花草茶",
    "Hot chocolate": "热巧克力",
    "House blend Beans": "招牌拼配咖啡豆",
    "Housewares": "杯具",
    "Organic Beans": "有机咖啡豆",
    "Organic Chocolate": "有机巧克力",
    "Organic brewed coffee": "有机现煮咖啡",
    "Packaged Chocolate": "包装巧克力",
    "Pastry": "酥点",
    "Premium Beans": "高端咖啡豆",
    "Premium brewed coffee": "高端现煮咖啡",
    "Regular syrup": "常规糖浆",
    "Scone": "英式司康",
    "Sugar free syrup": "无糖糖浆",
}

PRODUCT_BASE_LABELS = {
    "Brazilian - Organic": "有机巴西咖啡豆",
    "Our Old Time Diner Blend": "老时光餐厅拼配咖啡",
    "Espresso Roast": "意式浓缩烘焙咖啡豆",
    "Primo Espresso Roast": "普里莫意式浓缩烘焙咖啡豆",
    "Columbian Medium Roast": "哥伦比亚中度烘焙咖啡",
    "Ethiopia": "埃塞俄比亚咖啡",
    "Jamacian Coffee River": "牙买加 Coffee River 咖啡豆",
    "Jamaican Coffee River": "牙买加 Coffee River 咖啡",
    "Civet Cat": "猫屎咖啡豆",
    "Organic Decaf Blend": "有机低因拼配咖啡豆",
    "Guatemalan Sustainably Grown": "危地马拉可持续种植咖啡豆",
    "Lemon Grass": "柠檬草茶",
    "Peppermint": "薄荷茶",
    "English Breakfast": "英式早餐茶",
    "Earl Grey": "伯爵茶",
    "Serenity Green Tea": "静心绿茶",
    "Traditional Blend Chai": "传统拼配印度香料茶",
    "Morning Sunrise Chai": "晨曦印度香料茶",
    "Spicy Eye Opener Chai": "醒神辛香印度香料茶",
    "Dark chocolate": "黑巧克力",
    "Sustainably Grown Organic": "可持续种植有机巧克力",
    "Chili Mayan": "玛雅辣椒巧克力",
    "Brazilian": "巴西咖啡",
    "Espresso shot": "浓缩咖啡",
    "Latte": "拿铁",
    "Cappuccino": "卡布奇诺",
    "Carmel syrup": "焦糖糖浆",
    "Hazelnut syrup": "榛果糖浆",
    "Sugar Free Vanilla syrup": "无糖香草糖浆",
    "Hazelnut Biscotti": "榛果意式脆饼",
    "Cranberry Scone": "蔓越莓司康",
    "Chocolate Croissant": "巧克力可颂",
    "Ginger Scone": "姜味司康",
    "Almond Croissant": "杏仁可颂",
    "Ginger Biscotti": "姜味意式脆饼",
    "Croissant": "原味可颂",
    "Chocolate Chip Biscotti": "巧克力豆意式脆饼",
    "Oatmeal Scone": "燕麦司康",
    "Scottish Cream Scone": "苏格兰奶油司康",
    "Jumbo Savory Scone": "大份咸味司康",
    "I Need My Bean! T-shirt": "I Need My Bean! T恤",
    "I Need My Bean! Diner mug": "I Need My Bean! 餐厅马克杯",
    "I Need My Bean! Latte cup": "I Need My Bean! 拿铁杯",
    "Chocolate syrup": "巧克力糖浆",
    "Ouro Brasileiro shot": "巴西黄金浓缩咖啡",
}

SIZE_LABELS = {"Sm": "小杯", "Rg": "中杯", "Lg": "大杯"}

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
    "store_id_feature": "门店差异",
    "product_id_feature": "商品差异",
    "category_code": "品类差异",
    "product_type_code": "商品类型差异",
    "forecast_horizon": "预测提前天数",
    "is_sparse": "是否稀疏品",
    "zero_rate": "历史零销量率",
    "adi": "平均需求间隔（天）",
    "mean_qty": "历史日均销量",
}


def _round(value, digits=4):
    return round(float(value), digits)


def _translate_product_name(product_name):
    parts = product_name.rsplit(" ", 1)
    size = SIZE_LABELS.get(parts[-1]) if len(parts) == 2 else None
    base_name = parts[0] if size else product_name
    translated = PRODUCT_BASE_LABELS.get(base_name, base_name)
    return f"{translated}·{size}" if size else translated


@lru_cache(maxsize=1)
def _product_catalog():
    catalog = (
        pd.read_excel(
            SALES_FILE,
            usecols=[
                "product_id",
                "product_category",
                "product_type",
                "product_detail",
            ],
        )
        .drop_duplicates("product_id")
        .rename(columns={"product_detail": "product_name"})
        .sort_values("product_id")
    )
    catalog["product_name_zh"] = catalog["product_name"].map(
        _translate_product_name
    )
    catalog["product_label"] = (
        catalog["product_name_zh"] + "（" + catalog["product_name"] + "）"
    )
    return catalog.reset_index(drop=True)


def _with_product_names(predictions):
    product_columns = {"product_name", "product_name_zh", "product_label"}
    if product_columns.issubset(predictions.columns):
        return predictions
    return predictions.merge(
        _product_catalog()[
            ["product_id", "product_name", "product_name_zh", "product_label"]
        ],
        on="product_id",
        how="left",
        validate="many_to_one",
    )


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
    if feature in {"is_sparse", "zero_rate", "adi", "mean_qty"}:
        return "稀疏需求"
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
        "稀疏需求": "该 SKU 历史销售稀疏程度",
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


def _series_by_product(predictions):
    grouped = (
        predictions.groupby(
            ["product_id", "product_label", "date"],
            dropna=False,
        )
        .agg(actual=("qty", "sum"), pred=(PREDICTION_COLUMN, "sum"))
        .reset_index()
        .sort_values(["product_id", "date"])
    )
    result = {}
    for (product_id, product_label), rows in grouped.groupby(
        ["product_id", "product_label"],
        dropna=False,
    ):
        result[str(int(product_id))] = {
            "label": product_label,
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
            "product_name": row.product_name,
            "product_name_zh": row.product_name_zh,
            "product_label": row.product_label,
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
        [
            "store_name",
            "category",
            "product_type",
            "product_id",
            "product_name",
            "product_name_zh",
            "product_label",
        ],
        sort=True,
    ):
        (
            store,
            category,
            product_type,
            sku,
            product_name,
            product_name_zh,
            product_label,
        ) = keys
        values_by_date = {
            row.date.strftime("%Y-%m-%d"): _round(row.pred_lightgbm, 1)
            for row in group.itertuples()
        }
        rows.append({
            "store": store,
            "category": category,
            "type": product_type,
            "sku": int(sku),
            "product_name": product_name,
            "product_name_zh": product_name_zh,
            "product_label": product_label,
            "values": [values_by_date.get(date) for date in dates],
        })
    return {
        "dates": dates,
        "rows": rows,
        "unit": "杯 / SKU / 日",
        "value": "Global LightGBM 点预测",
    }


def _risk_ranking_payload(predictions):
    frame = predictions.copy()
    frame["date_key"] = frame["date"].dt.strftime("%Y-%m-%d")
    frame["upside_gap"] = (
        frame["pred_p90"] - frame[PREDICTION_COLUMN]
    ).clip(lower=0)
    rankings = {}
    stores = sorted(frame["store_name"].unique())
    for date, date_rows in frame.groupby("date_key", sort=True):
        rankings[date] = {}
        for scope in ["__all__", *stores]:
            scoped = (
                date_rows
                if scope == "__all__"
                else date_rows[date_rows["store_name"] == scope]
            )
            top = scoped.nlargest(
                RISK_RANKING_LIMIT,
                ["upside_gap", PREDICTION_COLUMN],
            )
            rankings[date][scope] = [
                {
                    "prediction_id": row.prediction_id,
                    "date": row.date_key,
                    "store": row.store_name,
                    "category": row.category,
                    "type": row.product_type,
                    "sku": int(row.product_id),
                    "product_name": row.product_name,
                    "product_name_zh": row.product_name_zh,
                    "product_label": row.product_label,
                    "prediction": _round(row.pred_lightgbm, 1),
                    "p90": _round(row.pred_p90, 1),
                    "upside_gap": _round(row.upside_gap, 1),
                    "interval_width": _round(
                        row.pred_p90 - row.pred_p10,
                        1,
                    ),
                }
                for row in top.itertuples()
            ]
    return {
        "ranking_limit": RISK_RANKING_LIMIT,
        "metric": "P90 - 点预测",
        "unit": "杯 / 门店 × SKU × 日",
        "uses_actual_sales": False,
        "scope_note": (
            "按潜在追加需求杯数排序，仅用于备货关注；"
            "未接入库存与在途数据，不代表缺货概率。"
        ),
        "by_date": rankings,
    }


def build_dashboard_payload(
    predictions, quality, shap_contributions, run_metadata, generated_at=None
):
    predictions = _with_product_names(predictions.copy())
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
            "product_name": row.product_name,
            "product_name_zh": row.product_name_zh,
            "product_label": row.product_label,
            "actual": _round(row.qty, 1),
            "pred": _round(row.pred_lightgbm, 1),
            "error": _round(row.prediction_error, 1),
        }
        for row in predictions.nlargest(10, "absolute_error").itertuples()
    ]

    hierarchy = (
        predictions[
            [
                "store_id",
                "store_name",
                "category",
                "product_type",
                "product_id",
                "product_name",
                "product_name_zh",
                "product_label",
            ]
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
            "product_name": row.product_name,
            "product_name_zh": row.product_name_zh,
            "product_label": row.product_label,
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
            f"完整 {run_metadata['feature_count']} 项明细保留在回测 SHAP 产物中。"
        ),
        "decrease_explanation": (
            "为保证瀑布图与原始模型输出可核对，其余较小贡献合并展示；"
            f"完整 {run_metadata['feature_count']} 项明细保留在回测 SHAP 产物中。"
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
        "risk_rankings": _risk_ranking_payload(predictions),
        "daily_predictions": daily_rows,
        "feature_importance": feature_importance,
        "store_predictions": _series_by_entity(predictions, "store_name"),
        "category_predictions": _series_by_entity(predictions, "category"),
        "type_predictions": _series_by_entity(predictions, "product_type"),
        "product_predictions": _series_by_product(predictions),
        "dates": dates,
        "hierarchy": hierarchy_rows,
        "labels": {
            "stores": STORE_LABELS,
            "categories": CATEGORY_LABELS,
            "types": TYPE_LABELS,
        },
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
    predictions = _with_product_names(predictions.copy())
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
                "product_name": row.product_name,
                "product_name_zh": row.product_name_zh,
                "product_label": row.product_label,
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
