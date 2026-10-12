"""
Coffee Shop Sales Prediction Pipeline
======================================
Global LightGBM + SHAP Explainability

架构：Global Single Model（跨门店、SKU 共享）
- 主力模型：lightgbm.LGBMRegressor
- 可解释性：TreeSHAP + 大模型根因分析
"""
import pandas as pd
import numpy as np
import shap
import joblib
import warnings
from pathlib import Path
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
DATA_DIR = Path(__file__).parent
SALES_FILE = DATA_DIR / "Coffee Shop Sales.xlsx"
WEATHER_FILE = DATA_DIR / "nyc_weather_2023_h1.csv"
MODEL_DIR = DATA_DIR / "models"
PREDICTION_LOOKAHEAD = 7
STORE_NAME_MAP = {
    3: "Astoria",
    5: "Lower Manhattan",
    8: "Hell's Kitchen",
}
HISTORY_LAGS = [1, 2, 3, 7, 14, 28]
HISTORY_WINDOWS = [7, 14, 28]
LIGHTGBM_PARAMS = {
    'objective': 'regression_l1',
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'num_leaves': 63,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.9,
    'subsample_freq': 1,
    'colsample_bytree': 0.9,
    'reg_alpha': 0.05,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': 1,
    'verbosity': -1,
}


def add_history_features(df):
    """Build target-derived features using observations strictly before each row."""
    df = df.sort_values(['store_id', 'product_id', 'date']).reset_index(drop=True)
    entity_keys = ['store_id', 'product_id']

    for lag in HISTORY_LAGS:
        df[f'qty_lag_{lag}'] = df.groupby(entity_keys)['qty'].shift(lag)

    for window in HISTORY_WINDOWS:
        qty_group = df.groupby(entity_keys)['qty']
        shifted = qty_group.transform(lambda values: values.shift(1))
        grouped_shifted = shifted.groupby([df[key] for key in entity_keys])
        df[f'qty_roll_mean_{window}d'] = grouped_shifted.transform(
            lambda values: values.rolling(window, min_periods=1).mean()
        )
        df[f'qty_roll_std_{window}d'] = grouped_shifted.transform(
            lambda values: values.rolling(window, min_periods=1).std().fillna(0)
        )

    df['dow'] = df['date'].dt.dayofweek
    df['qty_dow_avg'] = df.groupby(entity_keys + ['dow'])['qty'].transform(
        lambda values: values.shift(1).expanding(min_periods=1).mean()
    )

    store_day_total = df.groupby(['store_id', 'date'])['qty'].transform('sum')
    df['_observed_sku_share'] = (
        df['qty'] / store_day_total.replace(0, np.nan)
    ).fillna(0)
    df['sku_share_28d'] = df.groupby(entity_keys)['_observed_sku_share'].transform(
        lambda values: values.shift(1).rolling(28, min_periods=1).mean()
    )
    return df.drop(columns=['_observed_sku_share'])

# ============================================================
# 1. 数据加载与预处理
# ============================================================
def load_and_preprocess():
    """加载销售和天气数据，聚合为门店 x SKU x 日 粒度"""
    print("=" * 60)
    print("1. 加载数据与特征工程")
    print("=" * 60)

    sales = pd.read_excel(SALES_FILE)
    sales['date'] = pd.to_datetime(sales['transaction_date'])
    weather = pd.read_csv(WEATHER_FILE, parse_dates=['date'])

    # 按 门店 x SKU x 日 聚合
    df = sales.groupby(['date', 'store_id', 'product_id']).agg(
        qty=('transaction_qty', 'sum'),
        price=('unit_price', 'mean'),
        category=('product_category', 'first'),
        product_type=('product_type', 'first'),
    ).reset_index()
    df['store_name'] = df['store_id'].map(STORE_NAME_MAP)

    # 构建完整时间面板
    dates = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
    stores = df['store_id'].unique()
    skus = df['product_id'].unique()

    full_index = pd.MultiIndex.from_product(
        [dates, stores, skus], names=['date', 'store_id', 'product_id']
    )
    df_full = pd.DataFrame(index=full_index).reset_index()
    df_full['date'] = pd.to_datetime(df_full['date'])

    df = df_full.merge(df, on=['date', 'store_id', 'product_id'], how='left')
    df['qty'] = df['qty'].fillna(0).astype(int)
    df['store_name'] = df['store_name'].fillna(
        df['store_id'].map(STORE_NAME_MAP)
    )
    df['price'] = df['price'].fillna(df.groupby('product_id')['price'].transform('mean'))
    df['price'] = df['price'].fillna(df['price'].mean())
    df['category'] = df['category'].fillna(df.groupby('product_id')['category'].transform('first'))
    df['product_type'] = df['product_type'].fillna(df.groupby('product_id')['product_type'].transform('first'))

    print(f"  聚合后数据: {len(df):,} 行 ({df['date'].nunique()} 天 x {df['store_id'].nunique()} 店 x {df['product_id'].nunique()} SKU)")

    # --- 特征工程 ---
    # 第一梯队：时序核心
    print("  构造时序特征...")
    df = add_history_features(df)

    # 第二梯队：日历（含节假日增强）
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['date'].dt.month
    df['day_of_month'] = df['date'].dt.day
    df['week_of_month'] = ((df['date'].dt.day - 1) // 7 + 1).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

    # 季节编码
    season_map = {'winter': 0, 'spring': 1, 'summer': 2, 'fall': 3}
    def get_season(m):
        if m in [12, 1, 2]: return 'winter'
        elif m in [3, 4, 5]: return 'spring'
        elif m in [6, 7, 8]: return 'summer'
        else: return 'fall'
    df['season_code'] = df['date'].dt.month.apply(get_season).map(season_map)

    # 季中第几周
    season_start_month = {'winter': 1, 'spring': 4, 'summer': 7, 'fall': 10}
    df['week_of_season'] = df.apply(
        lambda r: (r['date'] - pd.Timestamp(year=r['date'].year,
                   month=season_start_month[get_season(r['date'].month)], day=1)).days // 7 + 1,
        axis=1)

    # 第三梯队：天气（含增强节假日）
    weather_cols = [
        'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
        'precipitation_sum', 'is_rain', 'is_snow', 'is_clear',
        'sunshine_duration', 'wind_speed_10m_max', 'relative_humidity_2m_mean',
        'is_holiday_detailed', 'is_pre_holiday_1d', 'is_pre_holiday_3d',
        'is_post_holiday_1d', 'is_long_weekend'
    ]
    df = df.merge(weather[['date'] + weather_cols], on='date', how='left')

    # 假日影响力编码（高=2, 中=1, 无=0）
    impact_map = {'high': 2, 'medium': 1, 'none': 0}
    df_weather_full = pd.read_csv(WEATHER_FILE, parse_dates=['date'])
    df = df.merge(df_weather_full[['date', 'holiday_impact']], on='date', how='left')
    df['holiday_impact_code'] = df['holiday_impact'].map(impact_map).fillna(0).astype(int)

    df['temp_range'] = df['temperature_2m_max'] - df['temperature_2m_min']
    df['is_hot'] = (df['temperature_2m_max'] > 25).astype(int)
    df['is_cold'] = (df['temperature_2m_max'] < 5).astype(int)

    # 第四梯队：价格
    cat_avg_price = df.groupby('category')['price'].transform('mean')
    df['price_vs_category_avg'] = df['price'] / cat_avg_price.replace(0, 1)
    df['price'] = df['price'].fillna(df['price'].median())

    print(f"  特征矩阵: {df.shape[1]} 列")
    return df


def get_feature_cols(df):
    exclude = ['date', 'store_id', 'store_name', 'product_id', 'category',
               'product_type', 'qty', 'dow', 'holiday_impact',
               'season_spring', 'season_summer', 'season_winter']
    return [c for c in df.columns if c not in exclude]


# ============================================================
# 2. 训练/测试切分
# ============================================================
def prepare_train_test(df):
    print("\n" + "=" * 60)
    print("2. 训练/测试切分")
    print("=" * 60)
    dates = sorted(df['date'].unique())
    test_start = dates[-PREDICTION_LOOKAHEAD]
    train = df[df['date'] < test_start].copy()
    test = df[df['date'] >= test_start].copy()
    print(f"  训练集: {train['date'].min().date()} ~ {train['date'].max().date()} ({len(train):,} 行)")
    print(f"  测试集: {test['date'].min().date()} ~ {test['date'].max().date()} ({len(test):,} 行)")
    return train, test


# ============================================================
# 3. TimeSeriesSplit CV
# ============================================================
def time_series_cv(df, feature_cols, n_splits=3):
    print("\n" + "=" * 60)
    print("3. TimeSeriesSplit 交叉验证")
    print("=" * 60)
    dates = sorted(df['date'].unique())
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(dates)):
        train_dates = [dates[i] for i in train_idx]
        val_dates = [dates[i] for i in val_idx]
        train_data = df[df['date'].isin(train_dates)].dropna(subset=feature_cols + ['qty'])
        val_data = df[df['date'].isin(val_dates)].dropna(subset=feature_cols + ['qty'])

        model = LGBMRegressor(**LIGHTGBM_PARAMS)
        model.fit(
            train_data[feature_cols],
            train_data['qty'],
            eval_X=val_data[feature_cols],
            eval_y=val_data['qty'],
            eval_metric='mae',
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        y_pred = model.predict(val_data[feature_cols])
        mae = mean_absolute_error(val_data['qty'], y_pred)
        scores.append(mae)
        print(f"  Fold {fold+1}: {val_dates[0].date()} ~ {val_dates[-1].date()}, MAE={mae:.4f}")

    print(f"\n  CV MAE: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return scores


# ============================================================
# 4. 模型训练
# ============================================================
def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train the single global LightGBM model with holdout early stopping."""
    print("\n" + "=" * 60)
    print("4. Global LightGBM 训练")
    print("=" * 60)

    model = LGBMRegressor(**LIGHTGBM_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_X=X_val,
        eval_y=y_val,
        eval_metric='mae',
        callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
    )

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    print(
        f"  LightGBM 验证集: MAE={mae:.4f}, RMSE={rmse:.4f}, "
        f"R²={r2:.4f}, best_iteration={model.best_iteration_}"
    )
    return model


# ============================================================
# 5. SHAP 可解释性
# ============================================================
def shap_explain(model, X_sample, feature_cols):
    print("\n" + "=" * 60)
    print("6. SHAP 可解释性分析")
    print("=" * 60)

    print("  计算 TreeSHAP (sampled)...")
    explain_data = X_sample.iloc[:500]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(explain_data)
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    reconstructed = base_value + np.asarray(shap_values).sum(axis=1)
    max_additivity_error = np.max(np.abs(reconstructed - model.predict(explain_data)))
    if max_additivity_error > 1e-6:
        raise ValueError(
            f"TreeSHAP additivity validation failed: max error={max_additivity_error:.8f}"
        )
    print(f"  SHAP 加法校验通过，最大误差={max_additivity_error:.2e}")

    importance = np.abs(shap_values).mean(axis=0)
    feat_imp = pd.DataFrame({
        'feature': feature_cols,
        'shap_importance': importance
    }).sort_values('shap_importance', ascending=False)

    print("\n  Top 20 特征重要性 (SHAP):")
    for i, (_, row) in enumerate(feat_imp.head(20).iterrows()):
        print(f"  {i+1:2d}. {row['feature']:<30s} {row['shap_importance']:.6f}")

    return feat_imp, shap_values, explainer


# ============================================================
# 6. 预测与评估
# ============================================================
def predict_and_report(model, test_df, feature_cols):
    print("\n" + "=" * 60)
    print("6. 最终预测")
    print("=" * 60)

    X_test = test_df[feature_cols].dropna()
    y_test = test_df.loc[X_test.index, 'qty']

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print(f"  LightGBM: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

    errors = y_test - predictions
    within_10pct = (np.abs(errors / (y_test + 1)) < 0.10).mean()
    within_20pct = (np.abs(errors / (y_test + 1)) < 0.20).mean()
    print(f"\n  预测误差在 ±10% 以内的比例: {within_10pct:.1%}")
    print(f"  预测误差在 ±20% 以内的比例: {within_20pct:.1%}")

    # 保存详细结果
    result_df = test_df.loc[X_test.index].copy()
    result_df['pred_lightgbm'] = predictions
    result_df['error'] = errors
    result_df['error_pct'] = np.abs(errors / (result_df['qty'] + 1))

    # 按日聚合
    daily = result_df.groupby('date').agg(
        actual=('qty', 'sum'), predicted=('pred_lightgbm', 'sum'),
        error=('error', 'sum')
    ).reset_index()
    daily['error_pct'] = np.abs(daily['error'] / daily['actual'])

    print(f"\n  每日总销量预测 (测试集 {len(daily)} 天):")
    for _, row in daily.iterrows():
        flag = "  ***" if row['error_pct'] > 0.15 else ""
        print(f"  {row['date'].date()}  实际={row['actual']:.0f}  预测={row['predicted']:.0f}  "
              f"偏差={row['error_pct']:.1%}{flag}")

    return result_df, daily


# ============================================================
# 7. LLM 根因分析
# ============================================================
def llm_root_cause_analysis(result_df, feat_imp):
    print("\n" + "=" * 60)
    print("8. 根因分析")
    print("=" * 60)

    result_df = result_df.sort_values('error_pct', ascending=False)
    top_errors = result_df.head(5)

    print("\n  预测偏差最大的 5 个样本:")
    for _, row in top_errors.iterrows():
        print(f"  {row['date'].date()} | {row['store_name']} | SKU={row['product_id']} | "
              f"实际={row['qty']:.0f} | 预测={row['pred_lightgbm']:.1f} | 偏差={row['error_pct']:.1%}")

    top_features = feat_imp.head(10).to_dict('records')
    lines = [
        "你是茶饮行业销量预测分析专家。请分析以下咖啡门店销量预测结果。",
        "",
        "## 背景",
        "- 3 家纽约门店，181 天数据，80 个 SKU",
        "- 模型: Global LightGBM (lightgbm.LGBMRegressor)",
        "- 特征: 时序 lag、滚动统计、天气、日历、价格",
        "- 当前 SHAP 仅为全局平均绝对贡献，不含单条预测的方向",
        "- 禁止把全局 SHAP、天气相关性或业务假设表述为单条预测的确定性因果根因",
        "",
        "## Top 10 特征重要性 (SHAP)",
    ]
    for f in top_features:
        lines.append(f"- {f['feature']}: {f['shap_importance']:.6f}")

    lines += ["", "## 预测偏差最大的样本", "| 日期 | 门店 | SKU | 实际 | 预测 | 偏差率 |",
              "|------|------|-----|------|------|--------|"]
    for _, row in top_errors.iterrows():
        lines.append(f"| {row['date'].date()} | {row['store_name']} | {row['product_id']} "
                     f"| {row['qty']:.0f} | {row['pred_lightgbm']:.1f} | {row['error_pct']:.1%} |")

    lines += ["", "## 请分析",
              "1. 全局模型主要依赖哪些特征？不要据此断言单条预测根因。",
              "2. 哪些异常需要逐预测带符号 SHAP 或额外业务数据才能解释？",
              "3. 模型改进建议（特征、算法、数据清洗）。",
              "4. 如果这是茶饮连锁（如柠季），这种预测精度对订货和排班够不够用？"]

    prompt = "\n".join(lines)
    print(f"\n  根因分析提示词已生成 ({len(prompt)} 字符)")
    print("  " + "-" * 50)
    print(prompt)

    with open(DATA_DIR / "root_cause_prompt.txt", "w") as f:
        f.write(prompt)

    return prompt


# ============================================================
# Main Pipeline
# ============================================================
def run_pipeline():
    print("\n" + "=" * 60)
    print("Coffee Shop Sales 销量预测 Pipeline")
    print("Global LightGBM + SHAP")
    print("=" * 60)

    # 1. 数据加载 & 特征工程
    df = load_and_preprocess()

    # 2. 切分
    train, test = prepare_train_test(df)
    feature_cols = get_feature_cols(df)

    # 3. CV
    time_series_cv(df, feature_cols, n_splits=3)

    # 4. 训练数据
    X_train = train.dropna(subset=feature_cols + ['qty'])[feature_cols]
    y_train = train.loc[X_train.index, 'qty']
    X_val = test.dropna(subset=feature_cols + ['qty'])[feature_cols]
    y_val = test.loc[X_val.index, 'qty']

    # 5. 训练
    model = train_lightgbm(X_train, y_train, X_val, y_val)

    # 6. 预测
    result_df, daily_result = predict_and_report(model, test, feature_cols)

    # 7. SHAP
    X_sample = X_train.sample(min(2000, len(X_train)), random_state=42)
    feat_imp, shap_values, explainer = shap_explain(model, X_sample, feature_cols)

    # 8. 根因分析
    llm_root_cause_analysis(result_df, feat_imp)

    # 9. 保存
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "lightgbm_model.pkl")
    feat_imp.to_csv(MODEL_DIR / "shap_importance.csv", index=False)
    result_df.to_csv(MODEL_DIR / "test_predictions.csv", index=False)
    daily_result.to_csv(MODEL_DIR / "daily_predictions.csv", index=False)

    print(f"\n模型和结果已保存到 {MODEL_DIR}/")
    return model, result_df, feat_imp


if __name__ == "__main__":
    run_pipeline()
