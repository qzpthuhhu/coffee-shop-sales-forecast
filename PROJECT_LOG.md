# Coffee Shop Sales Forecast — 项目日志

**目标**：基于 Coffee Shop Sales 数据集搭建销量预测 pipeline，模拟柠季茶饮门店的日销量预测解决方案。

**数据源**：[Kaggle: Coffee Shop Sales Dataset](https://www.kaggle.com/datasets/ahmedabbas757/coffee-sales)

## 终态推进状态（2026-10-12）

| 模块 | 完成度 | 当前状态 |
|------|:---:|------|
| 防数据泄露特征 | 70% | 已移除预测日实际 `sku_share`，改为前 28 天历史滚动占比；星期均值仅使用历史标签；新增 Lag 28 和 6 项回归测试 |
| 真实 LightGBM 与区间 | 10% | 当前仍为 sklearn HGB + GB，不能标记为 LightGBM；P10/P50/P90 未实现 |
| 滚动回测与基线 | 15% | 现有 3 折时间切分只供诊断，尚未实现多个固定 7 天滚动起点、朴素基线和分层质量报告 |
| 逐预测 SHAP | 30% | TreeSHAP 已恢复非零输出并通过加法校验；尚未持久化每条预测的带符号贡献 |
| 妙搭全栈 | 0% | 尚无本地 `.spark/meta.json`；应用检索被 `spark:app:read` 用户授权阻塞，未创建应用 |
| Dashboard | 20% | `dashboard/data.json` 仍是泄露修复前的遗留快照，不能代表当前模型；需改为受控 API 数据源 |
| Agent | 5% | 现有浏览器直连 OpenRouter 仅为遗留 Demo，不符合服务端密钥和白名单工具要求 |

> 当前可信离线证据来自 `models/*.csv` 和本日志。旧 Dashboard 中的 0.8 杯 MAE、0.961 R²、69.6%/92.9% 命中率已判定为受标签泄露污染的历史结果，不得继续作为模型能力结论。

---

## 一、数据概况

### 1.1 销售数据

| 维度 | 详情 |
|------|------|
| 数据量 | 149,116 条交易记录 |
| 时间范围 | 2023-01-01 ~ 2023-06-30（181 天） |
| 门店 | 3 家（Lower Manhattan / Hell's Kitchen / Astoria） |
| 产品层级 | 9 品类 → 29 子类型 → 80 SKU |
| 总营收 | ~$698,812 |

### 1.2 补充数据

- **天气数据**：从 Open-Meteo API 获取 NYC 2023 年上半年逐日天气（最高/最低/均温、降水量、日照、风速、湿度等）
- **节假日数据**：手工标注 13 个美国节假日，包含前后效应窗口
- **日历衍生**：星期几、月内周次、月初/月末、季节编码、长周末等

---

## 二、模型方案

### 2.1 核心架构

参考了 [柠季 AI 店长方案调研](https://bytedance.larkoffice.com/wiki/Jyd4w3brDi73Jrk5wNVcpkSPnxg) 和 [销量预测方法调研](https://feishu.doubao.com/docx/R9E4dhfFDoxVoYxaHZLc0ofUnnf) 两份文档的结论：

> 调研指出：Meta-LightGBM 在真实餐饮数据上精度最高，Global Single Model 泛化能力最强。

当前采用 **Global HistGradientBoostingRegressor + GradientBoostingRegressor 加权融合**，不做逐店逐 SKU 独立建模。它不是 LightGBM，下一阶段必须迁移到真实 `lightgbm.LGBMRegressor` 后才能使用 LightGBM 名称。

### 2.2 特征工程（43 个特征，四梯队）

| 梯队 | 类别 | 特征数 | 代表特征 |
|------|------|:---:|------|
| 第一梯队 | 时序销量 | 14 | lag_1/2/3/7/14/28, rolling_mean/std 7/14/28d, 历史同星期均值 |
| 第二梯队 | 日历基础 | 7 | day_of_week, is_weekend, day_of_month, week_of_month, month_start/end |
| 第三梯队 | 天气 | 13 | temperature_max/min/mean, precipitation, is_rain/snow/clear, humidity, wind |
| 第四梯队 | 节假日 | 5 | is_holiday_detailed (13种), is_pre_holiday_1d/3d, is_post_holiday_1d, is_long_weekend |
| 第五梯队 | 品类/价格 | 2 | price_vs_category_avg, sku_share_28d（前 28 天历史滚动占比） |

### 2.3 训练策略

- **交叉验证**：TimeSeriesSplit (n_splits=3)，严格按时间切分，防止未来信息泄露
- **训练/测试切分**：前 174 天训练，最后 7 天（6/24-6/30）测试
- **集成方式**：HGB 权重 0.6 + GB 权重 0.4

### 2.4 关键决策记录

| 决策 | 结论 | 原因 |
|------|------|------|
| 是否加入 holiday_index 连续特征 | **否** | 已有的二值特征（is_holiday_detailed + pre/post 标记）已足够，连续指数反而稀释信号（CV MAE 0.9802 vs 0.9740） |
| 最优预测粒度 | **子类型级（29 种）** | R² 0.991，±10% 命中 85.5%，比 SKU 级稳定，比品类级细粒度 |
| Global vs 逐店建模 | **Global** | 所有门店/SKU 共享模型，泛化能力更强 |

---

## 三、预测结果

### 3.1 当前无泄露 SKU 级结果

| 验证范围 | 样本量 | R² | MAE | RMSE | ±10% 命中 | ±20% 命中 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 2023-06-24 ~ 2023-06-30 | 1,680 | 0.652 | 2.67 杯 | 3.89 杯 | 11.7% | 27.3% |

> 这只是单个 7 天留出窗口，不代表生产精度。旧三层高分结果因 `sku_share` 和同星期均值使用预测日或未来标签而作废。

### 3.2 测试期日总量（6/24-6/30）

| 日期 | 实际 | 预测 | 偏差 |
|------|------|------|------|
| 6/24 (六) | 1,846 | 1,707 | 7.5% |
| 6/25 (日) | 1,802 | 1,690 | 6.2% |
| 6/26 (一) | 1,837 | 1,704 | 7.2% |
| 6/27 (二) | 1,962 | 1,780 | 9.3% |
| 6/28 (三) | 1,531 | 1,603 | 4.7% |
| 6/29 (四) | 1,429 | 1,553 | 8.7% |
| 6/30 (五) | 1,590 | 1,558 | 2.0% |

> 上述误差仅是观测事实。天气、长假和办公人流等解释仍是业务假设，当前数据不足以证明因果。

### 3.3 6/27 销量异常分析

6/27 销量 1,962 杯（全周最高），拆解发现 **Lower Manhattan 单店贡献 +37%**（770 vs 日均 563）。原因推测：
1. 暴雨（18.7mm）迫使办公人群留在室内 → 咖啡需求激增
2. 周二办公到岗率通常较高
3. 曼哈顿金融区特有的室内消费模式

---

## 四、可解释性

- 使用 **TreeSHAP** 评估 HGB 子模型贡献，并校验 `base_value + SHAP = model prediction`（最大误差 `4.79e-14`）
- 当前全局 SHAP top 3：`qty_roll_mean_28d`（1.709 杯）、`qty_roll_mean_7d`（1.161 杯）、`qty_lag_28`（0.346 杯）
- 生成 LLM 根因分析提示词，对接柠季「寻因」需求

---

## 五、文件结构

```
门店预测kaggle数据/
├── Coffee Shop Sales.xlsx          # 原始销售数据
├── nyc_weather_2023_h1.csv         # 补充天气 + 日历特征
├── predict_pipeline.py             # 核心预测 pipeline
├── run_test.py                     # 测试脚本
├── root_cause_prompt.txt           # LLM 根因分析提示词
├── PROJECT_LOG.md                  # 本文件
├── models/
│   ├── hgb_model.pkl               # HGB 模型
│   ├── gb_model.pkl                # GB 模型
│   ├── test_predictions.csv        # SKU 级预测结果
│   ├── daily_predictions.csv       # 日总量预测
│   └── feature_importance.csv      # 特征重要性
└── recruit-restaurant/             # Recruit Restaurant 数据集（另一竞赛）
```

---

## 六、与柠季标准的对齐

| 柠季要求 | 当前状态 | 差距 |
|---------|---------|------|
| 90% 置信度 ±5% | 85.5% 置信度 ±10%（子类型级） | 可进一步提升至门店日总量粒度 |
| SHAP 可解释 | 已实现 TreeSHAP + LLM 根因分析 | ✅ |
| 天气驱动 | 完整 13 维天气特征 | ✅ |
| BOM 订货换算 | 未实现 | 需构造虚拟 BOM 表 |
| 促销因素 | 未纳入 | 数据集中无促销字段 |

---

## 七、可改进方向

1. **门店日总量预测**：当前 SKU → 聚合，可单独训练门店日总量模型，精度预估 ±3-5%
2. **BOM 订货链路**：虚拟成本卡 + 杯量 → 原材料需求，闭环供应链演示
3. **实时叠加层**：结合天气预警、节假日日历做日内动态修正
4. **冷启动处理**：新品用 Chronos/TimesFM 零样本预测
5. **门店外部特征**：POI 数据、写字楼入驻率、周边事件日历
