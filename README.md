# Coffee Shop Sales Forecast

A sparsity-aware direct multi-step demand forecasting pipeline for small-scale retail environments, evaluated on a real-world NYC coffee shop dataset.

## Overview

This project implements a **Global LightGBM + Tweedie loss + sparsity-aware features** pipeline for daily store × SKU sales forecasting. It demonstrates that careful feature engineering can outperform deep learning architectures (DeepAR, PatchTST, TFT) on small-scale, highly intermittent retail demand data.

**Key results** (5-window rolling backtest, 8,400 predictions):

| Metric | This method | DeepAR | PatchTST | TFT |
|--------|:-----------:|:------:|:--------:|:---:|
| MAE | **2.411** | 3.13 | 2.84 | 3.14 |
| WAPE | 34.7% | — | — | — |
| R² | 0.661 | — | — | — |
| Direction hit | 61.4% | — | — | — |

## Dataset

| Dimension | Value |
|-----------|-------|
| Transactions | 149,116 |
| Date range | 2023-01-01 ~ 2023-06-30 (181 days) |
| Stores | 3 (Astoria, Lower Manhattan, Hell's Kitchen) |
| Products | 80 SKUs across 9 categories |
| Store-SKU pairs | 240 (full panel filled with zeros) |

Source: [Kaggle — Coffee Shop Sales](https://www.kaggle.com/datasets/ahmedabbas757/coffee-sales), enriched with NYC weather (Open-Meteo) and holiday calendar features.

## Architecture

```
Coffee Shop Sales.xlsx
        │
        ▼
  load_and_preprocess()          # aggregate → store × SKU × day, fill zeros
        │
        ▼
  add_history_features()         # 53 features across 6 tiers
   ├── Temporal: lag 1/2/3/7/14/28, rolling mean/std 7/14/28d, DOW avg
   ├── Calendar: weekday, weekend, month, week-of-month, season
   ├── Weather: temp (min/max/mean), precipitation, humidity, wind
   ├── Holiday: 13 US holidays + pre/post effects
   ├── Price/Category: price_vs_category_avg, sku_share_28d
   └── Identity & Sparsity: store, product, category, horizon, zero_rate, ADI, mean_qty, is_sparse
        │
        ▼
  Global LightGBM (Tweedie, power=1.3)
   ├── Direct multi-step (T+1 … T+7, one shared model with horizon feature)
   ├── Quantile models α=0.1/0.5/0.9 → P10/P50/P90
   └── Split conformal calibration
        │
        ▼
  Store-bias correction + TreeSHAP per-prediction explanations
```

## Repository Structure

```
.
├── Coffee Shop Sales.xlsx          # Raw sales data
├── nyc_weather_2023_h1.csv         # Weather + holiday features
├── requirements.txt
├── predict_pipeline.py             # Core training & inference pipeline
├── backtest_pipeline.py            # 5-window rolling-origin backtest + baselines
├── production_sparse.py            # Production model (Tweedie + sparsity labels)
├── first_round_model_experiments.py    # Direct multi-step + loss ablation
├── second_round_model_experiments.py   # Feature ablation (EWM, sparsity, hierarchy)
├── third_round_sparse_expert.py         # Sparse/dense expert routing
├── fourth_round_sparse_features.py      # Sparsity labels as global features (winner)
├── gluonts_bench.py                      # DeepAR / PatchTST / TFT benchmarks
├── deep_learning_benchmark.py            # DL benchmark orchestration
├── export_dashboard_data.py              # Dashboard data export
├── run_test.py                           # Quick pipeline runner
├── test_*.py                             # Regression & contract tests (9 files)
├── dashboard/
│   ├── index.html                        # Interactive dashboard (mobile-first)
│   ├── data.json                         # Aggregated backtest results
│   └── explanations/                     # 35 daily SHAP explanation shards
└── reports/model_quality/
    ├── rolling_backtest_predictions.csv  # 8,400 backtest predictions
    ├── rolling_backtest_quality.csv      # Quality metrics
    ├── rolling_backtest_shap.csv.gz      # Per-prediction TreeSHAP values
    ├── rolling_backtest_scenarios.csv.gz # What-if scenario responses
    ├── first_round/  second_round/  third_round/  fourth_round/
    └── deep_learning_benchmark/
```

## Quick Start

```bash
pip install -r requirements.txt

# Run the production pipeline (train + predict + save)
python3 predict_pipeline.py

# Run 5-window rolling backtest with baselines
python3 backtest_pipeline.py

# Reproduce the sparsity-label experiment (4th round, the winning model)
python3 fourth_round_sparse_features.py

# Run deep learning baselines (DeepAR, PatchTST, TFT)
python3 gluonts_bench.py

# Run all tests
python3 -m pytest test_*.py -v
```

## Experimental Rounds

| Round | Approach | MAE | Δ vs Round 1 |
|-------|----------|:---:|:---:|
| Round 1 | Direct multi-step Tweedie 1.3 | 2.442 | baseline |
| Round 2 | + EWM / sparsity / hierarchy features | 2.447 | +0.19% |
| Round 3 | Sparse/dense expert routing | 2.468 | +1.1% |
| **Round 4** | **+ sparsity labels (zero_rate, ADI, mean_qty, is_sparse)** | **2.411** | **−1.29%** |

Round 4 is the winning approach: it's the first round where **both sparse (−4.8%) and dense (−1.1%) SKU segments improve simultaneously**, by injecting sparsity metadata into a single global model rather than splitting training data.

## Explainability

- **TreeSHAP**: Every prediction has 53 signed feature contributions, verified additivity (`base + Σ SHAP = model output`, max error 7.46e-14)
- **Conformal intervals**: P10–P90 cover 80.6% of actuals
- **Dashboard**: Interactive SHAP waterfall plots, what-if scenario simulation, high-risk SKU ranking

## Citation

This work is submitted to *Expert Systems with Applications*. The accompanying paper and full reproducibility code are available at:
- Paper repo: https://github.com/qzpthuhhu/eswa-retail-forecasting

## License

This project is provided for research and educational purposes. The Coffee Shop Sales dataset is sourced from Kaggle.
