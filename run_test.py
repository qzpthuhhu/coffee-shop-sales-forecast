"""
测试脚本：运行预测 Pipeline 并输出报告

用法：
    python3 run_test.py          # 运行完整 pipeline
"""
import time


def main():
    print("""
╔══════════════════════════════════════════════════════╗
║   Coffee Shop Sales 销量预测测试                       ║
║   Global LightGBM + SHAP                             ║
╚══════════════════════════════════════════════════════╝
    """)

    start = time.time()
    from predict_pipeline import run_pipeline
    model, result_df, feat_imp = run_pipeline()

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  Pipeline 完成，耗时 {elapsed:.1f} 秒")
    print(f"{'='*60}")

    print(f"\n  ✓ 测试覆盖 {result_df['date'].nunique()} 天")
    print(f"  ✓ 模型: models/lightgbm_model.pkl ({type(model).__name__})")
    print(f"  ✓ SHAP 重要性: models/shap_importance.csv")
    print(f"  ✓ 预测结果: models/test_predictions.csv")
    print(f"  ✓ 日聚合: models/daily_predictions.csv")
    print(f"  ✓ 根因分析: root_cause_prompt.txt")

    print("\n  [INFO] 多维表格: https://bytedance.larkoffice.com/base/BiEub9FWIaxwpXszm6fcQ5tUnmd")


if __name__ == "__main__":
    main()
