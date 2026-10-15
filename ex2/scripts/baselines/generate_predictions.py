"""
scripts/baselines/generate_predictions.py
生成最终预测脚本

选择最优配置，用其模型对 test_data_unlabeled.csv 预测
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_PROCESSED = PROJECT_ROOT / "data-processed"
DATA_RAW = PROJECT_ROOT / "data-Project1"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


def load_best_config(model_name):
    """加载最优配置"""
    merged_file = RESULTS_DIR / "baseline" / "merged_results.csv"
    
    if not merged_file.exists():
        raise FileNotFoundError("合并结果文件不存在，请先运行训练")
    
    df = pd.read_csv(merged_file)
    model_df = df[df['model'] == model_name]
    model_df = model_df.sort_values('val_accuracy', ascending=False)
    
    if len(model_df) == 0:
        raise ValueError(f"找不到 {model_name} 的结果")
    
    best = model_df.iloc[0]
    return best


def generate_predictions_for_model(model_name):
    """为单个模型生成预测"""
    print(f"\n{'='*60}")
    print(f"生成 {model_name.upper()} 预测")
    print(f"{'='*60}")
    
    # 加载最优配置
    best = load_best_config(model_name)
    split = best['split']
    version = best['version']
    
    print(f"最优配置:")
    print(f"  划分: {split}")
    print(f"  版本: {version}")
    print(f"  Val Accuracy: {best['val_accuracy']:.4f}")
    print(f"  Test Accuracy: {best['test_accuracy']:.4f}")
    
    # 加载模型
    model_dir = MODELS_DIR / model_name / f"split_{split}" / version
    
    if model_name == 'svm':
        C = best['C']
        model_dir = model_dir / f"C_{C}"
    else:
        hidden = best['hidden_layer_sizes']
        alpha = best['alpha']
        config_name = f"hidden_{'-'.join(map(str, eval(hidden)))}_alpha_{alpha}"
        model_dir = model_dir / config_name
    
    model_file = model_dir / "model.pkl"
    
    if not model_file.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_file}")
    
    print(f"\n加载模型: {model_file}")
    with open(model_file, 'rb') as f:
        vectorizer, clf = pickle.load(f)
    
    # 加载无标签测试数据
    unlabeled_file = DATA_PROCESSED / f"split_{split}" / version / "unlabeled.csv"
    unlabeled_df = pd.read_csv(unlabeled_file)
    
    print(f"无标签测试数据: {len(unlabeled_df)} 条")
    
    # 转换特征
    X_unlabeled = vectorizer.transform(unlabeled_df['text'].fillna('').astype(str))
    
    # 预测
    print("进行预测...")
    predictions = clf.predict(X_unlabeled)
    
    # 保存单个模型预测
    model_pred_file = PREDICTIONS_DIR / f"predictions_{model_name}.csv"
    pd.DataFrame(predictions).to_csv(model_pred_file, index=False, header=False)
    print(f"保存模型预测: {model_pred_file}")
    
    return predictions


def generate_final_predictions():
    """生成最终提交文件"""
    print("\n" + "="*60)
    print("生成最终提交预测")
    print("="*60)
    
    # 加载两个模型的预测
    svm_pred_file = PREDICTIONS_DIR / "predictions_svm.csv"
    mlp_pred_file = PREDICTIONS_DIR / "predictions_mlp.csv"
    
    # 按 val_accuracy 选择最终模型
    merged_file = RESULTS_DIR / "baseline" / "merged_results.csv"
    df = pd.read_csv(merged_file)
    
    best_rows = []
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        model_df = model_df.sort_values('val_accuracy', ascending=False)
        best_rows.append(model_df.iloc[0])
    
    best_df = pd.DataFrame(best_rows)
    best_overall = best_df.sort_values('val_accuracy', ascending=False).iloc[0]
    
    print("\n各模型最优配置:")
    print(best_df[['model', 'split', 'version', 'val_accuracy', 'test_accuracy']].to_string(index=False))
    
    # 选择最终模型
    if best_overall['model'] == 'svm':
        final_pred_file = svm_pred_file
        final_model = 'svm'
    else:
        final_pred_file = mlp_pred_file
        final_model = 'mlp'
    
    print(f"\n选择 {final_model.upper()} 作为最终提交模型 (val_acc={best_overall['val_accuracy']:.4f})")
    
    # 复制到 predictions.csv
    final_output = PREDICTIONS_DIR / "predictions.csv"
    
    if final_pred_file.exists():
        pred_df = pd.read_csv(final_pred_file, header=None)
        pred_df.to_csv(final_output, index=False, header=False)
        print(f"保存最终预测: {final_output}")
    else:
        raise FileNotFoundError(f"预测文件不存在: {final_pred_file}")
    
    # 验证预测格式
    print("\n验证预测格式:")
    print(f"  行数: {len(pred_df)}")
    print(f"  预期行数: 2457")
    print(f"  标签取值: {sorted(pred_df[0].unique())}")
    
    if len(pred_df) != 2457:
        print(f"[WARNING] 预测行数 {len(pred_df)} != 2457")
    
    return final_output


def main():
    print("="*60)
    print("生成预测")
    print("="*60)
    
    # 生成各模型预测
    for model in ['svm', 'mlp']:
        try:
            generate_predictions_for_model(model)
        except Exception as e:
            print(f"[ERROR] {model.upper()} 预测失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 生成最终提交
    try:
        final_file = generate_final_predictions()
        print(f"\n[OK] 最终预测已生成: {final_file}")
    except Exception as e:
        print(f"[ERROR] 最终预测生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n预测生成完成!")


if __name__ == "__main__":
    main()
