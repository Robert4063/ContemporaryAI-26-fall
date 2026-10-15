"""
scripts/plots/plot_evaluation.py
评估结果可视化脚本

输出:
- 各模型超参数对比柱状图
- 模型综合对比 (Accuracy / F1)
- 最优配置表 best_config_summary.csv
- 混淆矩阵
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
RESULTS_DIR = PROJECT_ROOT / "results"
EVAL_DIR = RESULTS_DIR / "evaluation" / "figures"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def plot_hyperparam_comparison():
    """绘制超参数对比柱状图"""
    print("\n" + "="*60)
    print("绘制超参数对比图")
    print("="*60)
    
    merged_file = RESULTS_DIR / "baseline" / "merged_results.csv"
    if not merged_file.exists():
        print("[WARNING] 合并结果文件不存在，跳过")
        return
    
    df = pd.read_csv(merged_file)
    
    # SVM C 参数对比
    svm_df = df[df['model'] == 'svm']
    if len(svm_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 按 C 分组
        for i, split in enumerate(['A', 'B', 'C']):
            split_df = svm_df[svm_df['split'] == split]
            if len(split_df) == 0:
                continue
            
            ax = axes[i]
            
            # 获取唯一版本和 C 值
            versions = split_df['version'].unique()
            c_values = sorted(split_df['C'].unique())
            
            x = np.arange(len(c_values))
            width = 0.2
            
            for j, version in enumerate(versions):
                version_df = split_df[split_df['version'] == version]
                vals = [version_df[version_df['C'] == c]['val_accuracy'].values[0] 
                        if len(version_df[version_df['C'] == c]) > 0 else 0 
                        for c in c_values]
                ax.bar(x + j * width, vals, width, label=version)
            
            ax.set_xlabel('C')
            ax.set_ylabel('Val Accuracy')
            ax.set_title(f'SVM - Split {split}')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(c_values)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = EVAL_DIR / "svm_c_comparison.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] 保存 SVM C 对比图: {output_file}")
    
    # MLP hidden layers 对比
    mlp_df = df[df['model'] == 'mlp']
    if len(mlp_df) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, split in enumerate(['A', 'B', 'C']):
            split_df = mlp_df[mlp_df['split'] == split]
            if len(split_df) == 0:
                continue
            
            ax = axes[i]
            
            # 按 hidden_layer_sizes 分组
            configs = split_df.groupby('hidden_layer_sizes')['val_accuracy'].mean().sort_index()
            
            bars = ax.bar(range(len(configs)), configs.values, color='steelblue')
            ax.set_xlabel('Hidden Layer Sizes')
            ax.set_ylabel('Mean Val Accuracy')
            ax.set_title(f'MLP - Split {split}')
            ax.set_xticks(range(len(configs)))
            ax.set_xticklabels([str(h) for h in configs.index], rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = EVAL_DIR / "mlp_hidden_comparison.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] 保存 MLP hidden 对比图: {output_file}")


def plot_model_comparison():
    """绘制模型综合对比图"""
    print("\n" + "="*60)
    print("绘制模型综合对比图")
    print("="*60)
    
    merged_file = RESULTS_DIR / "baseline" / "merged_results.csv"
    if not merged_file.exists():
        print("[WARNING] 合并结果文件不存在，跳过")
        return
    
    df = pd.read_csv(merged_file)
    
    # 每个模型取最优配置
    best_rows = []
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        model_df = model_df.sort_values('val_accuracy', ascending=False)
        best_rows.append(model_df.iloc[0])
    
    best_df = pd.DataFrame(best_rows)
    
    # 绘制对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['val_accuracy', 'val_f1_macro', 'test_accuracy']
    titles = ['Validation Accuracy', 'Validation F1 (Macro)', 'Test Accuracy']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        colors = ['#1f77b4', '#ff7f0e']
        
        bars = ax.bar(best_df['model'].str.upper(), best_df[metric], color=colors)
        
        # 添加数值标签
        for bar, val in zip(bars, best_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = EVAL_DIR / "model_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] 保存模型对比图: {output_file}")


def create_best_config_summary():
    """创建最优配置汇总表"""
    print("\n" + "="*60)
    print("创建最优配置汇总表")
    print("="*60)
    
    merged_file = RESULTS_DIR / "baseline" / "merged_results.csv"
    if not merged_file.exists():
        print("[WARNING] 合并结果文件不存在，跳过")
        return
    
    df = pd.read_csv(merged_file)
    
    best_rows = []
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        model_df = model_df.sort_values('val_accuracy', ascending=False)
        best = model_df.iloc[0]
        
        # 格式化输出
        row = {
            'Model': model.upper(),
            'Split': best['split'],
            'Version': best['version'],
            'Val_Accuracy': f"{best['val_accuracy']:.4f}",
            'Val_F1': f"{best['val_f1_macro']:.4f}",
            'Test_Accuracy': f"{best['test_accuracy']:.4f}",
            'Test_F1': f"{best['test_f1_macro']:.4f}",
            'Test_Precision': f"{best['test_precision']:.4f}",
            'Test_Recall': f"{best['test_recall']:.4f}"
        }
        
        if model == 'svm':
            row['C'] = best['C']
            row['Alpha'] = f"{best['alpha']:.6f}"
        elif model == 'mlp':
            row['Hidden'] = best['hidden_layer_sizes']
            row['Alpha'] = f"{best['alpha']:.4f}"
        
        best_rows.append(row)
    
    best_df = pd.DataFrame(best_rows)
    
    output_file = RESULTS_DIR / "evaluation" / "best_config_summary.csv"
    best_df.to_csv(output_file, index=False)
    print(f"[OK] 保存最优配置汇总: {output_file}")
    print("\n最优配置:")
    print(best_df.to_string(index=False))


def plot_confusion_matrices():
    """绘制混淆矩阵"""
    print("\n" + "="*60)
    print("绘制混淆矩阵")
    print("="*60)
    
    merged_file = RESULTS_DIR / "baseline" / "merged_results.csv"
    if not merged_file.exists():
        print("[WARNING] 合并结果文件不存在，跳过")
        return
    
    df = pd.read_csv(merged_file)
    
    # 每个模型取最优配置
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        model_df = model_df.sort_values('val_accuracy', ascending=False)
        best = model_df.iloc[0]
        
        # 读取混淆矩阵
        baseline_dir = RESULTS_DIR / "baseline" / model
        split = best['split']
        version = best['version']
        
        if model == 'svm':
            c = best['C']
            result_file = baseline_dir / f"split_{split}" / version / f"results_C_{c}.json"
        else:
            hidden = best['hidden_layer_sizes']
            alpha = best['alpha']
            config_name = f"hidden_{'-'.join(map(str, eval(hidden)))}_alpha_{alpha}"
            result_file = baseline_dir / f"split_{split}" / version / f"results_{config_name}.json"
        
        if not result_file.exists():
            print(f"[WARNING] 结果文件不存在: {result_file}")
            continue
        
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        cm = data['test_metrics']['confusion_matrix']
        
        # 绘制
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{model.upper()} Confusion Matrix\n(Split {split}, {version})')
        
        output_file = EVAL_DIR / f"confusion_matrix_{model}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] 保存混淆矩阵: {output_file}")


def main():
    print("="*60)
    print("绘制评估图表")
    print("="*60)
    
    plot_hyperparam_comparison()
    plot_model_comparison()
    create_best_config_summary()
    plot_confusion_matrices()
    
    print("\n" + "="*60)
    print("评估图表绘制完成!")
    print(f"输出目录: {EVAL_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
