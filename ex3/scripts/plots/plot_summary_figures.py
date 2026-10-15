"""
绘制汇总图表
- 各模型按版本对比
- 模型与清洗版本的验证集 Accuracy 对比
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MultipleLocator
import pandas as pd
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "baseline"
PLOT_DIR = PROJECT_ROOT / "results" / "evaluation" / "figures"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def load_all_results():
    """加载所有结果 CSV 文件"""
    all_results = []
    for csv_file in RESULTS_DIR.glob("*_metrics.csv"):
        df = pd.read_csv(csv_file)
        all_results.append(df)
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

def plot_model_comparison_by_split():
    """每种清洗版本并排显示 SVM/MLP 的验证集准确率均值。"""
    df = load_all_results()
    if df.empty:
        print("No results to plot")
        return
    
    df_val = df[df['eval_set'] == 'val'].copy()
    models = ['svm', 'mlp']
    versions = ['raw', 'body_only', 'subject_body', 'subject_body_author']
    version_labels = ['Raw email', 'Body only', 'Subject + body', 'Author + subject\n+ body']
    colors = {'svm': '#2878B5', 'mlp': '#E88A36'}
    expected_counts = {'svm': 3, 'mlp': 6}
    split_ratios = {'split_A': '80/10/10', 'split_B': '85/5/10', 'split_C': '90/5/5'}

    # 保留原图的均值口径；明确分组和排序，防止模型标签与数据错配。
    summary = df_val.groupby(['split', 'version', 'model']).agg(
        accuracy=('accuracy', 'mean'),
        n_configs=('accuracy', 'count'),
    )
    expected_index = pd.MultiIndex.from_product(
        [list(split_ratios), versions, models], names=['split', 'version', 'model']
    )
    summary = summary.reindex(expected_index).sort_index()
    if summary.isna().any().any():
        raise ValueError('Incomplete model/version results; comparison requires all 24 groups.')
    for model, count in expected_counts.items():
        if not summary.xs(model, level='model')['n_configs'].eq(count).all():
            raise ValueError(f'Expected {count} validation configurations per {model}/split/version.')
    if not df_val['accuracy'].between(0, 1).all():
        raise ValueError('Validation accuracy must be between 0 and 1.')

    # 三张图使用相同纵轴范围，避免不同划分之间的视觉比例不一致。
    lower = max(0.0, np.floor((summary['accuracy'].min() - 0.03) / 0.05) * 0.05)
    for split in ["split_A", "split_B", "split_C"]:
        accuracy = summary.loc[split, 'accuracy'].unstack('model').reindex(
            index=versions, columns=models
        )
        fig, ax = plt.subplots(figsize=(11, 6))
        fig.subplots_adjust(left=0.10, right=0.98, top=0.80, bottom=0.23)
        x_pos = np.arange(len(versions))
        width = 0.32
        for i, model in enumerate(models):
            bars = ax.bar(
                x_pos + (i - 0.5) * width,
                accuracy[model].to_numpy(), width=width,
                color=colors[model], label=model.upper(), zorder=3,
            )
            ax.bar_label(bars, labels=[f'{value:.2%}' for value in accuracy[model]],
                         padding=5, fontsize=10)

        fig.suptitle(f'SVM vs MLP: Validation Accuracy | {split} ({split_ratios[split]})',
                     x=0.10, y=0.96, ha='left', fontsize=16, fontweight='bold')
        fig.text(0.10, 0.89,
                 'Mean across hyperparameter configurations: SVM = 3, MLP = 6 per text version',
                 fontsize=10, color='#555555')
        ax.set_ylabel('Validation accuracy', fontsize=11)
        ax.set_xlabel('Text preprocessing version', labelpad=12, fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(version_labels, fontsize=10)
        ax.set_ylim(lower, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.legend(loc='upper right', ncol=2, frameon=False)
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.25, axis='y')
        ax.spines[['top', 'right']].set_visible(False)
        fig.text(0.10, 0.04,
                 f'Validation data only. Bars show means, not best configurations. Y-axis starts at {lower:.0%}.',
                 fontsize=9, color='#555555')

        plot_path = PLOT_DIR / f"model_comparison_{split}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {plot_path}")

    summary_path = PLOT_DIR.parent / 'model_comparison_summary.csv'
    summary.rename(columns={'accuracy': 'mean_val_accuracy'}).reset_index().to_csv(
        summary_path, index=False
    )
    print(f"Saved: {summary_path}")

def plot_version_effect():
    """绘制文本版本的影响"""
    df = load_all_results()
    if df.empty:
        return
    
    df_val = df[df['eval_set'] == 'val'].copy()
    version_order = ['raw', 'body_only', 'subject_body', 'subject_body_author']
    
    # 按 model 聚合
    for model in ['svm', 'mlp']:
        df_model = df_val[df_val['model'] == model]
        if df_model.empty:
            continue
        
        # 按 version 和 split 聚合
        df_agg = df_model.groupby(['version', 'split'])[['accuracy', 'macro_f1']].mean().reset_index()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        for split in ["split_A", "split_B", "split_C"]:
            df_split = df_agg[df_agg['split'] == split].set_index('version').reindex(version_order)
            if df_split[['accuracy', 'macro_f1']].isna().any().any():
                raise ValueError(f'Incomplete version results: {model}/{split}')
            versions = df_split.index.to_numpy()
            accs = df_split['accuracy'].values
            f1s = df_split['macro_f1'].values
            
            ax1.plot(range(len(versions)), accs, marker='o', label=split, linewidth=2)
            ax2.plot(range(len(versions)), f1s, marker='s', label=split, linewidth=2)
        
        ax1.set_ylabel("Accuracy")
        ax1.set_xlabel("Text Version")
        ax1.set_title(f"{model.upper()}: Accuracy by Text Version")
        ax1.set_xticks(range(4))
        ax1.set_xticklabels(['raw', 'body', 'subj_body', 'subj_body_auth'], rotation=45)
        ax1.legend()
        ax1.set_ylim([0.5, 1.0])
        ax1.grid(True, alpha=0.3)
        
        ax2.set_ylabel("Macro F1")
        ax2.set_xlabel("Text Version")
        ax2.set_title(f"{model.upper()}: Macro F1 by Text Version")
        ax2.set_xticks(range(4))
        ax2.set_xticklabels(['raw', 'body', 'subj_body', 'subj_body_auth'], rotation=45)
        ax2.legend()
        ax2.set_ylim([0.5, 1.0])
        ax2.grid(True, alpha=0.3)
        
        plot_path = PLOT_DIR / f"version_effect_{model}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {plot_path}")

def plot_top_configs():
    """绘制每个 split 的最优配置"""
    df = load_all_results()
    if df.empty:
        return
    
    df_val = df[df['eval_set'] == 'val'].copy()
    
    for split in ["split_A", "split_B", "split_C"]:
        df_split = df_val[df_val['split'] == split].copy()
        if df_split.empty:
            continue
        
        # 获取每个模型类型的前 3 个配置
        top_configs = []
        for model in ['svm', 'mlp']:
            df_model = df_split[df_split['model'] == model].nlargest(3, 'accuracy')
            top_configs.append(df_model)
        
        df_top = pd.concat(top_configs, ignore_index=True) if top_configs else pd.DataFrame()
        if df_top.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(df_top))
        colors = ['#1f77b4' if m == 'svm' else '#ff7f0e' for m in df_top['model']]
        bars = ax.bar(x_pos, df_top['accuracy'], color=colors)
        
        ax.set_ylabel("Validation Accuracy")
        ax.set_title(f"Top Configurations - {split}")
        ax.set_xticks(x_pos)
        labels = [f"{row['model']}-{row['version'][:4]}" for _, row in df_top.iterrows()]
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylim([0.8, 1.0])
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=8)
        
        plot_path = PLOT_DIR / f"top_configs_{split}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {plot_path}")

def main():
    print("Generating Summary Figures...\n")
    
    print("=== Model Comparison by Split ===")
    plot_model_comparison_by_split()
    
    print("\n=== Text Version Effect ===")
    plot_version_effect()
    
    print("\n=== Top Configurations ===")
    plot_top_configs()
    
    print(f"\nAll figures saved to: {PLOT_DIR}")

if __name__ == "__main__":
    main()
