"""
plot_mlp_loss_compare.py
绘制 MLP 训练集 loss 曲线 (sklearn loss_curve_, 对数纵轴) 的三种对比:
  1. 不同划分方法 (Split A vs B vs C)
  2. 不同预处理方法 (raw/body_only/subject_body/subject_body_author)
  3. 不同超参数 (hidden layer size, alpha)
"""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端,避免 Qt 插件问题
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOSS_DIR = ROOT / "results" / "loss"
FIG_DIR = ROOT / "results" / "evaluation" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 最优配置: Split B, subject_body_author, hidden=(50,), alpha=0.001
best_split = "B"
best_version = "subject_body_author"
best_hidden = "(50,)"
best_alpha = 0.001

# 1. 划分方法对比 (Split A vs B vs C)
print("=== 图1: 不同划分方法的 loss 曲线 ===")
fig, ax = plt.subplots(figsize=(10, 6))
colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}
for split in ['A', 'B', 'C']:
    path = LOSS_DIR / f"MLP_{split}_{best_version}.csv"
    if path.exists():
        df = pd.read_csv(path)
        # 筛选最优超参数
        subset = df[(df['hidden_layer_sizes'] == best_hidden) & 
                    (df['alpha'] == best_alpha)]
        if not subset.empty:
            subset = subset.sort_values('epoch')
            ax.plot(subset['epoch'], subset['loss'], 
                   label=f"Split {split}", linewidth=2.5, color=colors[split])
            print(f"  Split {split}: {len(subset)} epochs")

ax.set_xlabel("Epoch", fontsize=12, fontweight='bold')
ax.set_ylabel("Training Loss (log scale)", fontsize=12, fontweight='bold')
ax.set_yscale("log")  # 与 plot_loss.py 保持一致
ax.set_title(f"MLP Training Loss Across Different Data Splits\n(subject_body_author, hidden=(50,), alpha=0.001)", 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(FIG_DIR / "MLP_train_loss_splits.png", dpi=300, bbox_inches='tight')
print(f"✓ 保存 -> {FIG_DIR / 'MLP_train_loss_splits.png'}\n")
plt.close()

# 2. 预处理方法对比 (raw/body_only/subject_body/subject_body_author)
print("=== 图2: 不同预处理方法的 loss 曲线 ===")
fig, ax = plt.subplots(figsize=(10, 6))
versions = ['raw', 'body_only', 'subject_body', 'subject_body_author']
colors_v = {'raw': '#d62728', 'body_only': '#9467bd', 
           'subject_body': '#8c564b', 'subject_body_author': '#e377c2'}
for version in versions:
    path = LOSS_DIR / f"MLP_{best_split}_{version}.csv"
    if path.exists():
        df = pd.read_csv(path)
        subset = df[(df['hidden_layer_sizes'] == best_hidden) & 
                    (df['alpha'] == best_alpha)]
        if not subset.empty:
            subset = subset.sort_values('epoch')
            ax.plot(subset['epoch'], subset['loss'], 
                   label=version, linewidth=2.5, color=colors_v[version])
            print(f"  {version}: {len(subset)} epochs")

ax.set_xlabel("Epoch", fontsize=12, fontweight='bold')
ax.set_ylabel("Training Loss (log scale)", fontsize=12, fontweight='bold')
ax.set_yscale("log")  # 与 plot_loss.py 保持一致
ax.set_title(f"MLP Training Loss Across Different Preprocessing Methods\n(Split B, hidden=(50,), alpha=0.001)", 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(FIG_DIR / "MLP_train_loss_versions.png", dpi=300, bbox_inches='tight')
print(f"✓ 保存 -> {FIG_DIR / 'MLP_train_loss_versions.png'}\n")
plt.close()

# 3. 超参数对比 (hidden layer size 和 alpha)
print("=== 图3: 不同超参数的 loss 曲线 ===")
path = LOSS_DIR / f"MLP_{best_split}_{best_version}.csv"
df = pd.read_csv(path)

# 提取所有独特的超参数组合
unique_configs = df[['hidden_layer_sizes', 'alpha']].drop_duplicates()
print(f"  找到 {len(unique_configs)} 组超参数组合")

fig, ax = plt.subplots(figsize=(12, 7))
linestyles = {0.0001: '-', 0.001: '--', 0.01: '-.', 0.1: ':'}
for i, (_, row) in enumerate(unique_configs.iterrows()):
    hidden = row['hidden_layer_sizes']
    alpha = row['alpha']
    subset = df[(df['hidden_layer_sizes'] == hidden) & (df['alpha'] == alpha)]
    if not subset.empty:
        subset = subset.sort_values('epoch')
        label = f"hidden={hidden}, alpha={alpha}"
        ls = linestyles.get(alpha, '-')
        ax.plot(subset['epoch'], subset['loss'], 
               label=label, linewidth=2.2, linestyle=ls, alpha=0.8)
        print(f"    {label}: {len(subset)} epochs")

ax.set_xlabel("Epoch", fontsize=12, fontweight='bold')
ax.set_ylabel("Training Loss (log scale)", fontsize=12, fontweight='bold')
ax.set_yscale("log")  # 与 plot_loss.py 保持一致
ax.set_title(f"MLP Training Loss Across Different Hyperparameters\n(Split B, subject_body_author)", 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.95, ncol=2, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(FIG_DIR / "MLP_train_loss_hyperparams.png", dpi=300, bbox_inches='tight')
print(f"✓ 保存 -> {FIG_DIR / 'MLP_train_loss_hyperparams.png'}\n")
plt.close()

print("=== 完成 ===")
print("已生成 3 张图:")
print("  1. MLP_train_loss_splits.png (划分方法对比)")
print("  2. MLP_train_loss_versions.png (预处理方法对比)")
print("  3. MLP_train_loss_hyperparams.png (超参数对比)")
