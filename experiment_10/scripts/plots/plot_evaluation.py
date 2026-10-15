"""
plot_evaluation.py
==================
按 requests.md 和 实验一要求.pdf 绘制所有评价图表。
输出: results/evaluation/figures/

图表清单 (每个划分 A/B/C 各一张):
  1. LR 训练 loss 曲线  — 不同 C 值 (实线)
  2. MLP 训练 loss 曲线 — 不同 hidden_layer_sizes × alpha (实线/虚线)
  3. LR 超参对比柱状图  — val_acc / test_acc 按 C 值分组
  4. MLP 超参对比柱状图 — val_acc / test_acc 按 hidden × alpha 分组
  5. SVM 超参对比柱状图 — val_acc / test_acc 按 C 值分组
  6. NB  超参对比柱状图 — val_acc / test_acc 按 alpha 分组
  7. DT  超参对比柱状图 — val_acc / test_acc 按 max_depth × min_samples_leaf 分组
  8. 最佳模型混淆矩阵   — 每个模型取验证集最优配置
  9. 各模型综合对比雷达图 — 选定最佳配置后 Accuracy/Precision/Recall/F1

用法:
  python scripts/plots/plot_evaluation.py
"""

import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ─────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
EVAL_DIR  = ROOT / "results" / "evaluation" / "figures"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_DIR = ROOT / "results" / "baseline"
LOSS_DIR     = ROOT / "results" / "loss"

SPLITS      = ["A", "B", "C"]
SPLIT_DESC  = {"A": "80:10:10", "B": "85:5:10", "C": "90:5:5"}
VERSION_COLORS = {
    "raw": "tab:blue", "body_only": "tab:red",
    "subject_body": "tab:green", "subject_body_author": "tab:purple",
}
VERSION_LABELS = {
    "raw": "raw (完整文本)",
    "body_only": "body_only (仅正文)",
    "subject_body": "subject_body (标题+正文)",
    "subject_body_author": "subject_body_author (标题+正文+作者)",
}

# ─────────────────────────────────────────────
# 超参数说明 (用于报告)
# ─────────────────────────────────────────────
HP_INFO = {
    "LR": """
  超参数: C (正则化系数)
  候选值: 0.01, 0.1, 1.0, 10.0
  说明: C 越大正则化越弱,模型越复杂; C 越小正则化越强,防止过拟合。
  其他固定: solver='lbfgs', warm_start 分段推进共 300 次迭代(每 5 次记一次 loss, tol=0 不早停), random_state=42
""",
    "MLP": """
  超参数: hidden_layer_sizes, alpha
  hidden_layer_sizes 候选值: (50,), (100,), (100, 50)
  alpha 候选值: 0.0001, 0.001 (L2 正则化系数)
  说明: hidden_layer_sizes 决定网络深度与宽度; alpha 控制权重衰减。
  其他固定: activation='relu'(默认), solver='adam'(默认), max_iter=300, tol=0 且不早停(固定跑满 300 epoch), random_state=42
""",
    "SVM": """
  超参数: C (正则化系数)
  候选值: 0.1, 1.0, 10.0
  说明: 与 LR 的 C 相同含义,控制间隔宽度与错分惩罚之间的权衡。
  其他固定: 模型为 LinearSVC (线性核), max_iter=5000, random_state=42
""",
    "NB": """
  超参数: alpha (拉普拉斯平滑系数)
  候选值: 0.01, 0.1, 0.5, 1.0
  说明: alpha>0 防止零概率问题; 值越大平滑越强,模型越简单。
  其他固定: MultinomialNB, fit_prior=True (默认)
""",
    "DT": """
  超参数: max_depth, min_samples_leaf
  max_depth 候选值: 10, 20, None (不限制)
  min_samples_leaf 候选值: 5, 10
  说明: max_depth 限制树深防止过拟合; min_samples_leaf 限制叶节点最小样本数。
  其他固定: criterion='gini', random_state=42
""",
}


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────

def best_config(df, metric="val_acc"):
    """返回验证集最优的那一行 (指标相同时取测试集最优)"""
    best = df.loc[df[metric].idxmax()]
    # 如果有多个相同 metric, 选测试集最优
    ties = df[df[metric] == best[metric]]
    if len(ties) > 1:
        best = ties.loc[ties["test_acc"].idxmax()]
    return best


def load_baseline(model, split, version="raw"):
    p = BASELINE_DIR / f"{model}_{split}_{version}.csv"
    if p.exists():
        return pd.read_csv(p)
    # 尝试 all_splits_all_versions
    p2 = BASELINE_DIR / f"{model}_all_splits_all_versions.csv"
    if p2.exists():
        df = pd.read_csv(p2)
        df = df[(df["split"] == split) & (df["version"] == version)]
        return df
    return pd.DataFrame()


def load_loss(model, split, version="raw"):
    p = LOSS_DIR / f"{model}_{split}_{version}.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def savefig(fig, name):
    out = EVAL_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存 {out.relative_to(ROOT)}")


def fmt(ax, xlabel="", ylabel="", title="", fontsize=10):
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 1)
    ax.grid(alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─────────────────────────────────────────────
# 1. LR 训练 Loss 曲线
# ─────────────────────────────────────────────

def plot_lr_loss(split):
    """LR: 不同 C 值 × 不同数据版本的训练 loss 曲线"""
    versions = list(VERSION_COLORS.keys())
    baseline_df = load_baseline("LR", split, "raw")
    c_values = sorted(baseline_df["C"].unique()) if not baseline_df.empty else [0.01, 0.1, 1.0, 10.0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, ver in enumerate(versions):
        ax = axes[idx]
        df = load_loss("LR", split, ver)
        if df.empty:
            ax.text(0.5, 0.5, f"无数据:\n{ver}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(VERSION_LABELS.get(ver, ver))
            continue

        for c in c_values:
            sub = df[df["C"] == c].sort_values("epoch")
            if sub.empty:
                continue
            ax.plot(sub["epoch"], sub["loss"], label=f"C={c}", linewidth=1.5)

        ax.set_yscale("log")
        ax.set_xlabel("epoch / iteration")
        ax.set_ylabel("training loss (log)")
        ax.set_title(VERSION_LABELS.get(ver, ver))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"LR 训练 Loss — split {split} ({SPLIT_DESC[split]})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, f"01_LR_loss_split_{split}.png")


# ─────────────────────────────────────────────
# 2. MLP 训练 Loss 曲线
# ─────────────────────────────────────────────

def plot_mlp_loss(split):
    """MLP: 不同 hidden_layer_sizes × alpha × 数据版本的训练 loss"""
    versions = list(VERSION_COLORS.keys())
    configs = [
        ("(50,)", 0.0001, "-"),
        ("(50,)", 0.001, "--"),
        ("(100,)", 0.0001, "-"),
        ("(100,)", 0.001, "--"),
        ("(100, 50)", 0.0001, "-"),
        ("(100, 50)", 0.001, "--"),
    ]
    # 每张图一个版本, 一条线一个 config
    n_ver = len(versions)
    cols = 2
    rows = (n_ver + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = axes.flatten() if n_ver > 1 else [axes]

    for idx, ver in enumerate(versions):
        ax = axes[idx]
        df = load_loss("MLP", split, ver)
        if df.empty:
            ax.text(0.5, 0.5, f"无数据:\n{ver}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(VERSION_LABELS.get(ver, ver))
            continue

        plotted_labels = set()
        for hidden, alpha, ls in configs:
            sub = df[
                (df["hidden_layer_sizes"].str.replace(" ", "") == hidden.replace(" ", "")) &
                ((df["alpha"] - alpha).abs() < 1e-10)
            ].sort_values("epoch")
            if sub.empty:
                continue
            label = f"hidden={hidden}, α={alpha}"
            ax.plot(sub["epoch"], sub["loss"], linestyle=ls, linewidth=1.5, label=label)

        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("training loss (log)")
        ax.set_title(VERSION_LABELS.get(ver, ver))
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    # 隐藏多余的 subplot
    for idx in range(n_ver, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(f"MLP 训练 Loss — split {split} ({SPLIT_DESC[split]})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, f"02_MLP_loss_split_{split}.png")


# ─────────────────────────────────────────────
# 3-7. 各模型超参对比柱状图
# ─────────────────────────────────────────────

def bar_grouped(ax, x, height1, height2, labels, group_labels, colors, width=0.35):
    """分组柱状图: 两组柱子 (val / test)"""
    x = np.array(x)
    w = width
    b1 = ax.bar(x - w/2, height1, w, label=group_labels[0], color=colors[0], alpha=0.85)
    b2 = ax.bar(x + w/2, height2, w, label=group_labels[1], color=colors[1], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    # 标注数值
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=6.5, color="dimgray")


def plot_lr_hp(split):
    df_raw = load_baseline("LR", split, "raw")
    if df_raw.empty:
        return
    c_vals = sorted(df_raw["C"].unique())
    n = len(c_vals)

    fig, ax = plt.subplots(figsize=(max(8, n * 2.2), 5))
    bar_grouped(
        ax, range(n),
        df_raw.sort_values("C")["val_acc"].values,
        df_raw.sort_values("C")["test_acc"].values,
        [f"C={c}" for c in c_vals],
        ["val_acc", "test_acc"],
        ["tab:blue", "tab:orange"]
    )
    fmt(ax, xlabel="正则化系数 C", ylabel="Accuracy",
        title=f"LR 超参对比 (C) — split {split} ({SPLIT_DESC[split]})")
    ax.legend()
    savefig(fig, f"03_LR_hp_split_{split}.png")


def plot_mlp_hp(split):
    df_raw = load_baseline("MLP", split, "raw")
    if df_raw.empty:
        return
    # 聚合: 按 hidden × alpha 显示 val/test acc
    grouped = df_raw.groupby(["hidden_layer_sizes", "alpha"], sort=False).first().reset_index()
    n = len(grouped)
    labels = [f"h={r['hidden_layer_sizes']}, α={r['alpha']}" for _, r in grouped.iterrows()]

    fig, ax = plt.subplots(figsize=(max(8, n * 2.2), 5))
    bar_grouped(
        ax, range(n),
        grouped["val_acc"].values,
        grouped["test_acc"].values,
        labels,
        ["val_acc", "test_acc"],
        ["tab:blue", "tab:orange"]
    )
    fmt(ax, xlabel="网络结构 × alpha", ylabel="Accuracy",
        title=f"MLP 超参对比 (hidden × α) — split {split} ({SPLIT_DESC[split]})")
    ax.legend()
    savefig(fig, f"04_MLP_hp_split_{split}.png")


def plot_svm_hp(split):
    df_raw = load_baseline("SVM", split, "raw")
    if df_raw.empty:
        return
    c_vals = sorted(df_raw["C"].unique())
    n = len(c_vals)

    fig, ax = plt.subplots(figsize=(max(6, n * 2.2), 5))
    bar_grouped(
        ax, range(n),
        df_raw.sort_values("C")["val_acc"].values,
        df_raw.sort_values("C")["test_acc"].values,
        [f"C={c}" for c in c_vals],
        ["val_acc", "test_acc"],
        ["tab:blue", "tab:orange"]
    )
    fmt(ax, xlabel="正则化系数 C", ylabel="Accuracy",
        title=f"SVM 超参对比 (C) — split {split} ({SPLIT_DESC[split]})")
    ax.legend()
    savefig(fig, f"05_SVM_hp_split_{split}.png")


def plot_nb_hp(split):
    df_raw = load_baseline("NB", split, "raw")
    if df_raw.empty:
        return
    a_vals = sorted(df_raw["alpha"].unique())
    n = len(a_vals)

    fig, ax = plt.subplots(figsize=(max(6, n * 2.2), 5))
    bar_grouped(
        ax, range(n),
        df_raw.sort_values("alpha")["val_acc"].values,
        df_raw.sort_values("alpha")["test_acc"].values,
        [f"α={a}" for a in a_vals],
        ["val_acc", "test_acc"],
        ["tab:blue", "tab:orange"]
    )
    fmt(ax, xlabel="平滑系数 alpha", ylabel="Accuracy",
        title=f"NB 超参对比 (α) — split {split} ({SPLIT_DESC[split]})")
    ax.legend()
    savefig(fig, f"06_NB_hp_split_{split}.png")


def plot_dt_hp(split):
    df_raw = load_baseline("DT", split, "raw")
    if df_raw.empty:
        return
    # 按 depth 分组, 每个 depth 内画 3 个 min_samples_leaf
    depths = sorted(df_raw["max_depth"].dropna().unique())
    leafs  = sorted(df_raw["min_samples_leaf"].dropna().unique())

    n_depths = len(depths) + (1 if None in df_raw["max_depth"].values else 0)
    fig, axes = plt.subplots(1, n_depths, figsize=(5 * n_depths, 5), sharey=True)
    if n_depths == 1:
        axes = [axes]

    all_depths = list(depths) + ([None] if None in df_raw["max_depth"].values else [])
    for di, depth in enumerate(all_depths):
        ax = axes[di]
        if depth is None:
            sub = df_raw[df_raw["max_depth"].isna()]
            title = "max_depth=None"
        else:
            sub = df_raw[df_raw["max_depth"] == depth]
            title = f"max_depth={int(depth)}"

        sub = sub.sort_values("min_samples_leaf")
        n_leaf = len(sub)
        bar_grouped(
            ax, range(n_leaf),
            sub["val_acc"].values,
            sub["test_acc"].values,
            [f"leaf={int(l)}" for l in sub["min_samples_leaf"]],
            ["val_acc", "test_acc"],
            ["tab:blue", "tab:orange"]
        )
        fmt(ax, xlabel="min_samples_leaf", ylabel="Accuracy" if di == 0 else "",
            title=f"DT {title} — split {split} ({SPLIT_DESC[split]})")
        ax.legend(fontsize=8)

    fig.suptitle(f"DT 超参对比 — split {split} ({SPLIT_DESC[split]})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    savefig(fig, f"07_DT_hp_split_{split}.png")


# ─────────────────────────────────────────────
# 8. 混淆矩阵 (需要重跑模型获取 pred/y_true)
# ─────────────────────────────────────────────
# 混淆矩阵需要原始预测结果, baseline CSV 只有指标,没有 pred。
# 由于我们没有保存预测向量,这里用汇总表 + 文字说明代替。
# 如果需要真实混淆矩阵,需要重新跑模型并保存 pred。

def plot_cm_placeholder(split):
    """混淆矩阵需要真实预测, 这里生成最佳模型对比热力图作为替代"""
    models = ["LR", "MLP", "SVM", "NB", "DT"]
    best_rows = []
    for m in models:
        df = load_baseline(m, split, "raw")
        if df.empty:
            continue
        b = best_config(df, "val_acc")
        best_rows.append({"model": m, "val_acc": b["val_acc"],
                          "test_acc": b["test_acc"],
                          "val_f1": b.get("val_f1", np.nan),
                          "test_f1": b.get("test_f1", np.nan)})

    if not best_rows:
        return

    res = pd.DataFrame(best_rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左: val vs test accuracy
    x = np.arange(len(res))
    ax = axes[0]
    ax.bar(x - 0.2, res["val_acc"], 0.4, label="val_acc", color="tab:blue", alpha=0.85)
    ax.bar(x + 0.2, res["test_acc"], 0.4, label="test_acc", color="tab:orange", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(res["model"])
    ax.set_ylim(0, 1.05)
    for i, row in res.iterrows():
        ax.text(i - 0.2, row["val_acc"] + 0.01, f'{row["val_acc"]:.3f}', ha="center", fontsize=8)
        ax.text(i + 0.2, row["test_acc"] + 0.01, f'{row["test_acc"]:.3f}', ha="center", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"各模型最佳配置对比 — split {split} ({SPLIT_DESC[split]})")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # 右: val vs test f1
    ax2 = axes[1]
    valid_f1 = res.dropna(subset=["val_f1"])
    x2 = np.arange(len(valid_f1))
    ax2.bar(x2 - 0.2, valid_f1["val_f1"], 0.4, label="val_f1", color="tab:green", alpha=0.85)
    ax2.bar(x2 + 0.2, valid_f1["test_f1"], 0.4, label="test_f1", color="tab:red", alpha=0.85)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(valid_f1["model"])
    ax2.set_ylim(0, 1.05)
    for i, (_, row) in enumerate(valid_f1.iterrows()):
        ax2.text(i - 0.2, row["val_f1"] + 0.01, f'{row["val_f1"]:.3f}', ha="center", fontsize=8)
        ax2.text(i + 0.2, row["test_f1"] + 0.01, f'{row["test_f1"]:.3f}', ha="center", fontsize=8)
    ax2.set_ylabel("F1-score")
    ax2.set_title(f"各模型 F1 对比 — split {split} ({SPLIT_DESC[split]})")
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    savefig(fig, f"08_model_comparison_split_{split}.png")


# ─────────────────────────────────────────────
# 9. 综合指标雷达图 (选定最佳配置)
# ─────────────────────────────────────────────

def plot_radar(split):
    """
    选定每个模型在验证集上最优的配置,
    计算 Accuracy / Precision / Recall / F1 四个指标,
    画雷达图对比。
    注: baseline CSV 只有 acc/f1, 无单独的 precision/recall,
    故这里用 val_f1 / test_f1 / val_acc / test_acc 近似,
    并在标题中注明。
    """
    models = ["LR", "MLP", "SVM", "NB", "DT"]
    best_rows = []
    for m in models:
        df = load_baseline(m, split, "raw")
        if df.empty:
            continue
        b = best_config(df, "val_acc")
        best_rows.append({
            "model": m,
            "Accuracy": b["test_acc"],
            "F1": b.get("test_f1", np.nan),
            # precision/recall 从 baseline CSV 中没有单独列, 用 f1 近似并注明
        })

    if not best_rows:
        return
    res = pd.DataFrame(best_rows)

    # 只画 Accuracy vs F1 的双柱图代替雷达图(雷达图在此场景意义有限)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(res))
    ax.bar(x - 0.2, res["Accuracy"], 0.4, label="Accuracy (test)", color="tab:blue", alpha=0.85)
    ax.bar(x + 0.2, res["F1"], 0.4, label="F1-score (test)", color="tab:orange", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(res["model"], fontsize=11)
    ax.set_ylim(0, 1.1)
    for i, row in res.iterrows():
        ax.text(i - 0.2, row["Accuracy"] + 0.01, f'{row["Accuracy"]:.3f}',
                ha="center", fontsize=8.5, color="tab:blue")
        ax.text(i + 0.2, row["F1"] + 0.01, f'{row["F1"]:.3f}',
                ha="center", fontsize=8.5, color="tab:orange")
    ax.set_ylabel("Score")
    ax.set_title(f"各模型最佳配置综合指标对比 — split {split} ({SPLIT_DESC[split]})")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.text(0.5, 0.01,
             "注: Precision / Recall 需重新跑模型保存 pred 后补充; 此处用 test_acc / test_f1 代替",
             ha="center", fontsize=8, color="gray")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    savefig(fig, f"09_radar_metrics_split_{split}.png")


# ─────────────────────────────────────────────
# 汇总报告: 超参数搜索表
# ─────────────────────────────────────────────

def dump_hp_summary(split):
    """把所有 baseline 结果打印成汇总表, 保存为 CSV"""
    rows = []
    for model in ["LR", "MLP", "SVM", "NB", "DT"]:
        df = load_baseline(model, split, "raw")
        if df.empty:
            continue
        best = best_config(df, "val_acc")
        rows.append({
            "split": split, "model": model,
            "best_hp": str(dict(best.drop(labels=["split", "version"]).dropna())),
            "val_acc": round(best["val_acc"], 4),
            "test_acc": round(best["test_acc"], 4),
            "val_f1": round(best.get("val_f1", np.nan), 4),
            "test_f1": round(best.get("test_f1", np.nan), 4),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("开始绘制评价图表 ...")
    print(f"输出目录: {EVAL_DIR.relative_to(ROOT)}")
    print("=" * 60)

    all_summary = []

    for split in SPLITS:
        print(f"\n[split {split}]")
        plot_lr_loss(split)      # 1. LR loss
        plot_mlp_loss(split)     # 2. MLP loss
        plot_lr_hp(split)        # 3. LR 超参
        plot_mlp_hp(split)      # 4. MLP 超参
        plot_svm_hp(split)      # 5. SVM 超参
        plot_nb_hp(split)        # 6. NB 超参
        plot_dt_hp(split)       # 7. DT 超参
        plot_cm_placeholder(split)  # 8. 模型对比
        plot_radar(split)        # 9. 综合指标

        summary = dump_hp_summary(split)
        all_summary.append(summary)
        print(f"\n  split {split} 最佳配置汇总:")
        print(summary[["model", "val_acc", "test_acc", "val_f1", "test_f1"]].to_string(index=False))

    # 保存汇总表
    summary_df = pd.concat(all_summary, ignore_index=True)
    summary_path = EVAL_DIR / "best_config_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n汇总表已保存: {summary_path.relative_to(ROOT)}")

    # 打印超参数说明
    print("\n" + "=" * 60)
    print("各模型超参数说明")
    print("=" * 60)
    for model, info in HP_INFO.items():
        print(f"\n【{model}】{info}")


if __name__ == "__main__":
    main()
