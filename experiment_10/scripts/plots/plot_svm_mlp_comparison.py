"""
SVM vs MLP 单独比较图
======================
参考 chart3_model_evaluation_comparison.png 风格:
  上: 分组柱状图 — SVM / MLP 在 split A/B/C 上的 Val/Test Accuracy
  下: 热力图    — 不同数据版本下的 Best Val Accuracy

输出: results/evaluation/figures/chart_svm_mlp_comparison.png
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─── 路径 ───────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "results" / "evaluation" / "figures"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

SPLITS      = ["A", "B", "C"]
SPLIT_DESC  = {"A": "80:10:10", "B": "85:5:10", "C": "90:5:5"}
VERSIONS    = ["raw", "body_only", "subject_body", "subject_body_author"]
VERSION_LABELS = {
    "raw":                  "raw ",
    "body_only":            "body_only",
    "subject_body":         "subject_body",
    "subject_body_author":  "subject_body_author",
}

# ─── 颜色 ───────────────────────────────────────────────────
C_SVM  = "#0072B2"   # 蓝
C_MLP  = "#D55E00"   # 朱红 / 橙红
C_VAL  = "#009E73"   # 蓝绿 / 绿色
C_TEST = "#CC79A7"   # 紫红

# ─── 加载数据 ───────────────────────────────────────────────
def load_all_versions(model):
    """返回 (split, version) → best val_acc 的 Series"""
    path = ROOT / "results" / "baseline" / f"{model}_all_splits_all_versions.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)

    # 对每个 (split, version) 取 val_acc 最高的那行
    idx = df.groupby(["split", "version"])["val_acc"].idxmax()
    return df.loc[idx].set_index(["split", "version"])["val_acc"]


def best_overall(model, split):
    """返回 (val_acc, test_acc) 两个浮点数, 所有超参中验证集最优"""
    path = ROOT / "results" / "baseline" / f"{model}_all_splits_all_versions.csv"
    if not path.exists():
        return np.nan, np.nan
    df = pd.read_csv(path)
    sub = df[df["split"] == split]
    if sub.empty:
        return np.nan, np.nan
    best_idx = sub["val_acc"].idxmax()
    best = sub.loc[best_idx]
    return float(best["val_acc"]), float(best["test_acc"])


# ─── 数据准备 ───────────────────────────────────────────────
svm_all = load_all_versions("SVM")
mlp_all = load_all_versions("MLP")

# 分组柱状图: 每 split 的 val / test
bar_data = {}
for sp in SPLITS:
    svm_val, svm_test = best_overall("SVM", sp)
    mlp_val, mlp_test = best_overall("MLP", sp)
    bar_data[sp] = {
        "SVM_val": svm_val,  "SVM_test": svm_test,
        "MLP_val": mlp_val,  "MLP_test": mlp_test,
    }

print("=== 分组柱状图数据 ===")
for sp, d in bar_data.items():
    print(f"  Split {sp} ({SPLIT_DESC[sp]}): "
          f"SVM val={d['SVM_val']:.4f} test={d['SVM_test']:.4f}  |  "
          f"MLP val={d['MLP_val']:.4f} test={d['MLP_test']:.4f}")

print("\n=== 热力图数据 ===")
print(f"{'split':<6} {'version':<25} {'SVM_val_acc':>12} {'MLP_val_acc':>12}")
for sp in SPLITS:
    for ver in VERSIONS:
        sv = svm_all.get((sp, ver), np.nan)
        mv = mlp_all.get((sp, ver), np.nan)
        print(f"{sp:<6} {VERSION_LABELS[ver]:<25} {sv:>12.4f} {mv:>12.4f}")


# ─── 绘图 ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 11),
                         gridspec_kw={"height_ratios": [1, 1.3]})

# ── 上半: 分组柱状图 ──────────────────────────────────────
ax1 = axes[0]
n = len(SPLITS)
x = np.arange(n)
w = 0.19          # 每组 4 根柱子, 每根宽度
gap = 0.05

svm_val_list   = [bar_data[sp]["SVM_val"]   for sp in SPLITS]
svm_test_list  = [bar_data[sp]["SVM_test"]  for sp in SPLITS]
mlp_val_list   = [bar_data[sp]["MLP_val"]   for sp in SPLITS]
mlp_test_list  = [bar_data[sp]["MLP_test"]  for sp in SPLITS]

offsets = [-1.5*w - gap/2, -0.5*w - gap/2, +0.5*w + gap/2, +1.5*w + gap/2]
colors  = [C_SVM, C_SVM+"99", C_MLP, C_MLP+"99"]
labels  = ["SVM Val", "SVM Test", "MLP Val", "MLP Test"]
vals    = [svm_val_list, svm_test_list, mlp_val_list, mlp_test_list]

for off, col, lbl, vlist in zip(offsets, colors, labels, vals):
    bars = ax1.bar(x + off, vlist, w, label=lbl, color=col, edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, vlist):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color="dimgray")

ax1.set_xticks(x)
ax1.set_xticklabels([f"Split {sp}\n({SPLIT_DESC[sp]})" for sp in SPLITS], fontsize=11)
ax1.set_ylim(0.88, 1.01)
ax1.set_ylabel("Accuracy", fontsize=11)
ax1.set_title("SVM vs MLP — Val / Test Accuracy", fontsize=13, pad=10)
ax1.legend(loc="upper right", ncol=4, fontsize=10)
ax1.grid(axis="y", alpha=0.3)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ── 下半: 热力图 ───────────────────────────────────────────
ax2 = axes[1]

# 构造矩阵: 行=(SVM/MLP)×split, 列=version
row_labels = [f"{m} — Split {sp}" for m in ["SVM", "MLP"] for sp in SPLITS]
matrix = []
for model in ["SVM", "MLP"]:
    for sp in SPLITS:
        row = []
        for ver in VERSIONS:
            v = (svm_all if model == "SVM" else mlp_all).get((sp, ver), np.nan)
            row.append(v)
        matrix.append(row)

matrix = np.array(matrix)
row_labels_cat = [f"{m} Split {sp}" for m in ["SVM", "MLP"] for sp in SPLITS]

im = ax2.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.88, vmax=1.0)

# 标注数值
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = matrix[i, j]
        if not np.isnan(val):
            ax2.text(j, i, f"{val:.3f}", ha="center", va="center",
                     fontsize=10, color="black",
                     fontweight="bold" if val == matrix[i].max() else "normal")

ax2.set_xticks(np.arange(len(VERSIONS)))
ax2.set_xticklabels([VERSION_LABELS[v] for v in VERSIONS], fontsize=9.5, rotation=20, ha="right")
ax2.set_yticks(np.arange(len(row_labels_cat)))
ax2.set_yticklabels(row_labels_cat, fontsize=10)
ax2.set_title("Best Val Accuracy by Data Version (all hyper-parameters)", fontsize=13, pad=10)


# colorbar
cbar = plt.colorbar(im, ax=ax2, shrink=0.6, pad=0.02)
cbar.set_label("Val Accuracy", fontsize=10)

fig.tight_layout(rect=[0, 0, 1, 0.97])
out = EVAL_DIR / "chart_svm_mlp_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n已保存: {out.relative_to(ROOT)}")
