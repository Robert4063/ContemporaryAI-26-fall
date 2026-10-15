"""
plot_loss.py
============
为每个 (模型, 划分) 画一张图, 保存到 results/plots/。

- LR / MLP: 真实的逐轮训练 loss 曲线
    颜色 = 数据版本 (raw / body_only ...), 线型 = 超参数配置
- NB / SVM / DT: 没有逐轮 loss (一次性求解), 画训练/验证准确率随超参数配置的变化
    颜色 = 数据版本, 实线 = train_acc, 虚线 = val_acc (标题中注明不是 loss 曲线)

用法:
  python scripts/plots/plot_loss.py
  python scripts/plots/plot_loss.py --versions raw body_only subject_body
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT     = Path(__file__).resolve().parents[2]
LOSS_DIR = ROOT / "results" / "loss"
OUT_DIR  = ROOT / "results" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["A", "B", "C"]
SPLIT_DESC = {"A": "80:10:10", "B": "85:5:10", "C": "90:5:5"}
VERSION_COLORS = {
    "raw": "tab:blue", "body_only": "tab:red",
    "subject_body": "tab:green", "subject_body_author": "tab:purple",
}
LINESTYLES = ["-", "--", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1))]

# 模型 -> (超参数列, 是否有逐轮 loss)
ITERATIVE = {
    "LR":  ["C"],
    "MLP": ["hidden_layer_sizes", "alpha"],
    "SVM": ["C"],
}
STATIC = {
    "NB":  ["alpha"],
    "DT":  ["max_depth", "min_samples_leaf"],
}


def cfg_label(row, cols):
    return ", ".join(f"{c}={row[c]}" for c in cols)


def load_loss(model, split, versions):
    """读取每个 (model, split, version) 各自的 CSV, 不依赖可能被并行进程覆盖的汇总文件。"""
    frames = []
    for ver in versions:
        p = LOSS_DIR / f"{model}_{split}_{ver}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
        else:
            print(f"  缺少 {p.name}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_iterative(model, split, versions):
    df = load_loss(model, split, versions)
    if df.empty:
        print(f"  跳过 {model} split {split}: 没有数据")
        return
    cols = ITERATIVE[model]
    df = df.assign(cfg=df.apply(lambda r: cfg_label(r, cols), axis=1))
    cfgs = list(dict.fromkeys(df["cfg"]))

    # SVM 使用 "step" 列，MLP 使用 "epoch" 列
    x_col = "step" if model == "SVM" else "epoch"

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for ver in versions:
        for i, cfg in enumerate(cfgs):
            sub = df[(df["version"] == ver) & (df["cfg"] == cfg)].sort_values(x_col)
            if sub.empty:
                continue
            # SVM loss 列名是 train_loss，MLP 是 loss
            loss_col = "train_loss" if model == "SVM" else "loss"
            if loss_col not in sub.columns:
                continue
            ax.plot(sub[x_col], sub[loss_col], color=VERSION_COLORS.get(ver, "k"),
                    linestyle=LINESTYLES[i % len(LINESTYLES)], linewidth=1.6,
                    label=f"{ver} | {cfg}")
    ax.set_xlabel("iteration / epoch")
    ax.set_ylabel("training loss")
    ax.set_yscale("log")
    ax.set_title(f"{model} training loss - split {split} ({SPLIT_DESC[split]})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = OUT_DIR / f"{model}_loss_split_{split}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  已保存 {out.relative_to(ROOT)}")


def plot_static(model, split, versions):
    df = load_loss(model, split, versions)
    if df.empty:
        print(f"  跳过 {model} split {split}: 没有数据")
        return
    cols = STATIC[model]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for ver in versions:
        sub = df[df["version"] == ver].sort_values("step")
        if sub.empty:
            continue
        x = range(len(sub))
        color = VERSION_COLORS.get(ver, "k")
        ax.plot(x, sub["train_acc"], color=color, linestyle="-", marker="o",
                label=f"{ver} train_acc")
        ax.plot(x, sub["val_acc"], color=color, linestyle="--", marker="s",
                label=f"{ver} val_acc")
        ticks = [cfg_label(r, cols) for _, r in sub.iterrows()]
    ax.set_xticks(list(range(len(ticks))))
    ax.set_xticklabels(ticks, rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("hyperparameter config")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{model} (no per-iteration loss; accuracy vs hyperparameter) "
                 f"- split {split} ({SPLIT_DESC[split]})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = OUT_DIR / f"{model}_acc_split_{split}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  已保存 {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", default=["raw", "body_only"])
    args = ap.parse_args()

    for split in SPLITS:
        print(f"[split {split}]")
        for model in ITERATIVE:
            plot_iterative(model, split, args.versions)
        for model in STATIC:
            plot_static(model, split, args.versions)


if __name__ == "__main__":
    main()
