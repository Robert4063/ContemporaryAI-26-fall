"""
plot_mlp_versions.py
====================
固定 MLP 结构 hidden_layer_sizes=(100, 50), 比较四种数据清洗版本的训练 loss。
每个划分 (A/B/C) 一张图, 保存到 results/plots/。

- 颜色 = 数据版本 (raw / body_only / subject_body / subject_body_author)
- 线型 = alpha (实线 0.0001, 虚线 0.001)

用法:
  python scripts/plots/plot_mlp_versions.py
  python scripts/plots/plot_mlp_versions.py --hidden "(100,)"
  python scripts/plots/plot_mlp_versions.py --alpha 0.0001   # 固定 alpha, 只比较数据版本
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
VERSIONS = ["raw", "body_only", "subject_body", "subject_body_author"]
VERSION_COLORS = {
    "raw": "tab:blue", "body_only": "tab:red",
    "subject_body": "tab:green", "subject_body_author": "tab:purple",
}
ALPHA_STYLES = {0.0001: "-", 0.001: "--"}


def plot_split(split, hidden, alpha=None):
    """alpha=None 时画该结构下所有 alpha (线型区分); 指定 alpha 时只画这一组。"""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for ver in VERSIONS:
        path = LOSS_DIR / f"MLP_{split}_{ver}.csv"
        if not path.exists():
            print(f"  缺少 {path.name}")
            continue
        df = pd.read_csv(path)
        # CSV 中 hidden_layer_sizes 以字符串保存, 去空格后比较
        df = df[df["hidden_layer_sizes"].str.replace(" ", "") == hidden.replace(" ", "")]
        if alpha is not None:
            df = df[(df["alpha"] - alpha).abs() < 1e-12]
            if df.empty:
                print(f"  {path.name} 中没有 hidden={hidden}, alpha={alpha:g}")
                continue
        for a, sub in df.groupby("alpha"):
            sub = sub.sort_values("epoch")
            ax.plot(sub["epoch"], sub["loss"], color=VERSION_COLORS[ver],
                    linestyle="-" if alpha is not None else ALPHA_STYLES.get(round(a, 6), ":"),
                    linewidth=1.6,
                    label=ver if alpha is not None else f"{ver} | alpha={a:g}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_yscale("log")
    alpha_txt = f", alpha={alpha:g}" if alpha is not None else ""
    ax.set_title(f"MLP hidden={hidden}{alpha_txt} training loss by data version "
                 f"- split {split} ({SPLIT_DESC[split]})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    tag = hidden.strip("()").replace(",", "_").replace(" ", "").strip("_")
    a_tag = f"_a{alpha:g}" if alpha is not None else ""
    out = OUT_DIR / f"MLP_h{tag}{a_tag}_versions_split_{split}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  已保存 {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", default="(100, 50)")
    ap.add_argument("--alpha", type=float, default=None,
                    help="只画指定 alpha (如 0.0001); 不给则画全部 alpha")
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    args = ap.parse_args()
    for split in args.splits:
        plot_split(split, args.hidden, args.alpha)


if __name__ == "__main__":
    main()
