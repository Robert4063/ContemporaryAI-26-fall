"""Generate the four comparison figures requested for the baseline experiment.

All plotting scripts live in ``scripts/plots``.  The script reads the numeric
records kept under ``results/loss`` and ``results/baseline`` and writes only
the four figures to ``results/evaluation/figures``.
"""

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LOSS_DIR = ROOT / "results" / "loss"
BASELINE_DIR = ROOT / "results" / "baseline"
FIG_DIR = ROOT / "results" / "evaluation" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["A", "B", "C"]
SPLIT_DESC = {"A": "80:10:10", "B": "85:5:10", "C": "90:5:5"}
VERSIONS = ["raw", "body_only", "subject_body", "subject_body_author"]
VERSION_LABELS = {
    "raw": "raw",
    "body_only": "body_only",
    "subject_body": "subject_body",
    "subject_body_author": "subject_body_author",
}
MODELS = ["LR", "MLP", "SVM", "NB", "DT"]
COLORS = {
    "A": "#4C72B0", "B": "#DD8452", "C": "#55A868",
    "raw": "#4C72B0", "body_only": "#DD8452",
    "subject_body": "#55A868", "subject_body_author": "#C44E52",
    "LR": "#4C72B0", "MLP": "#DD8452", "SVM": "#55A868",
    "NB": "#C44E52", "DT": "#8172B2",
}

# Microsoft YaHei is available on the project Windows host and prevents the
# Chinese labels from rendering as empty square glyphs.
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def read_csv(folder: Path, name: str) -> pd.DataFrame:
    path = folder / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def normalize_hidden(value: object) -> str:
    return str(value).replace(" ", "")


def mlp_curve(split: str, version: str) -> pd.DataFrame:
    df = read_csv(LOSS_DIR, f"MLP_{split}_{version}.csv")
    if df.empty:
        return df
    mask = (df["hidden_layer_sizes"].map(normalize_hidden) == "(100,50)")
    mask &= (df["alpha"].astype(float) - 0.001).abs() < 1e-10
    return df.loc[mask].sort_values("epoch")


def save(fig: plt.Figure, filename: str) -> None:
    path = FIG_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path.relative_to(ROOT)}")


def style_axis(ax: plt.Axes) -> None:
    ax.grid(alpha=0.28, axis="y", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def chart1() -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = 0
    for split in SPLITS:
        curve = mlp_curve(split, "raw")
        if curve.empty:
            continue
        ax.plot(curve["epoch"], curve["loss"], color=COLORS[split], linewidth=1.8,
                label=f"split {split} ({SPLIT_DESC[split]})")
        plotted += 1
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (log scale)")
    ax.set_title("MLP loss: different dataset splits\n(hidden=(100, 50), alpha=0.001, raw)")
    ax.legend(loc="upper right")
    style_axis(ax)
    if not plotted:
        ax.text(0.5, 0.5, "No matching records", ha="center", va="center", transform=ax.transAxes)
    save(fig, "chart1_mlp_loss_split_comparison.png")


def chart2() -> None:
    # Keep the split fixed so that only preprocessing changes between curves.
    split = "A"
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = 0
    for version in VERSIONS:
        curve = mlp_curve(split, version)
        if curve.empty:
            continue
        ax.plot(curve["epoch"], curve["loss"], color=COLORS[version], linewidth=1.8,
                label=VERSION_LABELS[version])
        plotted += 1
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (log scale)")
    ax.set_title(f"MLP loss: different preprocessing methods\nsplit {split} ({SPLIT_DESC[split]}), hidden=(100, 50), alpha=0.001")
    ax.legend(loc="upper right")
    style_axis(ax)
    if not plotted:
        ax.text(0.5, 0.5, "No matching records", ha="center", va="center", transform=ax.transAxes)
    save(fig, "chart2_mlp_loss_preprocessing_comparison.png")


def best_row(df: pd.DataFrame) -> pd.Series | None:
    if df.empty or "val_acc" not in df:
        return None
    valid = df.dropna(subset=["val_acc"])
    if valid.empty:
        return None
    max_val = valid["val_acc"].max()
    tied = valid[valid["val_acc"] == max_val]
    return tied.sort_values("test_acc", ascending=False).iloc[0]


def chart3() -> None:
    metrics = ["val_acc", "test_acc", "val_f1", "test_f1"]
    metric_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for index, split in enumerate(SPLITS):
        ax = axes[index]
        rows = []
        for model in MODELS:
            row = best_row(read_csv(BASELINE_DIR, f"{model}_{split}_raw.csv"))
            if row is not None:
                rows.append({"model": model, **{m: float(row.get(m, np.nan)) for m in metrics}})
        if not rows:
            ax.text(0.5, 0.5, "No matching records", ha="center", va="center", transform=ax.transAxes)
            continue
        result = pd.DataFrame(rows)
        x = np.arange(len(result))
        width = 0.2
        for offset, metric in enumerate(metrics):
            values = result[metric].to_numpy(dtype=float)
            ax.bar(x + (offset - 1.5) * width, values, width, color=metric_colors[offset], label=metric)
            for x_pos, value in zip(x + (offset - 1.5) * width, values):
                ax.text(x_pos, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(result["model"])
        ax.set_ylim(0, 1.08)
        ax.set_title(f"split {split} ({SPLIT_DESC[split]})")
        if index == 0:
            ax.set_ylabel("Score")
        ax.legend(fontsize=8, ncol=2, loc="lower right")
        style_axis(ax)
    fig.suptitle("Model evaluation comparison (best validation configuration)", y=1.02)
    fig.tight_layout()
    save(fig, "chart3_model_evaluation_comparison.png")


def chart4() -> None:
    # LR and MLP contain true epoch-level loss. Static estimators only retain
    # one row per hyperparameter setting, so their panel uses 1 - train_acc.
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    mlp = mlp_curve("A", "raw")
    if not mlp.empty:
        ax.plot(mlp["epoch"], mlp["loss"], color=COLORS["MLP"], linewidth=1.8,
                label="MLP (100, 50), alpha=0.001")
    lr = read_csv(LOSS_DIR, "LR_A_raw.csv")
    if not lr.empty:
        lr = lr[lr["C"].astype(float) == 1.0].sort_values("epoch")
        if not lr.empty:
            ax.plot(lr["epoch"], lr["loss"], color=COLORS["LR"], linewidth=1.8,
                    label="LR (C=1.0)")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (log scale)")
    ax.set_title("True epoch-level loss (split A, raw)")
    ax.legend()
    style_axis(ax)

    ax = axes[1]
    for model in ["SVM", "NB", "DT"]:
        df = read_csv(LOSS_DIR, f"{model}_A_raw.csv")
        if df.empty or "train_acc" not in df:
            continue
        df = df.sort_values("step")
        ax.plot(df["step"], 1.0 - df["train_acc"], color=COLORS[model], marker="o",
                linewidth=1.8, label=model)
    ax.set_xlabel("Hyperparameter setting (step)")
    ax.set_ylabel("1 - train_acc (proxy loss)")
    ax.set_title("Static estimators: retained loss proxy (split A, raw)")
    ax.legend()
    style_axis(ax)
    fig.suptitle("Training loss comparison across models", y=1.02)
    fig.tight_layout()
    save(fig, "chart4_model_loss_comparison.png")


def main() -> None:
    # Remove only the four deliverables managed by this script.
    for filename in [
        "chart1_mlp_loss_split_comparison.png",
        "chart2_mlp_loss_preprocessing_comparison.png",
        "chart3_model_evaluation_comparison.png",
        "chart4_model_loss_comparison.png",
    ]:
        path = FIG_DIR / filename
        if path.exists():
            path.unlink()
    chart1()
    chart2()
    chart3()
    chart4()


if __name__ == "__main__":
    main()
