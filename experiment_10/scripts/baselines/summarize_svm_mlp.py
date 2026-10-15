"""Summarize strict train/validation/test results for SVM and MLP."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "results" / "baseline"
FIGURES = ROOT / "results" / "evaluation" / "figures"
MODEL_DIR = ROOT / "models" / "svm_mlp"
FIGURES.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SPLITS = ["A", "B", "C"]
VERSIONS = ["raw", "body_only", "subject_body", "subject_body_author"]
MODELS = ["SVM", "MLP"]


def main():
    rows = []
    for model in MODELS:
        for split in SPLITS:
            for version in VERSIONS:
                path = BASELINE / f"{model}_{split}_{version}_final.csv"
                if not path.exists():
                    print(f"missing: {path.name}")
                    continue
                row = pd.read_csv(path).iloc[0].to_dict()
                row["model"] = model
                rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        raise SystemExit("No final result files found")
    result = result[
        ["model", "split", "version", "val_acc", "test_acc", "val_f1", "test_f1"]
        + [c for c in result.columns if c not in {"model", "split", "version", "val_acc", "test_acc", "val_f1", "test_f1"}]
    ]
    result = result.rename(columns={
        "val_acc": "val_accuracy_score",
        "test_acc": "test_accuracy_score",
    })
    result.to_csv(BASELINE / "SVM_MLP_final_summary.csv", index=False)

    best = (result.sort_values(["model", "val_accuracy_score", "val_f1"], ascending=[True, False, False])
            .groupby("model", as_index=False).first())
    best.to_csv(MODEL_DIR / "best_config.csv", index=False)
    print("Best configurations selected by validation accuracy:")
    print(best[["model", "split", "version", "val_accuracy_score", "test_accuracy_score"]].to_string(index=False))

    labels = [f"{m} / split {s}" for m in MODELS for s in SPLITS]
    for metric, title, filename in [
        ("val_accuracy_score", "Validation accuracy", "svm_mlp_val_accuracy_heatmap.png"),
        ("test_accuracy_score", "Test accuracy", "svm_mlp_test_accuracy_heatmap.png"),
    ]:
        matrix = result.pivot_table(index=["model", "split"], columns="version", values=metric)
        matrix = matrix.reindex(pd.MultiIndex.from_product([MODELS, SPLITS], names=["model", "split"]))
        matrix.index = labels
        matrix = matrix.reindex(columns=VERSIONS)
        plt.figure(figsize=(10, 6))
        sns.heatmap(matrix, annot=True, fmt=".4f", cmap="YlGnBu", vmin=0, vmax=1,
                    linewidths=0.5, cbar_kws={"label": metric})
        plt.title(f"SVM and MLP {title} by split and text version")
        plt.xlabel("Text version")
        plt.ylabel("Model / split")
        plt.tight_layout()
        plt.savefig(FIGURES / filename, dpi=180)
        plt.close()
        print(f"saved {filename}")

    best_plot = best.set_index("model")[["val_accuracy_score", "test_accuracy_score"]]
    ax = best_plot.plot(kind="bar", figsize=(7, 5), color=["#4C72B0", "#DD8452"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Best SVM and MLP: validation vs test accuracy")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "svm_mlp_best_val_test_comparison.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
