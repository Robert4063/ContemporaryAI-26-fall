"""Plot stored split_B MLP losses by hidden architecture, holding text fixed.

No fitting or prediction. The two panels hold alpha fixed separately so that
three hidden architectures can be compared without changing regularization.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "results" / "evaluation"
VERSION = "subject_body_author"
HIDDEN = ["50", "100", "100_50"]
ALPHAS = [0.0001, 0.001]
LABELS = {"50": "(50,)", "100": "(100,)", "100_50": "(100, 50)"}
COLORS = {"50": "#1675E8", "100": "#FF8500", "100_50": "#D92780"}
STYLES = {"50": "-", "100": "--", "100_50": "-."}
MARKERS = {"50": "o", "100": "s", "100_50": "^"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_curves():
    records = []
    for alpha in ALPHAS:
        for hidden in HIDDEN:
            config_id = f"mlp_split_B_{VERSION}_h{hidden}_a{alpha}"
            loss_path = ROOT / "results" / "loss" / f"{config_id}.npy"
            metric_path = ROOT / "results" / "baseline" / f"{config_id}_metrics.csv"
            loss = np.load(loss_path, allow_pickle=False)
            if loss.shape != (300,) or not np.isfinite(loss).all() or (loss <= 0).any():
                raise ValueError(f"Invalid stored loss curve: {config_id}")
            metrics = pd.read_csv(metric_path, float_precision="round_trip").set_index("eval_set")
            if set(metrics.index) != {"train", "val"} or len(metrics) != 2:
                raise ValueError(f"Invalid train/validation metrics: {config_id}")
            records.append({"config_id": config_id, "alpha": alpha, "hidden": hidden,
                            "loss": loss, "train_accuracy": float(metrics.loc["train", "accuracy"]),
                            "val_accuracy": float(metrics.loc["val", "accuracy"]),
                            "val_macro_f1": float(metrics.loc["val", "macro_f1"]),
                            "loss_path": loss_path, "metrics_path": metric_path})
    return records


def main():
    records = load_curves()
    summary = []
    full = {"iteration": np.arange(1, 301)}
    for row in records:
        full[f"alpha={row['alpha']};h={LABELS[row['hidden']]}"] = row["loss"]
        summary.append({"config_id": row["config_id"], "split": "split_B", "version": VERSION,
                        "alpha": row["alpha"], "hidden_layer_sizes": LABELS[row["hidden"]],
                        **{f"loss_iter_{i}": float(row["loss"][i-1]) for i in [1, 50, 100, 200, 300]},
                        "minimum_loss": float(row["loss"].min()),
                        "minimum_loss_iteration": int(row["loss"].argmin()+1),
                        "train_accuracy": row["train_accuracy"], "val_accuracy": row["val_accuracy"],
                        "val_macro_f1": row["val_macro_f1"]})
    data_path = EVAL / "mlp_split_B_hidden_size_loss_data.csv"
    summary_path = EVAL / "mlp_split_B_hidden_size_summary.csv"
    pd.DataFrame(full).to_csv(data_path, index=False, float_format="%.17g")
    pd.DataFrame(summary).to_csv(summary_path, index=False, float_format="%.17g")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.unicode_minus": False})
    fig = plt.figure(figsize=(14.5, 8.4))
    axes = [fig.add_axes([.07, .42, .405, .39]), fig.add_axes([.555, .42, .405, .39])]
    fig.suptitle("MLP hidden architectures: loss comparison on split_B", x=.07, y=.98,
                 ha="left", fontsize=20, fontweight="bold")
    fig.text(.07, .915, "Fixed text: subject_body_author (Author + Subject + Body) | train = 6,262 | validation = 369", fontsize=11)
    fig.text(.07, .879, "ReLU + Adam | learning rate = 0.001 | random state = 42 | 300 epochs | no early stopping or smoothing", fontsize=10, color="#555555")
    for ax, alpha in zip(axes, ALPHAS):
        for row in [r for r in records if r["alpha"] == alpha]:
            hidden = row["hidden"]
            ax.plot(np.arange(1,301), row["loss"], color=COLORS[hidden], linestyle=STYLES[hidden],
                    linewidth=2.1, marker=MARKERS[hidden], markevery=[0,49,99,199,299],
                    markersize=4.5, label=f"h = {LABELS[hidden]}")
        ax.set_title(f"Fixed alpha = {alpha:g}", fontsize=13, fontweight="bold", pad=13)
        ax.set_yscale("log")
        ax.set_ylim(0.0002, 3.5)
        ax.set_xlim(1,300)
        ax.set_xticks([1,50,100,150,200,250,300])
        ax.set_xlabel("Epoch / stored iteration (1-300)")
        ax.set_ylabel("Training loss: cross-entropy + L2 (log scale)")
        ax.grid(alpha=.25, which="major", linewidth=.7)
        ax.grid(alpha=.10, which="minor", linewidth=.5)
        ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax_table = fig.add_axes([.07, .095, .89, .22])
    ax_table.axis("off")
    columns = ["alpha", "hidden h", "Loss @1", "Loss @50", "Loss @100", "Loss @200", "Loss @300", "Train Acc", "Val Acc"]
    rows = [[f"{row['alpha']:g}", row["hidden_layer_sizes"],
             *[f"{row[f'loss_iter_{i}']:.6f}" for i in [1,50,100,200,300]],
             f"{row['train_accuracy']:.2%}", f"{row['val_accuracy']:.2%}"] for row in summary]
    table = ax_table.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center", colLoc="center",
                           colWidths=[.085,.105,.115,.115,.115,.115,.115,.115,.12])
    table.auto_set_font_size(False)
    table.set_fontsize(9.4)
    table.scale(1,1.55)
    for (i,j), cell in table.get_celld().items():
        cell.set_edgecolor("#D9E0E7")
        if i==0:
            cell.set_facecolor("#E9F0F8")
            cell.set_text_props(fontweight="bold", color="#20344A")
        else:
            cell.set_facecolor("#FFFFFF" if i%2 else "#F7FAFD")
            if j==1:
                cell.set_text_props(color=COLORS[records[i-1]["hidden"]], fontweight="bold")
    fig.text(.07,.335,"Exact checkpoints from stored curves (table rounded to 6 decimals; full precision in CSV)",fontsize=11,fontweight="bold")
    fig.text(.07,.045,"Source: MLPClassifier.loss_curve_ in results/loss/*.npy. Epoch loss is the weighted mean of mini-batch objectives.",fontsize=9,color="#555555")
    fig.text(.07,.018,"h denotes a hidden-layer tuple, not a loss-function parameter. Compare h within each alpha panel; validation accuracy is a final metric.",fontsize=9,color="#555555")
    image_path = EVAL / "figures" / "mlp_split_B_hidden_size_comparison.png"
    fig.savefig(image_path,dpi=200,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    manifest = {"split":"split_B", "version":VERSION, "alpha_values":ALPHAS,
                "hidden_layer_sizes":[LABELS[h] for h in HIDDEN], "epochs":300,
                "train_rows":6262,"validation_rows":369,"source":"MLPClassifier.loss_curve_",
                "definition":"Per-epoch sample-weighted average of mini-batch cross-entropy plus L2 training objectives; not validation loss.",
                "not_a_fixed_weights_full_dataset_objective":True,"smoothing":False,"refitted":False,
                "data_file":data_path.relative_to(ROOT).as_posix(),"data_sha256":sha256(data_path),
                "summary_file":summary_path.relative_to(ROOT).as_posix(),"summary_sha256":sha256(summary_path),
                "image_file":image_path.relative_to(ROOT).as_posix(),"image_sha256":sha256(image_path),
                "sources":[{"config_id":r["config_id"],"loss_file":r["loss_path"].relative_to(ROOT).as_posix(),
                            "loss_sha256":sha256(r["loss_path"]),"metrics_file":r["metrics_path"].relative_to(ROOT).as_posix(),
                            "metrics_sha256":sha256(r["metrics_path"])} for r in records]}
    (EVAL/"mlp_split_B_hidden_size_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"Saved: {image_path}\nSaved: {data_path}\nSaved: {summary_path}")


if __name__=="__main__":
    main()
