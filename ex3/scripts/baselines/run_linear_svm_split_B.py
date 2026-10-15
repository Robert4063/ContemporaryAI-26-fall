"""Train 12 LinearSVC configurations on split_B and compare existing baselines.

Run with the experiment_1 Python environment. Existing SGD SVM / MLP models
are loaded, never refitted. Test data is opened only after validation selection
has been frozen on disk. Completed training and final evaluation are reused.
"""

import hashlib
import json
import logging
import pickle
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MultipleLocator
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "linear_svm_split_B"
VERSIONS = ["raw", "body_only", "subject_body", "subject_body_author"]
MODELS = ["svm", "mlp", "linear_svm"]
MODEL_LABELS = {"svm": "SGD SVM", "mlp": "MLP", "linear_svm": "LinearSVC"}
C_VALUES = [0.1, 1.0, 10.0]
SELECTION_RULE = "validation accuracy desc, macro_f1 desc, config_id asc"
LOG = logging.getLogger("linear_svm_split_B")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_pickle(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def scores(y, prediction):
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "precision": float(precision_score(y, prediction, average="macro", zero_division=0)),
        "recall": float(recall_score(y, prediction, average="macro", zero_division=0)),
    }


def train_grid():
    records = []
    for version in VERSIONS:
        data_dir = ROOT / "data-processed" / "split_B" / version
        input_hashes = {subset: digest(data_dir / f"{subset}.csv") for subset in ["train", "val"]}
        train = pd.read_csv(data_dir / "train.csv", keep_default_na=False)
        val = pd.read_csv(data_dir / "val.csv", keep_default_na=False)
        if len(train) != 6262 or len(val) != 369:
            raise ValueError(f"Unexpected split_B sizes for {version}")
        vectorizer = TfidfVectorizer(min_df=2, max_df=0.95, max_features=20000)
        x_train = vectorizer.fit_transform(train["text"])
        x_val = vectorizer.transform(val["text"])
        # Assert the feature representation matches the original experiment.
        old_vectorizer = load_pickle(ROOT / "models" / f"svm_split_B_{version}_C0.1" / "vectorizer.pkl")
        if not (np.array_equal(vectorizer.get_feature_names_out(), old_vectorizer.get_feature_names_out())
                and np.array_equal(vectorizer.idf_, old_vectorizer.idf_)):
            raise ValueError(f"TF-IDF mismatch with original experiment: {version}")
        LOG.info("%s: train=%d val=%d features=%d; TF-IDF matches baseline", version, len(train), len(val), x_train.shape[1])

        for c in C_VALUES:
            config_id = f"linear_svm_split_B_{version}_C{c}"
            model_dir = ROOT / "models" / config_id
            metric_path = OUT / "metrics" / f"{config_id}_metrics.csv"
            params = {"C": c, "penalty": "l2", "loss": "squared_hinge", "dual": "auto",
                      "multi_class": "ovr", "tol": 1e-4, "max_iter": 10000, "random_state": 42}
            specification = {"model": "linear_svm", "estimator": "sklearn.svm.LinearSVC",
                             "split": "split_B", "version": version, "params": params,
                             "input_sha256": input_hashes, "sklearn_version": sklearn.__version__}
            if model_dir.exists() or metric_path.exists():
                config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
                if config["specification"] != specification:
                    raise ValueError(f"Cached training specification changed: {config_id}")
                for name, expected in config["artifact_sha256"].items():
                    path = metric_path if name == "metrics.csv" else model_dir / name
                    if digest(path) != expected:
                        raise ValueError(f"Cached artifact changed: {config_id}/{name}")
                records.extend(pd.read_csv(metric_path, float_precision="round_trip").to_dict("records"))
                LOG.info("Reused completed model: %s", config_id)
                continue

            classifier = LinearSVC(**params)
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                classifier.fit(x_train, train["target"])
            elapsed = time.perf_counter() - started
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                raise RuntimeError(f"LinearSVC did not converge: {config_id}; inspect log before retrying")
            for warning in caught:
                LOG.warning("%s: %s", config_id, warning.message)
            rows = []
            for subset, x, frame in [("train", x_train, train), ("val", x_val, val)]:
                rows.append({"config_id": config_id, "model": "linear_svm", "split": "split_B",
                             "version": version, "C": c, "eval_set": subset,
                             **scores(frame["target"], classifier.predict(x)),
                             "n_iter": int(classifier.n_iter_), "fit_seconds": elapsed,
                             "n_features": int(x_train.shape[1]), "converged": True})
            model_dir.mkdir()
            for name, value in [("model.pkl", classifier), ("vectorizer.pkl", vectorizer)]:
                with (model_dir / name).open("wb") as stream:
                    pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            pd.DataFrame(rows).to_csv(metric_path, index=False)
            config = {"specification": specification, "n_iter": int(classifier.n_iter_),
                      "fit_seconds": elapsed, "converged": True,
                      "loss_curve_available": False,
                      "artifact_sha256": {name: digest(model_dir / name) for name in ["model.pkl", "vectorizer.pkl"]}}
            config["artifact_sha256"]["metrics.csv"] = digest(metric_path)
            write_json(model_dir / "config.json", config)
            records.extend(rows)
            LOG.info("Completed %s | val accuracy=%.6f macro-F1=%.6f | iterations=%d | %.2fs",
                     config_id, rows[1]["accuracy"], rows[1]["macro_f1"], classifier.n_iter_, elapsed)
    return pd.DataFrame(records)


def compare_validation(linear_metrics):
    records = []
    for path in sorted((ROOT / "results" / "baseline").glob("*_split_B_*_metrics.csv")):
        df = pd.read_csv(path)
        if not df["model"].isin(["svm", "mlp"]).all():
            continue
        df["config_id"] = path.name.removesuffix("_metrics.csv")
        records.append(df[df["eval_set"] == "val"])
    records.append(linear_metrics[linear_metrics["eval_set"] == "val"])
    validation = pd.concat(records, ignore_index=True)
    expected = {"svm": 3, "mlp": 6, "linear_svm": 3}
    for model in MODELS:
        for version in VERSIONS:
            group = validation[(validation["model"] == model) & (validation["version"] == version)]
            if len(group) != expected[model] or group["config_id"].duplicated().any():
                raise ValueError(f"Incomplete validation grid for {model}/{version}")
    validation.to_csv(OUT / "validation_comparison.csv", index=False)
    ranked = validation.sort_values(["accuracy", "macro_f1", "config_id"], ascending=[False, False, True])
    per_version = ranked.drop_duplicates(["model", "version"])
    per_version.to_csv(OUT / "best_by_model_version.csv", index=False)
    best = ranked.drop_duplicates("model").set_index("model").loc[MODELS].reset_index()
    # Freeze every choice and the exact input/model hashes before opening test CSVs.
    choices = []
    for row in best.to_dict("records"):
        model_dir = ROOT / "models" / row["config_id"]
        choices.append({"model": row["model"], "config_id": row["config_id"], "version": row["version"],
                        "val_accuracy": row["accuracy"], "val_macro_f1": row["macro_f1"],
                        "model_sha256": digest(model_dir / "model.pkl"),
                        "vectorizer_sha256": digest(model_dir / "vectorizer.pkl")})
    selection = {"split": "split_B", "rule": SELECTION_RULE, "choices": choices,
                 "validation_sha256": digest(OUT / "validation_comparison.csv")}
    selection_path = OUT / "selection_manifest.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text(encoding="utf-8")) != selection:
            raise ValueError("Frozen selection changed; refusing to replace completed experiment")
    else:
        write_json(selection_path, selection)
    best.to_csv(OUT / "best_by_model.csv", index=False)
    LOG.info("Validation selection frozen: %s", [(c["model"], c["config_id"]) for c in choices])
    return validation, per_version, selection


def evaluate_selected(selection):
    records = []
    for choice in selection["choices"]:
        config_id = choice["config_id"]
        test_path = ROOT / "data-processed" / "split_B" / choice["version"] / "test.csv"
        signature = {"config_id": config_id, "test_sha256": digest(test_path),
                     "model_sha256": choice["model_sha256"], "vectorizer_sha256": choice["vectorizer_sha256"]}
        output = OUT / "test" / f"{choice['model']}_final.json"
        if output.exists():
            result = json.loads(output.read_text(encoding="utf-8"))
            if result["signature"] != signature:
                raise ValueError(f"Cached final evaluation inputs changed: {config_id}")
            for filename, expected_hash in result["output_sha256"].items():
                if digest(output.parent / filename) != expected_hash:
                    raise ValueError(f"Cached test artifact changed: {filename}")
            LOG.info("Reused final test evaluation: %s", config_id)
        else:
            test = pd.read_csv(test_path, keep_default_na=False)
            if len(test) != 737:
                raise ValueError("Unexpected split_B test size")
            model_dir = ROOT / "models" / config_id
            vectorizer = load_pickle(model_dir / "vectorizer.pkl")
            classifier = load_pickle(model_dir / "model.pkl")
            prediction = classifier.predict(vectorizer.transform(test["text"]))
            pred_path = output.parent / f"{choice['model']}_predictions.csv"
            cm_path = output.parent / f"{choice['model']}_confusion_matrix.csv"
            pd.DataFrame({"test_row_1based": np.arange(1, len(test) + 1),
                          "target": test["target"], "predicted_label": prediction}).to_csv(pred_path, index=False)
            pd.DataFrame(confusion_matrix(test["target"], prediction, labels=np.arange(10)),
                         index=np.arange(10), columns=np.arange(10)).rename_axis("true_label").to_csv(cm_path)
            result = {"signature": signature, "n_test": len(test), "metrics": scores(test["target"], prediction),
                      "output_sha256": {p.name: digest(p) for p in [pred_path, cm_path]}}
            write_json(output, result)
            LOG.info("Final test evaluated once: %s | accuracy=%.6f", config_id, result["metrics"]["accuracy"])
        records.append({"model": choice["model"], "config_id": config_id, "version": choice["version"],
                        "val_accuracy": choice["val_accuracy"], "val_macro_f1": choice["val_macro_f1"],
                        **{f"test_{key}": value for key, value in result["metrics"].items()}, "n_test": result["n_test"]})
    final = pd.DataFrame(records)
    final.to_csv(OUT / "test_final_comparison.csv", index=False)
    return final


def plot_comparison(validation, per_version):
    mean = validation.groupby(["version", "model"])["accuracy"].mean().unstack("model").reindex(index=VERSIONS, columns=MODELS)
    best = per_version.pivot(index="version", columns="model", values="accuracy").reindex(index=VERSIONS, columns=MODELS)
    means = mean.stack().rename("mean_val_accuracy").reset_index()
    means.to_csv(OUT / "mean_by_model_version.csv", index=False)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.25, top=0.77, wspace=0.10)
    colors = ["#2878B5", "#E88A36", "#3B9970"]
    lower = max(0, np.floor((min(mean.min().min(), best.min().min()) - 0.04) / 0.05) * 0.05)
    for ax, values, title in zip(axes, [mean, best], ["Mean across hyperparameter configurations", "Best validation configuration per model/version"]):
        for i, model in enumerate(MODELS):
            bars = ax.bar(np.arange(4) + (i - 1) * 0.24, values[model], width=0.24,
                          color=colors[i], label=MODEL_LABELS[model], zorder=3)
            ax.bar_label(bars, labels=[f"{v:.1%}" for v in values[model]], padding=4, fontsize=8)
        ax.set_title(title, fontsize=12, pad=14)
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(["Raw email", "Body only", "Subject\n+ body", "Author + subject\n+ body"], fontsize=10)
        ax.set_ylim(lower, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Validation accuracy", fontsize=11)
    fig.suptitle("split_B (85/5/10): SGD SVM vs MLP vs LinearSVC", fontsize=18, fontweight="bold", y=0.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=3, frameon=False)
    fig.text(0.06, 0.07, "Same train/validation rows and TF-IDF. Configurations per version: SGD SVM = 3, MLP = 6, LinearSVC = 3.", fontsize=10)
    fig.text(0.06, 0.025, f"Validation data only; test is excluded from selection. Shared y-axis starts at {lower:.0%}. Known content duplicates are unchanged.", fontsize=9, color="#555555")
    output = ROOT / "results" / "evaluation" / "figures" / "model_comparison_split_B_with_linear_svm.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved comparison figure: %s", output)


def main():
    for directory in [OUT, OUT / "metrics", OUT / "test"]:
        directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / "experiment.log", encoding="utf-8")])
    LOG.info("Starting supplemental split_B experiment")
    linear = train_grid()
    linear.to_csv(OUT / "linear_svm_metrics.csv", index=False)
    validation, per_version, selection = compare_validation(linear)
    final = evaluate_selected(selection)
    plot_comparison(validation, per_version)
    manifest = {"status": "complete", "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
                "new_model_count": 12, "new_training_splits": ["split_B"], "training_rows": 6262,
                "validation_rows": 369, "test_rows": 737, "selection_rule": SELECTION_RULE,
                "loss_curve_available": False, "existing_models_retrained": False,
                "data_cleaning_or_splits_changed": False,
                "limitations": ["Existing SGD SVM and MLP previously accessed test during training.",
                                "Known cross-subset content duplicates remain; these are supplementary results.",
                                "LinearSVC squared_hinge differs from SGD hinge; this is not a solver-only ablation.",
                                "LinearSVC exposes final n_iter_, not an iteration loss history; no synthetic curve created.",
                                "Validation set is small and MLP has more hyperparameter configurations."]}
    write_json(OUT / "run_manifest.json", manifest)
    LOG.info("Experiment complete\n%s", final.to_string(index=False))


if __name__ == "__main__":
    main()
