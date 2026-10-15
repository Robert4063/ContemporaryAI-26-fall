"""Evidence, plots and final prediction for the two-page notebook report.

Read the per-experiment CSVs rather than possibly stale merged files. Select
configurations using validation metrics only; leave the original outputs intact.
"""
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "assets"
SPLITS = ["A", "B", "C"]
RATIOS = {"A": "80:10:10", "B": "85:5:10", "C": "90:5:5"}
VERSIONS = ["raw", "body_only", "subject_body", "subject_body_author"]
NAMES = {"raw": "原文", "body_only": "正文", "subject_body": "标题+正文",
         "subject_body_author": "标题+正文+作者"}
MODELS = ["SVM", "MLP"]


def load_evidence():
    """Validate coverage and alignment; freeze ranking without reading test scores."""
    OUT.mkdir(parents=True, exist_ok=True)
    labeled = pd.read_csv(ROOT / "data-Project1/train_data.csv")
    unlabeled = pd.read_csv(ROOT / "data-Project1/test_data_unlabeled.csv")
    assert labeled.columns.tolist() == ["text", "target"]
    assert labeled.notna().all().all() and unlabeled.text.notna().all()
    assert labeled.target.nunique() == 10 and labeled.text.duplicated().sum() == 0
    frames, sources = [], [ROOT / "data-Project1/train_data.csv",
                          ROOT / "data-Project1/test_data_unlabeled.csv"]
    for model in MODELS:
        for split in SPLITS:
            for version in VERSIONS:
                path = ROOT / f"results/baseline/{model}_{split}_{version}.csv"
                d = pd.read_csv(path, keep_default_na=False).assign(model=model)
                assert d.split.eq(split).all() and d.version.eq(version).all()
                frames.append(d)
                sources.append(path)
    metrics = pd.concat(frames, ignore_index=True)
    assert len(metrics) == 108, "Expected 12 x (3+6) configurations"
    # Ranking keys deliberately exclude all test columns.
    rank = metrics.sort_values(["val_acc", "val_f1"], ascending=False, kind="stable")
    best = rank.drop_duplicates("model").set_index("model").loc[MODELS].reset_index()
    winner = rank.iloc[0]
    assert winner.model == "SVM" and winner.split == "B" and winner.version == "subject_body_author"
    assert winner.C == 1.0, "If evidence changes, rebuild and review the report"
    best.to_csv(OUT / "selected_configs.csv", index=False)
    rows = []
    for split in SPLITS:
        sets = {}
        for subset in ["train", "val", "test"]:
            path = ROOT / f"data-processed/split_{split}/{subset}_raw.csv"
            raw = pd.read_csv(path)
            sources.append(path)
            sets[subset] = raw
            for version in VERSIONS[1:]:
                vp = ROOT / f"data-processed/split_{split}/{subset}_{version}.csv"
                processed = pd.read_csv(vp)
                sources.append(vp)
                assert raw.target.equals(processed.target), "Version labels/order differ"
                assert len(raw) == len(processed)
            assert raw.target.nunique() == 10
        assert sum(len(d) for d in sets.values()) == len(labeled)
        for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
            assert not (set(sets[left].text) & set(sets[right].text))
        rows.append([split, RATIOS[split], *(len(sets[s]) for s in ["train", "val", "test"])])
    lengths = labeled.text.str.len()
    stats = {"labeled": len(labeled), "unlabeled": len(unlabeled),
             "classes": labeled.target.value_counts().sort_index().to_dict(),
             "mean_chars": float(lengths.mean()), "median_chars": int(lengths.median()),
             "p95_chars": float(lengths.quantile(.95)), "max_chars": int(lengths.max())}
    return {"metrics": metrics, "best": best, "winner": winner, "stats": stats,
            "split_rows": rows, "sources": sources}


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def parameter_label(row):
    if row.model in ["LR", "SVM"]:
        return f"C={row.C:g}"
    if row.model == "NB":
        return f"α={row.alpha:g}"
    if row.model == "DT":
        return f"depth={row.max_depth}, leaf={int(row.min_samples_leaf)}"
    return f"h={str(row.hidden_layer_sizes).replace(' ', '')}, α={row.alpha:g}"


def results_table(ctx):
    rows = []
    for _, row in ctx["best"].iterrows():
        rows.append([row.model, NAMES[row.version], parameter_label(row),
                     f"{100 * row.val_acc:.2f}", f"{100 * row.val_f1:.2f}",
                     f"{100 * row.test_acc:.2f}", f"{100 * row.test_f1:.2f}"])
    return markdown_table(["模型", "文本", "验证选定参数", "Val Acc%", "Val F1%",
                           "Test Acc%", "Test F1%"], rows)


def make_comparison(ctx):
    """Four panels in one image; no smoothing and no invented losses."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 9,
                         "axes.titlesize": 10, "legend.fontsize": 7.5})
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.15), constrained_layout=True)
    colors = ["#2463a1", "#df7b29", "#338763", "#a35386"]

    def mlp(split, version):
        path = ROOT / f"results/loss/MLP_{split}_{version}.csv"
        ctx["sources"].append(path)
        d = pd.read_csv(path)
        d = d[(d.hidden_layer_sizes.str.replace(" ", "") == "(100,50)") &
              np.isclose(d.alpha, .001)].sort_values("epoch")
        assert d.epoch.tolist() == list(range(1, 301))
        assert d.loss.gt(0).all() and np.isfinite(d.loss).all()
        return d

    for split, color in zip(SPLITS, colors):
        d = mlp(split, "raw")
        axes[0, 0].plot(d.epoch, d.loss, color=color, lw=1.2, label=RATIOS[split])
    axes[0, 0].set_title("(a) MLP：改变划分，固定原文")
    for version, color in zip(VERSIONS, colors):
        d = mlp("B", version)
        axes[0, 1].plot(d.epoch, d.loss, color=color, lw=1.2, label=NAMES[version])
    axes[0, 1].set_title("(b) MLP：改变文本，固定 85:5:10")
    for ax in axes[0]:
        ax.set(xlabel="Epoch", ylabel="训练 loss（对数轴）", yscale="log")
        ax.legend(loc="upper right", frameon=False)
    path = ROOT / "results/loss/MLP_B_subject_body_author.csv"
    ctx["sources"].append(path)
    mlp_records = pd.read_csv(path)
    for hidden, color in zip(["(50,)", "(100,)", "(100,50)"], colors):
        d = mlp_records[(mlp_records.hidden_layer_sizes.str.replace(" ", "") == hidden) &
                        np.isclose(mlp_records.alpha, .001)].sort_values("epoch")
        assert d.epoch.tolist() == list(range(1, 301))
        axes[1, 0].plot(d.epoch, d.loss, color=color, lw=1.2, label=f"h={hidden}")
    axes[1, 0].set(title="(c) MLP：85:5:10、标题+正文+作者",
                   xlabel="Epoch", ylabel="训练 loss（对数轴）", yscale="log")
    axes[1, 0].legend(loc="upper right", frameon=False)
    svm = ctx["metrics"][(ctx["metrics"].model == "SVM") & (ctx["metrics"].split == "B")]
    for version, color in zip(VERSIONS, colors):
        d = svm[svm.version == version].sort_values("C")
        axes[1, 1].plot(d.C, 100 * d.val_acc, "o-", color=color, lw=1.3,
                        markersize=3.5, label=NAMES[version])
    axes[1, 1].set(title="(d) SVM：85:5:10，验证集调参",
                   xlabel="C（对数轴）", ylabel="验证 Accuracy（%）", xscale="log")
    axes[1, 1].set_xticks([.1, 1, 10], ["0.1", "1", "10"])
    axes[1, 1].legend(loc="lower right", frameon=False)
    for ax in axes.flat:
        ax.grid(alpha=.2, lw=.6)
        ax.spines[["top", "right"]].set_visible(False)
    path = OUT / "experiment_comparison.png"
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)
    return path


def verify_and_predict(ctx):
    """Reproduce only the locked SVM, fit train only, and preserve supplied CSVs."""
    w = ctx["winner"]
    split_dir = ROOT / f"data-processed/split_{w.split}"
    train = pd.read_csv(split_dir / f"train_{w.version}.csv")
    val = pd.read_csv(split_dir / f"val_{w.version}.csv")
    test = pd.read_csv(split_dir / f"test_{w.version}.csv")
    unlabeled_path = split_dir / f"unlabeled_{w.version}.csv"
    unlabeled = pd.read_csv(unlabeled_path).fillna({"text": ""})
    ctx["sources"].append(unlabeled_path)
    vectorizer = TfidfVectorizer(min_df=2, max_df=.95, max_features=20000, norm="l2")
    x_train = vectorizer.fit_transform(train.text)
    model = LinearSVC(C=float(w.C), random_state=42, max_iter=5000)
    model.fit(x_train, train.target)
    val_pred = model.predict(vectorizer.transform(val.text))
    test_pred = model.predict(vectorizer.transform(test.text))
    scores = {"val_acc": accuracy_score(val.target, val_pred),
              "val_f1": f1_score(val.target, val_pred, average="macro"),
              "test_acc": accuracy_score(test.target, test_pred),
              "test_f1": f1_score(test.target, test_pred, average="macro"),
              "test_precision": precision_score(test.target, test_pred, average="macro", zero_division=0),
              "test_recall": recall_score(test.target, test_pred, average="macro", zero_division=0),
              "n_features": x_train.shape[1], "n_test_errors": int(sum(test_pred != test.target)),
              "sparsity": float(1 - x_train.nnz / np.prod(x_train.shape))}
    for key in ["val_acc", "val_f1", "test_acc", "test_f1"]:
        assert np.isclose(scores[key], w[key], atol=1e-10), f"Reproduction mismatch: {key}"
    prediction = model.predict(vectorizer.transform(unlabeled.text))
    assert len(prediction) == 2457 and set(prediction).issubset(set(train.target))
    path = ROOT / "reports" / "predictions.csv"
    pd.Series(prediction).to_csv(path, index=False, header=False)
    cm = confusion_matrix(test.target, test_pred, labels=list(range(10)))
    pd.DataFrame(cm, index=range(10), columns=range(10)).to_csv(OUT / "confusion_matrix.csv")
    pd.DataFrame([scores]).to_csv(OUT / "reproduced_svm_metrics.csv", index=False)
    ctx["scores"] = scores
    return scores


def write_manifest(ctx):
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(set(ctx["sources"]))}
    manifest = {"seed": 42, "python": platform.python_version(),
                "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                "selection": "validation accuracy descending, then validation macro F1 descending; stable ties",
                "models": MODELS, "versions": VERSIONS, "ratios": RATIOS,
                "grid_rows": len(ctx["metrics"]), "final_model_fit": "training set only",
                "historical_test_access": "Legacy scripts evaluated each grid candidate on test; retrospective limitation",
                "reproduced_scores": ctx["scores"], "input_sha256": hashes}
    (OUT / "evidence_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
