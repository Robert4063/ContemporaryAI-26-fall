"""
SVM baseline — SGDClassifier (hinge loss, 模拟线性 SVM)
=======================================================
- 数据集: split_A, split_B, split_C × raw / body_only / subject_body / subject_body_author
- 文本特征: TF-IDF (min_df=2, max_df=0.95, max_features=20000)
- 调参: C ∈ {0.1, 1.0, 10.0} — 网格搜索阶段只用训练集和验证集
- 流程:
    1. 对每个 C，用训练集拟合，记录训练 loss 曲线（逐 epoch）
    2. 在验证集上评估，选出 val_acc 最高的 C
    3. 用最优 C 在 (训练集+验证集) 上重训
    4. 在测试集上只预测一次，报告最终指标
- 输出:
    results/baseline/SVM_*.csv   — 网格搜索指标（train/val，不含测试集参与排序）
    results/loss/SVM_*.csv       — 每 epoch 训练 loss
    results/baseline/SVM_*_final.csv — 最终重训后测试集指标（不走网格，不参与排序比较）

用法:
  python scripts/baselines/train_svm.py --split A --version raw
  python scripts/baselines/train_svm.py --split all --version all
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data-processed"
OUT_BLINE = ROOT / "results" / "baseline"
OUT_LOSS  = ROOT / "results" / "loss"
OUT_BLINE.mkdir(parents=True, exist_ok=True)
OUT_LOSS.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / "models" / "svm_mlp"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SPLITS   = ["A", "B", "C"]
VERSIONS = ["raw", "body_only", "subject_body", "subject_body_author"]
C_GRID   = [0.1, 1.0, 10.0]
MAX_ITER = 300   # 每个 C 固定跑 MAX_ITER 个 epoch，保证 loss curve 完整可比


def load_split(split, version):
    base  = DATA_ROOT / f"split_{split}"
    train = pd.read_csv(base / f"train_{version}.csv")
    val   = pd.read_csv(base / f"val_{version}.csv")
    test  = pd.read_csv(base / f"test_{version}.csv")
    return train, val, test


def hinge_loss_raw(X, y, clf):
    """Multiclass Crammer-Singer hinge loss on the training examples."""
    scores = clf.decision_function(X)
    class_indices = {label: i for i, label in enumerate(clf.classes_)}
    target_indices = np.array([class_indices[label] for label in y])
    true_scores = scores[np.arange(len(y)), target_indices]
    margins = scores - true_scores[:, None] + 1.0
    margins[np.arange(len(y)), target_indices] = 0.0
    return float(np.maximum(0.0, margins.max(axis=1)).mean())


def train_and_eval(split, version):
    print(f"\n[{split}] {version}")
    train_df, val_df, test_df = load_split(split, version)

    X_train = train_df["text"].astype(str).tolist()
    y_train = train_df["target"].values
    X_val   = val_df["text"].astype(str).tolist()
    y_val   = val_df["target"].values
    X_test  = test_df["text"].astype(str).tolist()
    y_test  = test_df["target"].values

    # 同一份 TF-IDF 向量化器（用于网格搜索阶段）
    vec = TfidfVectorizer(min_df=2, max_df=0.95, max_features=20000)
    X_tr_v = vec.fit_transform(X_train)
    X_va_v = vec.transform(X_val)
    X_te_v = vec.transform(X_test)

    # ── 网格搜索阶段：只用 train + val ─────────────────────────
    grid_rows = []
    loss_rows = []
    best_clfs = {}

    for c_idx, C in enumerate(C_GRID, start=1):
        clf = SGDClassifier(
            loss="hinge", penalty="l2", alpha=1.0 / (C * len(y_train)),
            max_iter=1, tol=None, random_state=42,
            warm_start=True, learning_rate="optimal",
        )
        epoch_losses = []
        for epoch in range(1, MAX_ITER + 1):
            clf.partial_fit(X_tr_v, y_train, classes=np.unique(y_train))
            epoch_losses.append(hinge_loss_raw(X_tr_v, y_train, clf))
            if epoch % 25 == 0:
                print(f"  {split}/{version} C={C:g}: epoch {epoch}/{MAX_ITER}", flush=True)

        va_pred = clf.predict(X_va_v)
        tr_pred = clf.predict(X_tr_v)
        grid_rows.append({
            "split": split, "version": version, "C": C,
            "train_acc": accuracy_score(y_train, tr_pred),
            "val_acc": accuracy_score(y_val, va_pred),
            "val_f1": f1_score(y_val, va_pred, average="macro"),
        })
        best_clfs[C] = clf
        loss_rows.extend({
            "split": split, "version": version, "C": C,
            "step": epoch, "epoch": epoch, "train_loss": loss, "loss": loss,
        } for epoch, loss in enumerate(epoch_losses, start=1))
        print(f"  {split}/{version} candidate {c_idx}/{len(C_GRID)} complete: "
              f"C={C:g} val_acc={grid_rows[-1]['val_acc']:.4f}", flush=True)

    # ── Grid selection: no test set metrics are read here ────────
    grid_df = pd.DataFrame(grid_rows)
    best_row = grid_df.sort_values(["val_acc", "val_f1"], ascending=False).iloc[0]
    best_C = float(best_row["C"])
    print(f"  best C={best_C:g} (val_acc={best_row['val_acc']:.4f})", flush=True)

    # Retrain the selected configuration on train+val, then evaluate test once.
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    vec_full = TfidfVectorizer(min_df=2, max_df=0.95, max_features=20000)
    X_full_v = vec_full.fit_transform(full_df["text"].astype(str))
    y_full = full_df["target"].values
    clf_final = SGDClassifier(
        loss="hinge", penalty="l2", alpha=1.0 / (best_C * len(y_full)),
        max_iter=MAX_ITER, tol=None, random_state=42,
        learning_rate="optimal",
    )
    clf_final.fit(X_full_v, y_full)
    te_pred = clf_final.predict(vec_full.transform(X_test))
    final_df = pd.DataFrame([{
        "split": split, "version": version, "C": best_C,
        "val_acc": best_row["val_acc"], "val_f1": best_row["val_f1"],
        "test_acc": accuracy_score(y_test, te_pred),
        "test_f1": f1_score(y_test, te_pred, average="macro"),
    }])
    joblib.dump((vec_full, clf_final), MODEL_DIR / f"SVM_{split}_{version}_best.joblib")

    grid_df.to_csv(OUT_BLINE / f"SVM_{split}_{version}.csv", index=False)
    loss_df = pd.DataFrame(loss_rows)
    loss_df.to_csv(OUT_LOSS / f"SVM_{split}_{version}.csv", index=False)
    final_df.to_csv(OUT_BLINE / f"SVM_{split}_{version}_final.csv", index=False)
    print(f"  completed {split}/{version}: test_acc={final_df.iloc[0]['test_acc']:.4f}", flush=True)
    return grid_df, loss_df, final_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",   default="A")
    ap.add_argument("--version", default="raw")
    args = ap.parse_args()

    splits   = SPLITS   if args.split   == "all" else [args.split]
    versions = VERSIONS if args.version == "all" else [args.version]

    all_grid   = []
    all_loss   = []
    all_final  = []

    for sp in splits:
        for ver in versions:
            g, l, f = train_and_eval(sp, ver)
            all_grid.append(g)
            all_loss.append(l)
            all_final.append(f)

    if len(all_grid) > 1:
        pd.concat(all_grid, ignore_index=True).to_csv(
            OUT_BLINE / "SVM_all_splits_all_versions.csv", index=False)
        pd.concat(all_loss, ignore_index=True).to_csv(
            OUT_LOSS / "SVM_all_splits_all_versions.csv", index=False)
        pd.concat(all_final, ignore_index=True).to_csv(
            OUT_BLINE / "SVM_all_splits_all_versions_final.csv", index=False)

if __name__ == "__main__":
    main()
