"""
NB baseline — Naive Bayes (Multinomial)
======================================
- 数据集: split_A, split_B, split_C × raw / body_only / subject_body / subject_body_author
- 文本特征: TF-IDF (min_df=2, max_df=0.95, max_features=20000)
- 调参: alpha ∈ {0.01, 0.1, 0.5, 1.0}
- NB 是 batch EM 类算法,无逐 epoch 迭代;这里记录训练集/验证集随 alpha 的准确率曲线
- 输出:
    results/baseline/NB_*.csv    — 最终指标
    results/loss/NB_*.csv         — alpha sweep 训练准确率曲线

用法:
  python scripts/baselines/train_nb.py --split A --version raw
  python scripts/baselines/train_nb.py --split all --version all
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data-processed"
OUT_BLINE = ROOT / "results" / "baseline"
OUT_LOSS  = ROOT / "results" / "loss"
OUT_BLINE.mkdir(parents=True, exist_ok=True)
OUT_LOSS.mkdir(parents=True, exist_ok=True)

SPLITS     = ["A", "B", "C"]
VERSIONS   = ["raw", "body_only", "subject_body", "subject_body_author"]
ALPHA_GRID = [0.01, 0.1, 0.5, 1.0]

def load_split(split, version):
    base  = DATA_ROOT / f"split_{split}"
    train = pd.read_csv(base / f"train_{version}.csv")
    val   = pd.read_csv(base / f"val_{version}.csv")
    test  = pd.read_csv(base / f"test_{version}.csv")
    return train, val, test

def train_and_eval(split, version):
    print(f"\n[{split}] {version}")
    train_df, val_df, test_df = load_split(split, version)

    X_train = train_df["text"].astype(str).tolist()
    y_train = train_df["target"].values
    X_val   = val_df["text"].astype(str).tolist()
    y_val   = val_df["target"].values
    X_test  = test_df["text"].astype(str).tolist()
    y_test  = test_df["target"].values

    vec = TfidfVectorizer(min_df=2, max_df=0.95, max_features=20000)
    X_tr_v = vec.fit_transform(X_train)
    X_va_v = vec.transform(X_val)
    X_te_v = vec.transform(X_test)

    metric_rows = []
    loss_rows   = []

    for alpha in ALPHA_GRID:
        clf = MultinomialNB(alpha=alpha)
        clf.fit(X_tr_v, y_train)

        tr_pred = clf.predict(X_tr_v)
        va_pred = clf.predict(X_va_v)
        te_pred = clf.predict(X_te_v)

        metric_rows.append({
            "split": split, "version": version,
            "alpha": alpha,
            "train_acc": accuracy_score(y_train, tr_pred),
            "val_acc":   accuracy_score(y_val,  va_pred),
            "test_acc":  accuracy_score(y_test, te_pred),
            "val_f1":    f1_score(y_val,  va_pred, average="macro"),
            "test_f1":   f1_score(y_test, te_pred, average="macro"),
        })
        print(f"  alpha={alpha:.2f}  train_acc={metric_rows[-1]['train_acc']:.4f}  "
              f"val_acc={metric_rows[-1]['val_acc']:.4f}  test_acc={metric_rows[-1]['test_acc']:.4f}")

        # 记录 alpha sweep 曲线 (每个 alpha 对应一个点)
        loss_rows.append({
            "split": split,
            "version": version,
            "alpha": alpha,
            "step":  ALPHA_GRID.index(alpha) + 1,
            "train_acc": metric_rows[-1]["train_acc"],
            "val_acc":   metric_rows[-1]["val_acc"],
        })

    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(OUT_BLINE / f"NB_{split}_{version}.csv", index=False)

    loss_df = pd.DataFrame(loss_rows)
    loss_df.to_csv(OUT_LOSS / f"NB_{split}_{version}.csv", index=False)

    return metric_df, loss_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",   default="A")
    ap.add_argument("--version", default="raw")
    args = ap.parse_args()

    splits   = SPLITS   if args.split   == "all" else [args.split]
    versions = VERSIONS if args.version == "all" else [args.version]

    all_metrics = []
    all_losses  = []

    for sp in splits:
        for ver in versions:
            m, l = train_and_eval(sp, ver)
            all_metrics.append(m)
            all_losses.append(l)

    if len(all_metrics) > 1:
        pd.concat(all_metrics, ignore_index=True).to_csv(
            OUT_BLINE / "NB_all_splits_all_versions.csv", index=False)
        pd.concat(all_losses, ignore_index=True).to_csv(
            OUT_LOSS / "NB_all_splits_all_versions.csv", index=False)

if __name__ == "__main__":
    main()
