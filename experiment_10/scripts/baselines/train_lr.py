"""
LR baseline — Logistic Regression
=================================
- 数据集: split_A, split_B, split_C × raw / body_only / subject_body / subject_body_author
- 文本特征: TF-IDF (min_df=2, max_df=0.95, max_features=20000)
- 调参: C ∈ {0.01, 0.1, 1.0, 10.0}
- 不早停: lbfgs 用 warm_start 分段推进, 每段 STEP 次迭代, 固定跑满 MAX_ITER 次,
          每段结束记录一次训练/验证 log loss, 得到完整 loss 曲线
- 输出:
    results/baseline/LR_*.csv       — 最终指标
    results/loss/LR_*.csv           — 每个检查点的训练/验证 log loss

用法:
  python scripts/baselines/train_lr.py --split A --version raw
  python scripts/baselines/train_lr.py --split all --version all
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data-processed"
OUT_BLINE = ROOT / "results" / "baseline"
OUT_LOSS  = ROOT / "results" / "loss"
OUT_BLINE.mkdir(parents=True, exist_ok=True)
OUT_LOSS.mkdir(parents=True, exist_ok=True)

SPLITS   = ["A", "B", "C"]
VERSIONS = ["raw", "body_only", "subject_body", "subject_body_author"]
C_GRID   = [0.01, 0.1, 1.0, 10.0]

MAX_ITER = 300   # 总迭代数, 不因收敛提前结束
STEP     = 5     # 每隔多少次迭代记录一次 loss


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

    for C in C_GRID:
        # tol=0: 不因梯度足够小而停止; warm_start: 每次 fit 从上次系数继续
        clf = LogisticRegression(
            C=C, solver="lbfgs", max_iter=STEP, tol=0.0,
            warm_start=True, random_state=42,
        )
        for done in range(STEP, MAX_ITER + 1, STEP):
            clf.fit(X_tr_v, y_train)
            loss_rows.append({
                "split": split, "version": version, "C": C,
                "epoch": done,
                "loss":     log_loss(y_train, clf.predict_proba(X_tr_v), labels=clf.classes_),
                "val_loss": log_loss(y_val,   clf.predict_proba(X_va_v), labels=clf.classes_),
            })

        va_pred = clf.predict(X_va_v)
        te_pred = clf.predict(X_te_v)
        metric_rows.append({
            "split": split, "version": version, "C": C,
            "train_acc": accuracy_score(y_train, clf.predict(X_tr_v)),
            "val_acc":   accuracy_score(y_val,  va_pred),
            "test_acc":  accuracy_score(y_test, te_pred),
            "val_f1":    f1_score(y_val,  va_pred, average="macro"),
            "test_f1":   f1_score(y_test, te_pred, average="macro"),
            "n_iter":    MAX_ITER,
            "final_train_loss": loss_rows[-1]["loss"],
            "final_val_loss":   loss_rows[-1]["val_loss"],
        })
        print(f"  C={C:6.2f}  iters={MAX_ITER}  "
              f"train_loss={loss_rows[-1]['loss']:.4f}  val_loss={loss_rows[-1]['val_loss']:.4f}  "
              f"val_acc={metric_rows[-1]['val_acc']:.4f}  test_acc={metric_rows[-1]['test_acc']:.4f}")

    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(OUT_BLINE / f"LR_{split}_{version}.csv", index=False)
    loss_df = pd.DataFrame(loss_rows)
    loss_df.to_csv(OUT_LOSS / f"LR_{split}_{version}.csv", index=False)
    return metric_df, loss_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",   default="A")
    ap.add_argument("--version", default="raw")
    args = ap.parse_args()

    splits   = SPLITS   if args.split   == "all" else [args.split]
    versions = VERSIONS if args.version == "all" else [args.version]

    all_metrics, all_losses = [], []
    for sp in splits:
        for ver in versions:
            m, l = train_and_eval(sp, ver)
            all_metrics.append(m)
            all_losses.append(l)

    if len(all_metrics) > 1:
        pd.concat(all_metrics, ignore_index=True).to_csv(
            OUT_BLINE / "LR_all_splits_all_versions.csv", index=False)
        pd.concat(all_losses, ignore_index=True).to_csv(
            OUT_LOSS / "LR_all_splits_all_versions.csv", index=False)


if __name__ == "__main__":
    main()
