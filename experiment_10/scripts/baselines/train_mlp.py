"""
MLP baseline — Multi-Layer Perceptron
======================================
- 数据集: split_A, split_B, split_C × raw / body_only / subject_body / subject_body_author
- 文本特征: TF-IDF (min_df=2, max_df=0.95, max_features=20000)
- 调参: hidden_layer_sizes ∈ {(50,), (100,), (100, 50)} × alpha ∈ {0.0001, 0.001}
- 流程:
    1. 网格搜索阶段：只用训练集拟合，验证集选最优超参数（val_acc + val_f1）
    2. 测试集在最优配置选定后才评估一次，不参与排序
    3. 最终模型在 (训练集+验证集) 上重训
- 输出:
    results/baseline/MLP_*.csv              — 网格搜索指标（train/val）
    results/loss/MLP_*.csv                  — 每 epoch 训练 loss
    results/baseline/MLP_*_final.csv        — 最终重训后测试集指标

用法:
  python scripts/baselines/train_mlp.py --split A --version raw
  python scripts/baselines/train_mlp.py --split all --version all
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
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

SPLITS     = ["A", "B", "C"]
VERSIONS   = ["raw", "body_only", "subject_body", "subject_body_author"]
HIDDEN_GRID = [(50,), (100,), (100, 50)]
ALPHA_GRID  = [0.0001, 0.001]

# 不早停：每个配置固定跑满 MAX_ITER 个 epoch，保证 loss curve 完整且各配置可比
MAX_ITER = 300


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

    # ── 网格搜索阶段：只用 train + val ─────────────────────────
    grid_rows = []   # 保存每个 (hidden, alpha) 的 train/val 指标
    loss_rows = []   # 保存每个配置逐 epoch loss

    for hidden in HIDDEN_GRID:
        for alpha in ALPHA_GRID:
            config_label = f"hidden={hidden},alpha={alpha}"

            clf = MLPClassifier(
                hidden_layer_sizes=hidden,
                alpha=alpha,
                max_iter=MAX_ITER,
                random_state=42,
                warm_start=False,
                early_stopping=False,
                tol=0.0,
                n_iter_no_change=MAX_ITER + 1,
            )
            clf.fit(X_tr_v, y_train)

            va_pred = clf.predict(X_va_v)
            tr_pred = clf.predict(X_tr_v)

            grid_rows.append({
                "split": split,
                "version": version,
                "hidden_layer_sizes": str(hidden),
                "alpha": alpha,
                "train_acc": accuracy_score(y_train, tr_pred),
                "val_acc":   accuracy_score(y_val,  va_pred),
                "val_f1":    f1_score(y_val, va_pred, average="macro"),
                "n_epochs":  clf.n_iter_,
                "config_label": config_label,
            })
            print(f"  {config_label}  epochs={clf.n_iter_}  "
                  f"val_acc={grid_rows[-1]['val_acc']:.4f}")

            # Loss curve (MLP 内置 loss_curve_)
            if hasattr(clf, "loss_curve_") and clf.loss_curve_ is not None:
                for epoch, loss in enumerate(clf.loss_curve_, start=1):
                    loss_rows.append({
                        "split": split,
                        "version": version,
                        "hidden_layer_sizes": str(hidden),
                        "alpha": alpha,
                        "epoch": epoch,
                        "loss": loss,
                        "config_label": config_label,
                    })

    # ── 选出最优配置（只依据 val_acc，并列取 val_f1 更高者）─────
    grid_df = pd.DataFrame(grid_rows)
    grid_df = grid_df.sort_values(["val_acc", "val_f1"], ascending=False)
    best_row = grid_df.iloc[0]
    best_hidden = eval(best_row["hidden_layer_sizes"])
    best_alpha  = best_row["alpha"]
    print(f"  最优: hidden={best_hidden} alpha={best_alpha}  "
          f"val_acc={best_row['val_acc']:.4f}")

    # ── 最优配置在 train+val 上重训 ──────────────────────────
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    X_full = train_val_df["text"].astype(str).tolist()
    y_full = train_val_df["target"].values

    vec_full = TfidfVectorizer(min_df=2, max_df=0.95, max_features=20000)
    X_full_v = vec_full.fit_transform(X_full)
    X_test_v = vec_full.transform(X_test)

    clf_final = MLPClassifier(
        hidden_layer_sizes=best_hidden,
        alpha=best_alpha,
        max_iter=MAX_ITER,
        tol=0.0,
        n_iter_no_change=MAX_ITER + 1,
        random_state=42,
        early_stopping=False,
    )
    clf_final.fit(X_full_v, y_full)
    te_pred = clf_final.predict(X_test_v)

    final_metrics = {
        "split": split,
        "version": version,
        "hidden_layer_sizes": str(best_hidden),
        "alpha": best_alpha,
        "val_acc":  best_row["val_acc"],
        "val_f1":   best_row["val_f1"],
        "test_acc": accuracy_score(y_test, te_pred),
        "test_f1":  f1_score(y_test, te_pred, average="macro"),
        "note": "test set evaluated only after best config selected on val",
    }
    final_df = pd.DataFrame([final_metrics])
    joblib.dump((vec_full, clf_final), MODEL_DIR / f"MLP_{split}_{version}_best.joblib")

    # ── 保存 ──────────────────────────────────────────────────
    grid_df.to_csv(OUT_BLINE / f"MLP_{split}_{version}.csv", index=False)
    if loss_rows:
        loss_df = pd.DataFrame(loss_rows)
        loss_df.to_csv(OUT_LOSS / f"MLP_{split}_{version}.csv", index=False)
    final_df.to_csv(OUT_BLINE / f"MLP_{split}_{version}_final.csv", index=False)
    print(f"  最终: val_acc={final_metrics['val_acc']:.4f}  "
          f"test_acc={final_metrics['test_acc']:.4f}")

    return grid_df, loss_df if loss_rows else None, final_df


def main():
    global MAX_ITER
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",   default="A")
    ap.add_argument("--version", default="raw")
    ap.add_argument("--max-iter", type=int, default=MAX_ITER,
                    help="override the epoch count for timing or smoke tests")
    args = ap.parse_args()

    MAX_ITER = args.max_iter

    splits   = SPLITS   if args.split   == "all" else [args.split]
    versions = VERSIONS if args.version == "all" else [args.version]

    all_grid  = []
    all_loss  = []
    all_final = []

    for sp in splits:
        for ver in versions:
            g, l, f = train_and_eval(sp, ver)
            all_grid.append(g)
            if l is not None:
                all_loss.append(l)
            all_final.append(f)

    # Only the explicit all/all run writes aggregate files. Split-specific
    # terminals write independent per-combination files; merge_results.py
    # creates the aggregate after all terminals finish.
    if args.split == "all" and args.version == "all":
        pd.concat(all_grid, ignore_index=True).to_csv(
            OUT_BLINE / "MLP_all_splits_all_versions.csv", index=False)
        if all_loss:
            pd.concat(all_loss, ignore_index=True).to_csv(
                OUT_LOSS / "MLP_all_splits_all_versions.csv", index=False)
        pd.concat(all_final, ignore_index=True).to_csv(
            OUT_BLINE / "MLP_all_splits_all_versions_final.csv", index=False)


if __name__ == "__main__":
    main()
