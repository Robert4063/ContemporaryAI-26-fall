"""
predict.py — 用验证集最优配置预测 unlabeled 数据
=================================================
训练脚本没有保存模型, 所以这里按 results/baseline/ 里的结果重新训练一次:
  1. 每个模型按 val_acc 在 (split, version, 超参) 中选出最优配置
  2. 按 val_acc 选出最优模型
  3. TF-IDF 与模型都在 train+val 上重新拟合 (测试集不参与), 预测 unlabeled_{version}.csv
  4. 保存为无表头、无索引、每行一个标签的 predictions.csv (与示例文件格式一致)

用法:
  python scripts/baselines/predict.py              # 用验证集最优模型
  python scripts/baselines/predict.py --model svm  # 指定模型 (仍按验证集选其最优配置)
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data-processed"
BLINE_DIR = ROOT / "results" / "baseline"
OUT_DIR   = ROOT / "results" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["NB", "LR", "SVM", "DT", "MLP"]


def build_model(name, row):
    """按 baseline 结果里的一行超参构造与训练脚本一致的模型。"""
    if name == "NB":
        return MultinomialNB(alpha=float(row["alpha"]))
    if name == "LR":
        # 训练脚本: warm_start 分段推进, 总共 300 次迭代, tol=0
        return LogisticRegression(C=float(row["C"]), solver="lbfgs",
                                  max_iter=300, tol=0.0, random_state=42)
    if name == "SVM":
        return LinearSVC(C=float(row["C"]), random_state=42, max_iter=5000)
    if name == "DT":
        depth = None if str(row["max_depth"]) in ("None", "nan") else int(float(row["max_depth"]))
        return DecisionTreeClassifier(max_depth=depth,
                                      min_samples_leaf=int(row["min_samples_leaf"]),
                                      random_state=42)
    if name == "MLP":
        hidden = tuple(int(x) for x in str(row["hidden_layer_sizes"]).strip("()").split(",") if x.strip())
        return MLPClassifier(hidden_layer_sizes=hidden, alpha=float(row["alpha"]),
                             max_iter=300, tol=0.0, n_iter_no_change=301,
                             early_stopping=False, random_state=42)
    raise ValueError(name)


def best_row(name):
    """在该模型所有 (split, version, 超参) 里取 val_acc 最大的一行; 并列时取 val_f1 更高者。"""
    df = pd.read_csv(BLINE_DIR / f"{name}_all_splits_all_versions.csv")
    df = df.sort_values(["val_acc", "val_f1"], ascending=False)
    return df.iloc[0]


def fit_predict(name, row):
    split, version = row["split"], row["version"]
    base = DATA_ROOT / f"split_{split}"
    train = pd.read_csv(base / f"train_{version}.csv")
    val   = pd.read_csv(base / f"val_{version}.csv")
    unl   = pd.read_csv(base / f"unlabeled_{version}.csv")

    full = pd.concat([train, val], ignore_index=True)       # train + val, 不含 test
    vec  = TfidfVectorizer(min_df=2, max_df=0.95, max_features=20000)
    X    = vec.fit_transform(full["text"].astype(str))
    clf  = build_model(name, row)
    clf.fit(X, full["target"].values)
    return clf.predict(vec.transform(unl["text"].astype(str)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="auto", choices=["auto"] + [m.lower() for m in MODELS])
    args = ap.parse_args()

    if args.model == "auto":
        rows = {m: best_row(m) for m in MODELS}
        name = max(rows, key=lambda m: rows[m]["val_acc"])
    else:
        name = args.model.upper()
        rows = {name: best_row(name)}
    row = rows[name]

    hp = {k: row[k] for k in row.index
          if k not in ("split", "version", "train_acc", "val_acc", "test_acc", "val_f1", "test_f1",
                       "n_iter", "n_epochs", "final_train_loss", "final_val_loss", "config_label")}
    print(f"选中模型: {name}  split={row['split']}  version={row['version']}  超参={hp}")
    print(f"  验证集 acc={row['val_acc']:.4f}  f1={row['val_f1']:.4f}")

    pred = fit_predict(name, row)
    out = OUT_DIR / f"predictions_{name}.csv"
    pd.Series(pred).to_csv(out, index=False, header=False)
    print(f"已保存 {len(pred)} 条预测 -> {out}")


if __name__ == "__main__":
    main()
