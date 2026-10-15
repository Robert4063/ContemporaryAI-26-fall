"""
train_save_models.py
====================
为每个模型找到验证集最优配置，然后在 train+val 上重新训练并保存模型。

输出:
  models/NB_best.joblib    — (vectorizer, clf) 元组
  models/LR_best.joblib
  models/SVM_best.joblib
  models/DT_best.joblib
  models/MLP_best.joblib
  models/best_config.csv  — 各模型最优配置汇总

用法 (分终端运行，或一起跑):
  python scripts/baselines/train_save_models.py --model nb
  python scripts/baselines/train_save_models.py --model lr
  python scripts/baselines/train_save_models.py --model svm
  python scripts/baselines/train_save_models.py --model dt
  python scripts/baselines/train_save_models.py --model mlp
  python scripts/baselines/train_save_models.py --all   # 跑全部
"""

import argparse
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data-processed"
BLINE_DIR = ROOT / "results" / "baseline"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── 候选模型 ──────────────────────────────────────────────
MODELS = {
    "nb":  dict(cls=MultinomialNB,          col_alpha="alpha"),
    "lr":  dict(cls=LogisticRegression,     col_c="C"),
    "svm": dict(cls=SGDClassifier,           col_c="C"),
    "dt":  dict(cls=DecisionTreeClassifier, col_depth="max_depth", col_leaf="min_samples_leaf"),
    "mlp": dict(cls=MLPClassifier,          col_hidden="hidden_layer_sizes", col_alpha="alpha"),
}


def best_config(name):
    """从 baseline CSV 中找 val_acc 最高的配置 (并列取 val_f1 更高者)。"""
    df = pd.read_csv(BLINE_DIR / f"{name.upper()}_all_splits_all_versions.csv")
    df = df.sort_values(["val_acc", "val_f1"], ascending=False)
    return df.iloc[0]


def build_clf(name, row):
    """按配置行构造 sklearn 模型，保持与 train_*.py 一致。"""
    if name == "nb":
        return MultinomialNB(alpha=float(row["alpha"]))
    if name == "lr":
        # warm_start 分段: 总共 300 次迭代, tol=0
        return LogisticRegression(C=float(row["C"]), solver="lbfgs",
                                  max_iter=300, tol=0.0, random_state=42)
    if name == "svm":
        return SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=1.0 / float(row["C"]),
            max_iter=300,
            tol=1e-4,
            random_state=42,
            learning_rate="optimal",
        )
    if name == "dt":
        depth = None if str(row["max_depth"]) in ("None", "nan") else int(float(row["max_depth"]))
        return DecisionTreeClassifier(max_depth=depth,
                                      min_samples_leaf=int(row["min_samples_leaf"]),
                                      random_state=42)
    if name == "mlp":
        hidden = tuple(int(x) for x in
                       str(row["hidden_layer_sizes"]).strip("()").split(",") if x.strip())
        return MLPClassifier(
            hidden_layer_sizes=hidden, alpha=float(row["alpha"]),
            max_iter=300, tol=0.0, n_iter_no_change=301,
            early_stopping=False, random_state=42,
        )
    raise ValueError(name)


def train_and_save(name):
    row    = best_config(name)
    split, version = row["split"], row["version"]

    print(f"\n{'='*60}")
    print(f"[{name.upper()}] 最优配置:")
    print(f"  split={split}  version={version}")
    # 打印关键超参
    for k, v in row.items():
        if k not in ("split", "version", "train_acc", "val_acc", "test_acc",
                     "val_f1", "test_f1", "n_iter", "n_epochs",
                     "final_train_loss", "final_val_loss", "config_label"):
            print(f"  {k}={v}")
    print(f"  val_acc={row['val_acc']:.4f}  test_acc={row['test_acc']:.4f}")

    # ── 加载 train + val (不用 test) ──────────────────────
    base  = DATA_ROOT / f"split_{split}"
    train = pd.read_csv(base / f"train_{version}.csv")
    val   = pd.read_csv(base / f"val_{version}.csv")
    full  = pd.concat([train, val], ignore_index=True)

    X = full["text"].astype(str).tolist()
    y = full["target"].values

    vec = TfidfVectorizer(min_df=2, max_df=0.95, max_features=20000)
    X_vec = vec.fit_transform(X)

    clf = build_clf(name, row)
    print(f"  训练 {name.upper()} (train+val 共 {len(full)} 条)...")
    clf.fit(X_vec, y)
    print(f"  训练完成 (n_iter={getattr(clf,'n_iter_',getattr(clf,'n_iter',len(full)))}).")

    # ── 保存 (vec, clf) 元组 ──────────────────────────────
    out_path = MODEL_DIR / f"{name.upper()}_best.joblib"
    joblib.dump((vec, clf), out_path)
    print(f"  已保存 -> {out_path.relative_to(ROOT)}")

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all",
                    choices=list(MODELS.keys()) + ["all"])
    args = ap.parse_args()

    models = MODELS if args.model == "all" else {args.model: MODELS[args.model]}
    configs = {}

    for name in models:
        row = train_and_save(name)
        configs[name.upper()] = {
            "split": row["split"],
            "version": row["version"],
            "val_acc": row["val_acc"],
            "test_acc": row["test_acc"],
            "val_f1": row["val_f1"],
            "test_f1": row["test_f1"],
        }

    # ── 每个模型单独写一份配置 (多终端并行时互不覆盖), 再合并汇总 ──
    for name, cfg in configs.items():
        pd.DataFrame([{"model": name, **cfg}]).to_csv(
            MODEL_DIR / f"{name}_config.csv", index=False)
    parts = [pd.read_csv(p) for p in sorted(MODEL_DIR.glob("*_config.csv"))
             if p.name != "best_config.csv"]
    cfg_df = pd.concat(parts, ignore_index=True)
    cfg_path = MODEL_DIR / "best_config.csv"
    cfg_df.to_csv(cfg_path, index=False)
    print(f"\n最优配置汇总已保存 -> {cfg_path.relative_to(ROOT)}")
    print(cfg_df[["model", "split", "version", "val_acc", "test_acc"]].to_string(index=False))

    print(f"\n{'='*60}")
    print("所有模型已保存！")
    print(f"下一步: python scripts/baselines/predict_from_models.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
