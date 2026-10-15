"""
predict_from_models.py
=====================
加载 models/ 下已保存的模型，对 test_data_unlabeled.csv (即 split_A 的 unlabeled) 进行预测。

models/ 目录应包含:
  NB_best.joblib, LR_best.joblib, SVM_best.joblib,
  DT_best.joblib, MLP_best.joblib

用法:
  python scripts/baselines/predict_from_models.py              # 自动选验证集最优模型
  python scripts/baselines/predict_from_models.py --model svm   # 指定模型
  python scripts/baselines/predict_from_models.py --all         # 用所有模型分别预测
"""

import argparse
import warnings
from pathlib import Path

import joblib
import pandas as pd

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data-processed"
MODEL_DIR = ROOT / "models"
OUT_DIR   = ROOT / "results" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# unlabeled 数据使用 split_A 版本（与原始 predictions.csv 对应）
UNLABELED_PATH = DATA_ROOT / "split_A" / "unlabeled_raw.csv"
# 原始 predictions.csv 模板
PRED_TEMPLATE  = ROOT / "data-Project1" / "predictions.csv"


def load_unlabeled(name=None):
    """返回 unlabeled 数据 (text 列)。

    按模型训练时使用的 split/version 读取对应预处理版本，保证输入格式与训练一致。
    """
    if name is not None:
        cfg = pd.read_csv(MODEL_DIR / "best_config.csv")
        row = cfg[cfg["model"] == name.upper()]
        if not row.empty:
            split, version = row.iloc[0]["split"], row.iloc[0]["version"]
            path = DATA_ROOT / f"split_{split}" / f"unlabeled_{version}.csv"
            print(f"  使用数据: {path.relative_to(ROOT)}")
            return pd.read_csv(path)
    return pd.read_csv(UNLABELED_PATH)


def load_model(name):
    """加载 (vec, clf) 元组。"""
    vec, clf = joblib.load(MODEL_DIR / f"{name.upper()}_best.joblib")
    return vec, clf


def predict_single(name):
    """用指定模型预测，保存 predictions_{name}.csv，返回预测标签数组。"""
    print(f"\n[{name}] 加载模型...")
    vec, clf = load_model(name)

    unl = load_unlabeled(name)
    X_unl = vec.transform(unl["text"].astype(str).tolist())
    pred = clf.predict(X_unl)

    out = OUT_DIR / f"predictions_{name}.csv"
    pd.Series(pred).to_csv(out, index=False, header=False)
    print(f"  预测完成 {len(pred)} 条，保存 -> {out.relative_to(ROOT)}")
    return pred


def best_model():
    """按验证集 val_acc 选最优模型。"""
    cfg = pd.read_csv(MODEL_DIR / "best_config.csv")
    cfg = cfg.sort_values("val_acc", ascending=False)
    return cfg.iloc[0]["model"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="auto",
                    choices=["auto", "all"] + [m.lower() for m in
                        ["NB", "LR", "SVM", "DT", "MLP"]])
    ap.add_argument("--output", default=None,
                    help="最终 predictions.csv 保存路径 (默认为 results/predictions/predictions.csv)")
    args = ap.parse_args()

    if args.model == "auto":
        name = best_model()
        print(f"自动选中模型: {name} (验证集最优)")
        predict_single(name)
        # 复制为 predictions.csv
        final_out = Path(args.output) if args.output else OUT_DIR / "predictions.csv"
        pd.read_csv(OUT_DIR / f"predictions_{name}.csv", header=None).to_csv(
            final_out, index=False, header=False)
        print(f"最终文件 -> {final_out}")

    elif args.model == "all":
        cfg = pd.read_csv(MODEL_DIR / "best_config.csv")
        cfg = cfg.sort_values("val_acc", ascending=False)
        print("=" * 50)
        print("所有模型预测结果:")
        print(cfg[["model", "split", "version", "val_acc", "test_acc"]].to_string(index=False))
        print("=" * 50)
        for name in cfg["model"]:
            predict_single(name)

    else:
        name = args.model.upper()
        predict_single(name)
        final_out = Path(args.output) if args.output else OUT_DIR / "predictions.csv"
        pd.read_csv(OUT_DIR / f"predictions_{name}.csv", header=None).to_csv(
            final_out, index=False, header=False)
        print(f"最终文件 -> {final_out}")


if __name__ == "__main__":
    main()
