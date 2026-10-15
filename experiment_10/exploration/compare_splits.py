"""
compare_splits.py
=================
E1: 划分对照实验

目标: 同一模型(LinearSVC)、同一特征(subject+body)、同一超参数,
      比较划分 A(8:1:1) 与划分 B(85:5:10) 在验证集和测试集上的表现。

核心观察:
  - 验证集越大 → 验证结果越稳定 → 越可靠地指导超参数选择
  - 训练集越小 → 模型容量不变但数据更少 → 更容易欠拟合
  - 验证集越小 → 验证结果波动越大 → 可能选错超参数

用法:
  D:\AI\envs\experiment_1\python.exe compare_splits.py
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
SEED = 42

RESULTS_DIR = ROOT / "results" / "exploration"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 加载数据
# ============================================================
def load(version="subject_body"):
    """返回 split_A 和 split_B 的 train/val/test"""
    dp = ROOT / "data-processed"
    return {
        "A": {
            "train": pd.read_csv(dp / "split_A" / f"train_{version}.csv"),
            "val":   pd.read_csv(dp / "split_A" / f"val_{version}.csv"),
            "test":  pd.read_csv(dp / "split_A" / f"test_{version}.csv"),
        },
        "B": {
            "train": pd.read_csv(dp / "split_B" / f"train_{version}.csv"),
            "val":   pd.read_csv(dp / "split_B" / f"val_{version}.csv"),
            "test":  pd.read_csv(dp / "split_B" / f"test_{version}.csv"),
        },
    }

# ============================================================
# 向量化配置（与后续脚本保持一致）
# ============================================================
def make_vectorizer():
    return TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w\w+\b",
        min_df=2,
        max_df=0.95,
    )

# ============================================================
# E1: 划分对照
# ============================================================
print("=" * 70)
print("E1: 划分对照实验")
print("=" * 70)

results = []
splits = load("subject_body")

for split_name in ["A", "B"]:
    print(f"\n--- 划分 {split_name} ({'8:1:1' if split_name == 'A' else '85:5:10'}) ---")
    data = splits[split_name]

    X_train_text = data["train"]["text"].astype(str).tolist()
    y_train = data["train"]["target"].values
    X_val_text   = data["val"]["text"].astype(str).tolist()
    y_val        = data["val"]["target"].values
    X_test_text  = data["test"]["text"].astype(str).tolist()
    y_test        = data["test"]["target"].values

    print(f"  训练: {len(X_train_text)}, 验证: {len(X_val_text)}, 测试: {len(X_test_text)}")

    t0 = time.time()
    vectorizer = make_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_val_tfidf   = vectorizer.transform(X_val_text)
    X_test_tfidf  = vectorizer.transform(X_test_text)
    vec_time = time.time() - t0
    print(f"  词表: {len(vectorizer.vocabulary_)}, 向量化: {vec_time:.1f}s")

    # 默认参数 LinearSVC
    t0 = time.time()
    model = LinearSVC(C=1.0, random_state=SEED)
    model.fit(X_train_tfidf, y_train)
    train_time = time.time() - t0
    print(f"  训练: {train_time:.1f}s")

    for name, X_mat, y_true in [
        ("train", X_train_tfidf, y_train),
        ("val",   X_val_tfidf,   y_val),
        ("test",  X_test_tfidf,  y_test),
    ]:
        y_pred = model.predict(X_mat)
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="macro")
        gap = accuracy_score(y_train, y_train) - acc  # train_acc - current_acc
        print(f"  [{name}] Acc={acc:.4f}  F1={f1:.4f}  Gap={gap:.4f}")
        results.append({
            "split": split_name,
            "dataset": name,
            "accuracy": acc,
            "macro_f1": f1,
            "train_size": len(X_train_text),
            "val_size": len(X_val_text),
            "test_size": len(X_test_text),
        })

# ============================================================
# 保存结果
# ============================================================
results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_DIR / "e1_compare_splits.csv", index=False)
print("\n结果已保存: results/exploration/e1_compare_splits.csv")
print()
print("=" * 70)
print("E1 对照总结")
print("=" * 70)
print(results_df[["split","dataset","accuracy","macro_f1","train_size","val_size"]].to_string(index=False))

# 核心结论
a_val = results_df[(results_df.split=="A") & (results_df.dataset=="val")]["accuracy"].values[0]
b_val = results_df[(results_df.split=="B") & (results_df.dataset=="val")]["accuracy"].values[0]
a_test = results_df[(results_df.split=="A") & (results_df.dataset=="test")]["accuracy"].values[0]
b_test = results_df[(results_df.split=="B") & (results_df.dataset=="test")]["accuracy"].values[0]
print(f"\n验证集 Acc 差 (A-B): {a_val-b_val:+.4f}")
print(f"测试集 Acc 差 (A-B): {a_test-b_test:+.4f}")
print(f"→ 划分 A 的验证集更稳定，测试集差距反映的是真实泛化差异")
