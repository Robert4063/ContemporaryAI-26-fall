"""
author_analysis.py
==================
探索作者签名和类别的关联强度。

思路：
  1. 从 train_subject_body 文本中提取作者名（From: 行）
  2. 统计每个作者写了哪些类别的文章
  3. 看"作者 → 类别"有多强相关
  4. 对照：如果用作者做特征，理论上准确率能到多少？
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 70)
print("Step 1: 加载数据并提取作者")
print("=" * 70)

df = pd.read_csv("data-processed/split_A/train_subject_body.csv")
# 用原始文本提取 From: 行
raw_df = pd.read_csv("data-Project1/train_data.csv")

def extract_author_from_raw(text: str) -> str:
    """从原始文本的 From: 行提取作者名"""
    if not isinstance(text, str):
        return "UNKNOWN"
    match = re.match(r"From:\s*(.+?)(?:\s*<|$)", text, re.IGNORECASE)
    if match:
        raw = match.group(1).strip()
        # 去掉 <email> 部分，只保留名字
        raw = re.sub(r"<[^>]+>", "", raw).strip()
        # 去掉括号内容（组织机构）
        raw = re.sub(r"\([^)]*\)", "", raw).strip()
        raw = raw.strip()
        if raw:
            return raw
    return "UNKNOWN"

df["author"] = raw_df["text"].apply(extract_author_from_raw)

print(f"总文档数: {len(df)}")
print(f"不同作者数: {df['author'].nunique()}")
print(f"平均每作者文章数: {len(df) / df['author'].nunique():.1f}")

# ============================================================
# 2. 作者在各类别的分布
# ============================================================
print()
print("=" * 70)
print("Step 2: 作者 × 类别 交叉统计")
print("=" * 70)

author_class_matrix = pd.crosstab(df["author"], df["target"])

# 看作者的类别集中度
def entropy(row):
    """计算分布的熵，越低 = 越集中在某一个类别"""
    probs = row[row > 0] / row.sum()
    return -np.sum(probs * np.log(probs))

author_class_matrix["entropy"] = author_class_matrix.apply(entropy, axis=1)
author_class_matrix["total"]  = author_class_matrix.iloc[:, :10].sum(axis=1)
author_class_matrix["dominant_class"] = author_class_matrix.iloc[:, :10].idxmax(axis=1)

# 只看写了 3 篇以上的作者
multi_authors = author_class_matrix[author_class_matrix["total"] >= 3].sort_values("total", ascending=False)

print(f"写了 3 篇以上的作者数: {len(multi_authors)}")
print(f"\n前 20 位最多产作者（及其主要类别、熵）:")
print(f"  {'作者名':30s}  文章  熵     主要类别  各类别数量")
print(f"  {'-'*75}")
for idx, row in multi_authors.head(20).iterrows():
    cats = row.iloc[:10]
    cat_str = ", ".join([f"{c}:{int(cats[c])}" for c in range(10) if cats[c] > 0])
    print(f"  {idx:30s}  {int(row['total']):3d}   {row['entropy']:.2f}   类别{int(row['dominant_class'])}  {cat_str}")

# ============================================================
# 3. 强关联作者统计
# ============================================================
print()
print("=" * 70)
print("Step 3: 强关联作者（作者=某类别的强信号）")
print("=" * 70)

# 如果一个作者 100% 只写某一个类别，这就是完美信号
multi_authors["purity"] = multi_authors.iloc[:, :10].max(axis=1) / multi_authors["total"]
pure_authors = multi_authors[multi_authors["purity"] == 1.0].sort_values("total", ascending=False)

print(f"只写单一类别的作者数: {len(pure_authors)}")
print(f"这些作者覆盖的文章数: {pure_authors['total'].sum()}")
print(f"\n高纯度作者（只写一个类别，文章数>=3）:")
for idx, row in pure_authors[pure_authors["total"] >= 3].iterrows():
    dc = int(row["dominant_class"])
    print(f"  {idx:30s}  → 类别{dc}  ({int(row['total'])} 篇)")

# ============================================================
# 4. 用作者预测类别：理论上限
# ============================================================
print()
print("=" * 70)
print("Step 4: 如果只用「作者 → 类别」映射预测，准确率能到多少？")
print("=" * 70)

# 策略：训练集记住每个作者最常写的类别，测试集套用
# 在训练集上做 leave-one-out 估计泛化准确率

from collections import Counter

# 每个作者的 dominant class
author_to_class = {}
for author, grp in df.groupby("author"):
    if len(grp) >= 1:
        counter = Counter(grp["target"])
        dominant = counter.most_common(1)[0][0]
        author_to_class[author] = dominant

# 训练集本身准确率
train_preds = df["author"].map(author_to_class)
train_acc = (train_preds == df["target"]).mean()

print(f"训练集准确率（用作者名查表）: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"  → 这是在训练集上「背答案」的分数，不代表真实能力")
print()

# 如果去掉这些"唯一作者"的贡献呢？
unique_author_docs = df[df["author"].apply(lambda a: a in pure_authors.index)]
other_docs = df[~df["author"].isin(pure_authors.index)]

print(f"「强作者」文档（只写单一类别）: {len(unique_author_docs)} 篇 ({len(unique_author_docs)/len(df)*100:.1f}%)")
print(f"「弱作者」文档（跨类别或极少文章）: {len(other_docs)} 篇 ({len(other_docs)/len(df)*100:.1f}%)")

if len(other_docs) > 0:
    other_author_preds = other_docs["author"].map(author_to_class)
    other_acc = (other_author_preds == other_docs["target"]).mean()
    print(f"「弱作者」子集准确率: {other_acc:.4f} ({other_acc*100:.2f}%)")

# ============================================================
# 5. 结论：作者签名是否可作为强特征？
# ============================================================
print()
print("=" * 70)
print("Step 5: 诊断结论")
print("=" * 70)

avg_entropy = multi_authors["entropy"].mean()
avg_purity  = multi_authors["purity"].mean()

print(f"多产作者平均熵: {avg_entropy:.3f}  （0=完全专一, 2.3=10类均匀）")
print(f"多产作者平均纯度: {avg_purity:.3f}  （1.0=100%只写一类）")
print()

# 估计用作者特征的准确率上限（不考虑泛化）
theoretical_max = (pure_authors["total"].sum() + 
                   multi_authors[multi_authors["purity"] < 1]["dominant_class"].apply(
                       lambda c: multi_authors.loc[multi_authors[multi_authors["purity"] < 1].index, c] if c in multi_authors.columns else 0).sum()) / len(df)

print(f"结论:")
print(f"  作者签名 ≈ {train_acc*100:.1f}% 训练集准确率的「作弊特征」")
print(f"  但对 {pure_authors['total'].sum()} 篇 ({pure_authors['total'].sum()/len(df)*100:.1f}%) 文档来说，它确实是「完美信号」")
print(f"  对其余 {len(other_docs)} 篇文档，跨类别的作者无法提供可靠信号")
print()
print(f"  建议：保留作者作为特征之一，但同时需要真正的文本特征（如 TF-IDF）")
print(f"        让模型自己决定给作者信息多少权重")
