"""
tfidf_explore.py
================
TF-IDF 向量化探索脚本
第 1 步：先只跑通 TF-IDF，看词表、维度、稀疏度、长什么样。
不喂模型，只探索。

用法：
    python tfidf_explore.py
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 70)
print("Step 1: 加载数据")
print("=" * 70)

# 我们先聚焦 "subject+body" 版本（信息更全）
train_df = pd.read_csv(ROOT / "data-processed" / "split_A" / "train_subject_body.csv")
val_df   = pd.read_csv(ROOT / "data-processed" / "split_A" / "val_subject_body.csv")

X_train_text = train_df["text"].tolist()
y_train      = train_df["target"].tolist()
X_val_text   = val_df["text"].tolist()
y_val        = val_df["target"].tolist()

print(f"训练集文档数: {len(X_train_text)}")
print(f"验证集文档数: {len(X_val_text)}")
print(f"训练集类别分布: {dict(pd.Series(y_train).value_counts().sort_index())}")

# ============================================================
# 2. 建词表（fit） + 向量化（transform）
# ============================================================
print()
print("=" * 70)
print("Step 2: TF-IDF 向量化（带词表过滤）")
print("=" * 70)

vectorizer = TfidfVectorizer(
    lowercase=True,          # 默认 True，转小写
    token_pattern=r"(?u)\b\w\w+\b",  # 默认按字母数字分词
    
    # ===== 泛化能力过滤 =====
    stop_words='english',   # 剔除英语停用词
    min_df=2,                # 剔除只出现 1 次的词（太罕见，无泛化能力）
    max_df=0.95,             # 剔除出现在 95%+ 文档的词（太普遍，无区分度）
    
    # ===== 其他可选参数 =====
    # ngram_range = (1, 1),  # 默认只取单词，可扩展到 (1,2) 二元词组
    # sublinear_tf= True,     # 用 1+log(tf) 代替原始 tf
    # norm        = 'l2',    # 默认 L2 归一化
)

X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_val_tfidf   = vectorizer.transform(X_val_text)   # 注意：val 用 transform 不重新 fit

print(f"词表过滤参数: min_df=2, max_df=0.95")
print(f"fit 完成。词表大小: {len(vectorizer.vocabulary_)}")
print(f"训练集矩阵形状: {X_train_tfidf.shape}  (文档数 × 词表大小)")
print(f"验证集矩阵形状: {X_val_tfidf.shape}")

# ============================================================
# 3. 看稀疏度
# ============================================================
print()
print("=" * 70)
print("Step 3: 稀疏度统计")
print("=" * 70)

nnz      = X_train_tfidf.nnz
total    = X_train_tfidf.shape[0] * X_train_tfidf.shape[1]
density  = nnz / total * 100

print(f"非零元素数:  {nnz:>10,}")
print(f"总元素数:    {total:>10,}")
print(f"密度:        {density:.4f}%   （百分比越低越稀疏）")
print(f"平均每篇文档的非零词数: {nnz / X_train_tfidf.shape[0]:.1f} 个")

# ============================================================
# 4. 看词表本身
# ============================================================
print()
print("=" * 70)
print("Step 4: 词表探索")
print("=" * 70)

vocab      = vectorizer.vocabulary_
inv_vocab  = {v: k for k, v in vocab.items()}

# 词表按字母排序展示前后各 10 个
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
print(f"前 10 个词（按字母）:")
for word, idx in sorted_vocab[:10]:
    print(f"  {idx:>6d}  {word}")
print(f"...")
print(f"后 10 个词（按字母）:")
for word, idx in sorted_vocab[-10:]:
    print(f"  {idx:>6d}  {word}")

# 看哪些词在训练集中"高频出现"（区分度低的词）
# 用 idf_ 属性（每个词的 IDF 值），idf 越低 = 越常见
idf_vals = vectorizer.idf_
print()
print(f"区分度最低的 15 个词（IDF 最小 → 最常见）:")
worst_idx = np.argsort(idf_vals)[:15]
N = X_train_tfidf.shape[0]
for i in worst_idx:
    # 反推 df: IDF = log((N+1)/(df+1)) + 1 → df = (N+1)/exp(IDF-1) - 1
    df = (N + 1) / np.exp(idf_vals[i] - 1) - 1
    pct = df / N * 100
    print(f"  {inv_vocab[i]:20s}  IDF = {idf_vals[i]:.4f}  出现文档占比 = {pct:.1f}%")

# 简化展示
print()
print(f"区分度最高的 15 个词（IDF 最大 → 最罕见）:")
best_idx = np.argsort(idf_vals)[::-1][:15]
for i in best_idx:
    # IDF = log((N+1)/(df+1)) + 1, 所以 df = (N+1)/exp(IDF-1) - 1
    N = X_train_tfidf.shape[0]
    df = (N + 1) / np.exp(idf_vals[i] - 1) - 1
    print(f"  {inv_vocab[i]:20s}  IDF = {idf_vals[i]:.4f}  出现文档数 ≈ {df:.0f}")

# ============================================================
# 5. 看一篇文档的 TF-IDF 向量
# ============================================================
print()
print("=" * 70)
print("Step 5: 看一篇文档的真实向量")
print("=" * 70)

# 看类别 7（硬件类）的某篇文档
mask = train_df["target"] == 7
sample_idx = train_df[mask].index[0]
sample_text = train_df.loc[sample_idx, "text"]
sample_vec  = X_train_tfidf[sample_idx]

print(f"样本类别: 7（硬件类）")
print(f"原文前 200 字: {sample_text[:200]!r}")
print()
print(f"该文档的非零词数: {sample_vec.nnz}")
print(f"该文档向量（L2归一化后）的 L2 范数: {sparse.linalg.norm(sample_vec):.4f}  （应为 1.0）")
print()
print("该文档 TF-IDF 最高的 20 个词:")
col_data = sample_vec.toarray()[0]
top20 = np.argsort(col_data)[::-1][:20]
for i in top20:
    if col_data[i] > 0:
        print(f"  {inv_vocab[i]:20s}  TF-IDF = {col_data[i]:.4f}")

# ============================================================
# 6. 看两个不同类别文档的"区分性"
# ============================================================
print()
print("=" * 70)
print("Step 6: 两个不同类别文档的 TF-IDF 对比")
print("=" * 70)

idx_7 = train_df[train_df["target"] == 7].index[0]
idx_3 = train_df[train_df["target"] == 3].index[0]

vec_7 = X_train_tfidf[idx_7].toarray()[0]
vec_3 = X_train_tfidf[idx_3].toarray()[0]

print(f"类别 7 文档（硬件）TF-IDF 最高的 10 个词:")
top_7 = np.argsort(vec_7)[::-1][:10]
for i in top_7:
    if vec_7[i] > 0:
        print(f"  {inv_vocab[i]:20s}  TF-IDF = {vec_7[i]:.4f}")

print()
print(f"类别 3 文档（汽车）TF-IDF 最高的 10 个词:")
top_3 = np.argsort(vec_3)[::-1][:10]
for i in top_3:
    if vec_3[i] > 0:
        print(f"  {inv_vocab[i]:20s}  TF-IDF = {vec_3[i]:.4f}")

# 看两类有没有"重叠关键词"
top_7_words = {inv_vocab[i] for i in top_7 if vec_7[i] > 0}
top_3_words = {inv_vocab[i] for i in top_3 if vec_3[i] > 0}
overlap = top_7_words & top_3_words
print()
print(f"两类 top10 重叠词数: {len(overlap)} 个 → {overlap if overlap else '无'}")

print()
print("=" * 70)
print("✅ 探索完成。下一步：把这些数据喂给 SVM / MLP")
print("=" * 70)
