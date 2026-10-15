"""检查训练集和测试集的作者重叠情况"""
import pandas as pd, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 加载
train_raw = pd.read_csv(ROOT / "data-Project1" / "train_data.csv")
test_raw  = pd.read_csv(ROOT / "data-Project1" / "test_data_unlabeled.csv")

train_authors = set(train_raw['text'].apply(extract_author))
test_authors  = set(test_raw['text'].apply(extract_author))

overlap = train_authors & test_authors
only_train = train_authors - test_authors
only_test  = test_authors - train_authors

print(f"训练集不同作者: {len(train_authors)}")
print(f"测试集不同作者: {len(test_authors)}")
print(f"重叠作者数:     {len(overlap)}")
print(f"仅训练集有:     {len(only_train)}")
print(f"仅测试集有:     {len(only_test)}")
print(f"测试集作者覆盖率: {len(overlap) / len(test_authors) * 100:.1f}%")
print()

# 测试集文档中，有多少来自已知作者
test_author_col = test_raw['text'].apply(extract_author)
known_docs = sum(test_author_col.isin(train_authors))
unknown_docs = sum(~test_author_col.isin(train_authors))
print(f"测试集文档:")
print(f"  来自已知作者（训练集见过）: {known_docs} ({known_docs/len(test_raw)*100:.1f}%)")
print(f"  来自未知作者（训练集没见过）: {unknown_docs} ({unknown_docs/len(test_raw)*100:.1f}%)")
print()

# 如果用训练集的 "作者→类别" 映射来预测测试集
train_raw['author'] = train_raw['text'].apply(extract_author)
author_class_map = {}
for author, grp in train_raw.groupby('author'):
    counter = grp['target'].value_counts()
    dominant = counter.idxmax()
    author_class_map[author] = dominant

test_preds = test_author_col.map(author_class_map)
print(f"=== 训练集「作者→类别」映射直接应用到测试集 ===")
print(f"有确定预测的文档: {test_preds.notna().sum()} ({test_preds.notna().mean()*100:.1f}%)")
print(f"无法预测（未知作者）: {test_preds.isna().sum()} ({test_preds.isna().mean()*100:.1f}%)")

# 如果对未知作者用默认最常见类别
most_common_class = train_raw['target'].mode()[0]
print(f"\n对未知作者用最常见类别 {most_common_class} 填充")
print(f"测试集预测完成")
