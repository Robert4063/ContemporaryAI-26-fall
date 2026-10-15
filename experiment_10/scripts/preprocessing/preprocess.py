"""
preprocess.py
=============
从原始数据 `data-Project1/train_data.csv` 出发,按指定比例划分数据集,
每个划分生成 4 种文本版本,保存到 `data-processed/split_X/` 目录。

划分比例:
  split_A : train 80% / val 10% / test 10%
  split_B : train 85% / val 5%  / test 10%
  split_C : train 90% / val 5%  / test 5%

文本版本 (每种划分均生成):
  raw              : 原始文本,仅按 train/val/test 切分,不做任何清洗
  body_only        : 清洗邮件头,只保留正文 (clean body)
  subject_body     : 标题 + 正文
  subject_body_author : 标题 + 正文 + 作者名前缀

用法:
  python scripts/preprocessing/preprocess.py

输出 (每个 split_X 目录各 16 个文件):
  train_raw.csv              train_body_only.csv
  train_subject_body.csv     train_subject_body_author.csv
  val_raw.csv                val_body_only.csv
  val_subject_body.csv       val_subject_body_author.csv
  test_raw.csv               test_body_only.csv
  test_subject_body.csv      test_subject_body_author.csv
  unlabeled_raw.csv          unlabeled_body_only.csv
  unlabeled_subject_body.csv unlabeled_subject_body_author.csv
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# ============================================================
# 0. 配置
# ============================================================
ROOT       = Path(__file__).resolve().parents[2]
DATA_DIR   = ROOT / "data-Project1"
OUT_ROOT   = ROOT / "data-processed"
SEED       = 42
SPLITS     = {
    "A": {"train": 0.80, "val": 0.10, "test": 0.10},  # 8:1:1
    "B": {"train": 0.85, "val": 0.05, "test": 0.10},  # 85:5:10
    "C": {"train": 0.90, "val": 0.05, "test": 0.05},  # 90:5:5
}

np.random.seed(SEED)

# ============================================================
# 1. 加载原始数据 & 提取 author
# ============================================================
print("=" * 60)
print("Step 1: 加载数据 & 提取作者")
print("=" * 60)

train_raw_df = pd.read_csv(DATA_DIR / "train_data.csv")
test_raw_df  = pd.read_csv(DATA_DIR / "test_data_unlabeled.csv")

print(f"训练集: {len(train_raw_df)} 条, 类别数: {train_raw_df['target'].nunique()}")
print(f"无标签测试集: {len(test_raw_df)} 条")

def extract_author(text: str) -> str:
    if not isinstance(text, str):
        return "UNKNOWN"
    for line in text.split("\n"):
        if line.lower().startswith("from:"):
            content = line[5:].strip()
            m = re.match(r"^([^<]+)\s*<", content)
            if m:
                return m.group(1).strip()[:50]
            m = re.match(r"^<[^>]+>\s*\(([^)]+)\)", content)
            if m:
                return m.group(1).strip()[:50]
            return content.strip()[:50] if "@" not in content else content.strip()[:50]
    return "UNKNOWN"

# ============================================================
# 2. 邮件头清洗函数
# ============================================================
print("\n" + "=" * 60)
print("Step 2: 邮件头清洗函数")
print("=" * 60)

HEADER_LINE_PATTERNS = [
    r"^From:", r"^Reply-To:", r"^Originator:",
    r"^Subject:", r"^Organization:", r"^Lines:",
    r"^NNTP-Posting-Host:", r"^Nntp-Posting-Host:",
    r"^X-Newsreader:", r"^X-Trace:", r"^Xref:",
    r"^Date:", r"^Newsgroups:", r"^Message-ID:",
    r"^References:", r"^Keywords:", r"^Summary:",
    r"^Expires:", r"^Followup-To:", r"^Sender:",
    r"^Distribution:", r"^In-Reply-To:",
    r"^User-Agent:", r"^MIME-Version:",
    r"^Content-Type:", r"^Content-Transfer-Encoding:",
    r"^X-",
    r"^\S+@\S+\.\S+$",
    r"^\d+\.\d+\.\d+\.\d+$",
]
HEADER_RE = re.compile("|".join(HEADER_LINE_PATTERNS), re.MULTILINE | re.IGNORECASE)

def extract_subject(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    for line in raw.split("\n"):
        if line.strip().startswith("Subject:"):
            return line.strip()[len("Subject:"):].strip()
    return ""

def clean_body(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    lines = []
    for line in raw.split("\n"):
        s = line.strip()
        if not s or HEADER_RE.search(s):
            continue
        lines.append(s)
    return " ".join(lines)

def make_raw(text: str) -> str:
    """直接返回原始文本,不做任何处理"""
    return text if isinstance(text, str) else ""

def make_body_only(text: str) -> str:
    return clean_body(text)

def make_subject_body(text: str) -> str:
    subject = extract_subject(text)
    body    = clean_body(text)
    return (subject + " " + body).strip() if subject else body

def make_subject_body_author(text: str) -> str:
    author  = extract_author(text)
    subject = extract_subject(text)
    body    = clean_body(text)
    parts   = []
    if author:
        parts.append(f"From {author}")
    if subject:
        parts.append(subject)
    parts.append(body)
    return " | ".join(parts).strip()

VERSIONS = {
    "raw":                 make_raw,
    "body_only":           make_body_only,
    "subject_body":        make_subject_body,
    "subject_body_author":  make_subject_body_author,
}

# ============================================================
# 3. 划分 & 保存
# ============================================================
print("\n" + "=" * 60)
print("Step 3: 划分并保存所有数据集")
print("=" * 60)

from sklearn.model_selection import train_test_split

X_all     = train_raw_df["text"].values
y_all     = train_raw_df["target"].values
authors   = train_raw_df["text"].apply(extract_author).values

# 无标签测试集只有 text,不需要划分
X_unlabeled = test_raw_df["text"].values

# ---------- 保存全局元信息 ----------
OUT_ROOT.mkdir(exist_ok=True)
meta_rows = []

for split_name, ratios in SPLITS.items():
    print(f"\n--- split_{split_name} ({ratios['train']*100:.0f}:{ratios['val']*100:.0f}:{ratios['test']*100:.0f}) ---")

    out_dir = OUT_ROOT / f"split_{split_name}"
    out_dir.mkdir(exist_ok=True)

    # Step 3a: 先切 test
    test_ratio   = ratios["test"]
    val_ratio_of_remain = ratios["val"] / (1 - test_ratio)   # 在剩余数据中 val 的比例

    X_remain, X_test, y_remain, y_test = train_test_split(
        X_all, y_all,
        test_size=test_ratio,
        random_state=SEED,
        stratify=y_all,
    )

    # Step 3b: 再切 val
    X_train, X_val, y_train, y_val = train_test_split(
        X_remain, y_remain,
        test_size=val_ratio_of_remain,
        random_state=SEED,
        stratify=y_remain,
    )

    print(f"  train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    meta_rows.append({
        "split":           split_name,
        "train_size":      len(X_train),
        "val_size":        len(X_val),
        "test_size":       len(X_test),
        "train_ratio":     len(X_train) / len(X_all),
        "val_ratio":       len(X_val)   / len(X_all),
        "test_ratio":      len(X_test)  / len(X_all),
    })

    # ---------- 保存各版本 ----------
    def save_one(name, X_texts, y_labels=None):
        for ver, fn in VERSIONS.items():
            rows = [{"text": fn(t)} for t in X_texts]
            df   = pd.DataFrame(rows)
            if y_labels is not None:
                df["target"] = y_labels
            path = out_dir / f"{name}_{ver}.csv"
            df.to_csv(path, index=False)
            print(f"    {name}_{ver}.csv  ({len(df)} rows)")

    save_one("train", X_train, y_train)
    save_one("val",   X_val,   y_val)
    save_one("test",  X_test,  y_test)
    save_one("unlabeled", X_unlabeled)   # 无标签,无 target

# ---------- 保存全局 meta ----------
meta_df = pd.DataFrame(meta_rows)
meta_df.to_csv(OUT_ROOT / "split_meta.csv", index=False)
print(f"\n元信息已保存: {OUT_ROOT / 'split_meta.csv'}")
print(meta_df.to_string(index=False))

print("\n" + "=" * 60)
print("完成！")
for sn in SPLITS:
    print(f"  data-processed/split_{sn}/  (4 versions × 4 sets = 16 files)")
print("=" * 60)
