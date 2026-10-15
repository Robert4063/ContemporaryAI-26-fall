# -*- coding: utf-8 -*-
"""
数据加载、分层划分、四种文本版本的构造、TF-IDF 向量化。

约定：
- 划分只依赖标签 y 和随机种子，因此同一划分下四种文本版本使用完全相同的样本。
- 预处理是逐样本的规则（不依赖其他样本），不存在跨集合的统计量泄露；
  TF-IDF 的词表和 IDF 只在训练集上拟合。
"""

import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

# (train, val, test) 比例
SPLITS: Dict[str, Tuple[float, float, float]] = {
    '80_10_10': (0.80, 0.10, 0.10),
    '85_5_10': (0.85, 0.05, 0.10),
    '90_5_5': (0.90, 0.05, 0.05),
}

PREPROCESS_METHODS = ['raw', 'body_only', 'subject_body', 'subject_body_author']

TFIDF_PARAMS = dict(min_df=2, max_df=0.95, max_features=20000)

# 已知的新闻组头字段（小写）。未列出的头字段会被保留，这是有意为之：
# 方案只移除“已知”头字段，未知头字段、引用和签名可能残留，需要在结果中讨论。
KNOWN_HEADER_FIELDS = {
    'from', 'subject', 'organization', 'lines', 'nntp-posting-host',
    'reply-to', 'distribution', 'keywords', 'x-newsreader', 'article-i.d.',
    'summary', 'originator', 'in-reply-to', 'news-software', 'expires',
    'x-disclaimer', 'disclaimer', 'x-mailer', 'nf-id', 'nf-from',
    'x-posted-from', 'content-type', 'mime-version', 'x-xxdate',
    'content-transfer-encoding', 'x-useragent', 'followup-to', 'references',
    'sender', 'date', 'message-id', 'path', 'newsgroups',
}

_HEADER_RE = re.compile(r'^([A-Za-z][A-Za-z0-9\-\.]*):[ \t]*(.*)$')


def parse_message(text: str) -> Tuple[Dict[str, str], List[str], str]:
    """
    按首个空行把文本切成“头字段块 / 正文”。
    返回 (已知头字段 dict[小写字段名->值], 未知头字段行列表, 正文)。
    没有空行分隔时，整段文本视为正文。
    """
    text = text.replace('\r\n', '\n')
    head, sep, body = text.partition('\n\n')
    if not sep:
        return {}, [], text

    fields: Dict[str, str] = {}
    kept: List[str] = []
    last_known = None  # 上一行所属的已知字段名（用于处理续行）
    for line in head.split('\n'):
        if line[:1] in (' ', '\t'):            # 续行
            if last_known is not None:
                fields[last_known] = fields[last_known] + ' ' + line.strip()
            else:
                kept.append(line)
            continue
        m = _HEADER_RE.match(line)
        if m and m.group(1).lower() in KNOWN_HEADER_FIELDS:
            name = m.group(1).lower()
            fields.setdefault(name, m.group(2).strip())
            last_known = name if name in fields else None
        else:
            kept.append(line)
            last_known = None
    return fields, kept, body


def _join(parts: List[str]) -> str:
    return '\n'.join(p for p in parts if p).strip()


def make_raw(text: str) -> str:
    """raw：保留原文。"""
    return text


def make_body_only(text: str) -> str:
    """body_only：移除已知头字段，保留正文（未知头字段会残留）。"""
    _, kept, body = parse_message(text)
    return _join(kept + [body])


def make_subject_body(text: str) -> str:
    """subject_body：标题 + 正文。"""
    fields, kept, body = parse_message(text)
    return _join([fields.get('subject', '')] + kept + [body])


def make_subject_body_author(text: str) -> str:
    """subject_body_author：作者(From 字段原值) + 标题 + 正文。"""
    fields, kept, body = parse_message(text)
    return _join([fields.get('from', ''), fields.get('subject', '')] + kept + [body])


PREPROCESSORS = {
    'raw': make_raw,
    'body_only': make_body_only,
    'subject_body': make_subject_body,
    'subject_body_author': make_subject_body_author,
}


def preprocess_texts(texts: List[str], method: str) -> List[str]:
    if method not in PREPROCESSORS:
        raise ValueError(f'Unknown preprocessing method: {method}')
    fn = PREPROCESSORS[method]
    return [fn(t) for t in texts]


def load_labeled(path: str) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(path)
    return df['text'].astype(str).tolist(), df['target'].values


def load_unlabeled(path: str) -> List[str]:
    return pd.read_csv(path)['text'].astype(str).tolist()


def stratified_split_indices(y: np.ndarray, ratios: Tuple[float, float, float],
                             seed: int = RANDOM_SEED):
    """
    两步分层划分，返回 (idx_train, idx_val, idx_test)。
    第一步从全集中切出 test（占全集 test_ratio），
    第二步从剩余部分中切出 val（占剩余部分 val/(train+val)）。
    """
    train_r, val_r, test_r = ratios
    assert abs(train_r + val_r + test_r - 1.0) < 1e-9, 'ratios must sum to 1'
    idx = np.arange(len(y))
    idx_trval, idx_test = train_test_split(
        idx, test_size=test_r, stratify=y, random_state=seed)
    idx_train, idx_val = train_test_split(
        idx_trval, test_size=val_r / (train_r + val_r),
        stratify=y[idx_trval], random_state=seed)
    return np.sort(idx_train), np.sort(idx_val), np.sort(idx_test)


def vectorize(train_texts, val_texts, test_texts, extra_texts=None):
    """TF-IDF：词表与 IDF 仅在训练集上拟合。"""
    vec = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = vec.fit_transform(train_texts)
    X_val = vec.transform(val_texts)
    X_test = vec.transform(test_texts)
    return X_train, X_val, X_test, vec


def text_statistics(texts: List[str]) -> Dict[str, float]:
    lengths = np.array([len(t) for t in texts])
    return {
        'num_samples': len(texts),
        'avg_length': float(lengths.mean()),
        'median_length': float(np.median(lengths)),
        'min_length': int(lengths.min()),
        'max_length': int(lengths.max()),
    }
