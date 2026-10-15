# -*- coding: utf-8 -*-
"""
把划分和处理好的数据落盘到 experiment1/data_processed/。

目录结构：
data_processed/
  split_indices/<split>.csv          每个样本在 train_data.csv 中的行号及其所属子集
  <split>/<method>/{train,val,test}.csv    列：text, target, orig_index
  unlabeled/<method>.csv             无标签数据的同种处理结果（与划分无关）
  manifest.json                      划分比例、种子、各子集样本数、类别计数
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_preprocessing import (  # noqa: E402
    PREPROCESS_METHODS, RANDOM_SEED, SPLITS, TFIDF_PARAMS,
    load_labeled, load_unlabeled, preprocess_texts, stratified_split_indices,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, '..', 'data-Project1')
OUT_DIR = os.path.join(ROOT, 'data_processed')


def main():
    texts, y = load_labeled(os.path.join(DATA_DIR, 'train_data.csv'))
    unlabeled = load_unlabeled(os.path.join(DATA_DIR, 'test_data_unlabeled.csv'))
    os.makedirs(os.path.join(OUT_DIR, 'split_indices'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'unlabeled'), exist_ok=True)

    manifest = {'seed': RANDOM_SEED, 'tfidf_params': TFIDF_PARAMS,
                'n_labeled': len(texts), 'n_unlabeled': len(unlabeled),
                'splits': {}}

    # 预处理是逐样本规则，对全集做一次即可，不涉及跨样本统计量
    processed = {m: preprocess_texts(texts, m) for m in PREPROCESS_METHODS}

    for m in PREPROCESS_METHODS:
        pd.DataFrame({'text': preprocess_texts(unlabeled, m)}).to_csv(
            os.path.join(OUT_DIR, 'unlabeled', f'{m}.csv'), index=False)

    for split_name, ratios in SPLITS.items():
        parts = dict(zip(('train', 'val', 'test'),
                         stratified_split_indices(y, ratios, RANDOM_SEED)))
        # 保证子集互不相交且覆盖全集
        all_idx = np.concatenate(list(parts.values()))
        assert len(all_idx) == len(y) == len(set(all_idx.tolist()))

        subset = np.empty(len(y), dtype=object)
        for name, idx in parts.items():
            subset[idx] = name
        pd.DataFrame({'orig_index': np.arange(len(y)), 'target': y,
                      'subset': subset}).to_csv(
            os.path.join(OUT_DIR, 'split_indices', f'{split_name}.csv'),
            index=False)

        manifest['splits'][split_name] = {
            'ratios': ratios,
            'sizes': {k: int(len(v)) for k, v in parts.items()},
            'class_counts': {k: np.bincount(y[v], minlength=10).tolist()
                             for k, v in parts.items()},
        }

        for m in PREPROCESS_METHODS:
            d = os.path.join(OUT_DIR, split_name, m)
            os.makedirs(d, exist_ok=True)
            for name, idx in parts.items():
                pd.DataFrame({'text': [processed[m][i] for i in idx],
                              'target': y[idx], 'orig_index': idx}).to_csv(
                    os.path.join(d, f'{name}.csv'), index=False)
        print(f'{split_name}: {manifest["splits"][split_name]["sizes"]}')

    with open(os.path.join(OUT_DIR, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print('saved to', os.path.abspath(OUT_DIR))


if __name__ == '__main__':
    main()
