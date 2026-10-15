"""
merge_results.py
================
把并行跑出来的单组合 CSV ({MODEL}_{split}_{version}.csv) 合并为
{MODEL}_all_splits_all_versions.csv, 供 train_save_models.py / scripts/plots/*.py 使用。

用法:
  python scripts/baselines/merge_results.py
"""

from pathlib import Path

import pandas as pd

ROOT   = Path(__file__).resolve().parents[2]
MODELS = ["NB", "SVM", "DT", "LR", "MLP"]
SPLITS = ["A", "B", "C"]
VERSIONS = ["raw", "body_only", "subject_body", "subject_body_author"]


def merge(directory):
    for model in MODELS:
        parts = []
        missing = []
        for s in SPLITS:
            for v in VERSIONS:
                p = directory / f"{model}_{s}_{v}.csv"
                if p.exists():
                    parts.append(pd.read_csv(p))
                else:
                    missing.append(f"{s}_{v}")
        if missing:
            print(f"  [{directory.name}] {model}: 缺少 {len(missing)} 个组合, 跳过合并: {missing}")
            continue
        out = directory / f"{model}_all_splits_all_versions.csv"
        pd.concat(parts, ignore_index=True).to_csv(out, index=False)
        print(f"  [{directory.name}] {model}: 合并 {len(parts)} 个文件 -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    merge(ROOT / "results" / "baseline")
    merge(ROOT / "results" / "loss")
