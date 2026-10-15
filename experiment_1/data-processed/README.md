# 预处理生成目录

从实验根目录运行：

```powershell
python scripts/preprocessing/preprocess.py
```

生成A/B/C三种分层划分，每种划分有raw、body_only、subject_body、subject_body_author四种文本版本，各自包含train/val/test/unlabeled，共48个CSV，另外有划分统计与可读样例。标注列名为 `target`。

这些CSV是约198MiB的可重建缓存，Git只保留本说明。原始输入位于 `data-Project1/`，随机种子42。直接使用最终模型预测不依赖该缓存。

已知清洗后跨集合重复仍存在，详见 `docs/KNOWN_ISSUES.md`。脚本生成的 `split_meta.csv` 按四个文本版本合计计数，不能作为单个版本的样本数。
