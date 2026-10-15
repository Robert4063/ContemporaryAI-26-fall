# 实验结果与来源

| 目录 | 用途 |
| --- | --- |
| `baseline/` | 108个指标CSV，每个包含train/val两行，完整保留36组SGD SVM与72组MLP搜索结果 |
| `loss/` | 对应108组配置的真实300轮loss数组；SVM为margin hinge诊断值，MLP为交叉熵加L2训练目标，绝对值不能跨模型直接比较 |
| `linear_svm_split_B/metrics/` | 12组LinearSVC的train/val结果 |
| `linear_svm_split_B/` | 按模型/版本的均值与最佳验证结果、冻结选模记录、最终测试对比及运行配置 |
| `linear_svm_split_B/test/` | 三个锁定配置的逐样本测试预测、混淆矩阵与指标JSON |
| `evaluation/figures/` | 所有划分/清洗/结构的对比与收敛图，报告使用其中数据生成的紧凑版本 |
| `evaluation/` | loss完整曲线CSV、关键轮次摘要、版本效果均值与图表数据来源哈希 |
| `predictions/` | 全120组验证排序与最终prediction.csv来源记录 |

读取基线指标必须筛选 `eval_set == "val"` 再用于选模；不能把训练Accuracy当作验证结果。`linear_svm_split_B/test_final_comparison.csv` 为三模型测试结果汇总。

这些小体积数据是报告图表和判断的原始证据，因此完整保留，而不是只留下最佳分数。`project_audit.json`是较早的全量核查快照，其中历史文件路径/哈希不表示当前整理后的文件布局；当前整理检查另见 `submission_check.json`（运行提交核查工具生成）。

模型及输入来源哈希以本次提交原件为准。CSV、JSON和二进制文件通过 `.gitattributes` 禁止换行转换，避免Git在不同系统检出时改变已保存的哈希。
