# 已知问题与结果使用口径

本文件整理历史核查事实。目录整理、上传及文件哈希核查均没有修复这些实验设计问题，也没有改动最终报告正文或重训模型。原始核查数据见 `results/evaluation/project_audit.json`。

## 预处理文本跨集合重复

这里指同一划分内部的train、val、test之间存在处理后相同文本，并非仅指A/B/C三套划分共用了样本。

| 文本版本 | A | B | C |
| --- | --- | --- | --- |
| raw完整字符串 | 无跨集合相同文本 | 无 | 无 |
| body_only | train-test有3个相同文本 | train-test有3个相同文本 | train-val有1个，train-test有1个 |
| subject_body | train-test有1个相同文本 | train-test有1个相同文本 | train-test有1个相同文本 |
| subject_body_author | train-test有1个相同文本 | train-test有1个相同文本 | train-test有1个相同文本 |

以上为不同文本数，不能直接解释为独立邮件组数；所有组合的val-test交集为0。具体来源包括空正文、相同处理器正文但标签2/7冲突，以及作者/标题/正文相同、标签均为4的“First Bike??”邮件。清洗可能使原本不同的原文映射为同一输入，重复可能使评价偏乐观，标签冲突则让相同输入无法同时预测正确。未经分组重划分或敏感性分析，不能声称最终分数与接近的模型排名不受影响。

A/B的body_only有3/737条测试输入在训练中出现；C有1/369条，另有1条验证输入为空且训练中也有空正文。raw整段不相同也不能排除语义或内容重复。

## 划分、测试访问与模型比较

- A/B共用本地test，C的test是其子集，三套结果不能视为三次独立实验；B在当前结果中较好不证明85:5:10普遍最优。
- 旧SGD SVM、MLP训练脚本曾逐配置预测test并打印分数，CSV只保存train/val。不能声称历史test一直封存；目前选模脚本明确只依据验证Accuracy与宏F1。
- LinearSVC补充实验先冻结验证选择再保存测试评价；LinearSVC的squared_hinge与SGD的hinge不同，因此不只是更换求解器的消融。
- B验证集369条。LinearSVC验证Accuracy96.21%，MLP95.93%，只相差1条；测试MLP93.22%与LinearSVC93.08%也只相差1条，不能用测试排序反过来重新选模。
- 每版本MLP搜索6组、两种SVM各3组，且只使用一个种子。没有多种子、统计显著性或不同epochs实验，也没有逐轮验证曲线。

## 历史工具与元数据

旧 `scripts/check_no_leak.py` 项目根目录错误，缺失文件时会空遍历报通过；旧 `scripts/merge_results.py` 的混合参数列提取与选择粒度也不可靠。这两个脚本已从可运行主目录移至 `.local/archive/scripts/`，不作为提交证据。新的 `scripts/verify_submission.py` 只核查文件与来源一致性，不声称证明训练无泄露。

旧SGD `config.pkl` 的alpha保存时使用了指标字典长度，不能当作实际训练alpha。应从保存的分类器属性核验；训练代码实际使用 `alpha=1/(C*n_train)`。原模型和旧元数据没有被静默修改。

## 报告披露状态

用户最终Notebook未保留历史版本中部分重复文本、历史test访问与单种子的局限说明。本次按其最终稿只做目录整理和上传，未把旧内容写回正文。后续报告修订事项继续保存在 `TODO.md`。
