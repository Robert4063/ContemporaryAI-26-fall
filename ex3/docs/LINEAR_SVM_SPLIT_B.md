# LinearSVC 补充实验：split_B

实验日期：2026-10-15。目的：检查 LinearSVC 在当前数据和特征下是否优于现有 SGD SVM、MLP，不预设胜负。

## 实验方案

- 范围仅为 split_B（train=6262，val=369，test=737），四种文本版本全部保留。
- 新增 `sklearn.svm.LinearSVC`：C∈{0.1,1,10}，共 12 组配置；L2、squared_hinge、OvR、dual=auto、tol=1e-4、max_iter=10000、random_state=42。
- TF-IDF 与旧实验一致：min_df=2、max_df=0.95、max_features=20000，其余为库默认值；只在 train 拟合，并核对词表、IDF 与旧实验一致。
- 旧 SGD SVM、MLP 使用已保存模型和指标，不重训，不修改既有划分或清洗文本。
- 选模规则预先固定为 val Accuracy 降序、macro-F1 降序、config_id 字典序升序。各模型分别选择最优版本/超参数；先将选择和模型哈希写入 selection_manifest.json，再进行最终 test 评估。
- 图中同时给出每个模型/文本版本的超参数均值与最佳验证配置，避免将均值和最优值混为一谈。
- 最终三个入选配置在同一个 split_B 测试集上比较 Accuracy、macro Precision/Recall/F1，保存预测与混淆矩阵；重复运行复用已完成且哈希一致的结果。

## 解释与局限

原 SGDClassifier(loss=hinge) 本身也是线性 SVM。新增 LinearSVC 使用 LIBLINEAR 与 squared_hinge，因此属于实现和训练目标的补充对比，不能将差异完全归因于优化器。官方参考：[LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC)、[SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)。实际使用环境为 scikit-learn 1.7.1，以本地安装版本的行为为准。

LinearSVC 不提供逐次迭代 loss 历史；本实验记录真实耗时、最终迭代数、收敛情况和评价指标，不伪造 loss 曲线。不同求解器的迭代数不能与 SGD epoch 或 MLP iteration 直接比较。

已有跨集合内容重复、标签冲突及旧训练流程中 test 曾被访问的问题继续适用，本次不修改数据以保持横向比较口径。val 仅 369 条；三种模型搜索配置数不完全相同（每版本 SGD SVM=3、MLP=6、LinearSVC=3），需在报告说明。

## 复现与输出

在项目根目录使用原实验环境运行：

```powershell
& 'D:\AI\envs\experiment_1\python.exe' scripts/baselines/run_linear_svm_split_B.py
```

- 新模型：`models/linear_svm_split_B_*/`，每组含 model.pkl、vectorizer.pkl、config.json。
- 日志、逐配置指标、val 比较表、冻结选模记录、最终 test 比较、预测、混淆矩阵：`results/linear_svm_split_B/`。
- 新增对比图：`results/evaluation/figures/model_comparison_split_B_with_linear_svm.png`。
- 原有 baseline 文件、108 组模型、loss 及两模型对比图保留；补充实验单独存放，不混入原来的 108 组网格统计。

## 执行记录

2026-10-15 13:26（Asia/Shanghai）完成。12 组模型均收敛，迭代数为 12–327；四种版本各自的词表与 IDF 均与旧实验一致。原 108 组模型没有重训。

以下配置均由验证集选择；test 列仅用于选择固定后的评价，不参与排序：

| 模型 | 选中版本与超参数 | val Accuracy | val macro-F1 | test Accuracy | test macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| SGD SVM | subject_body，C=1 | 94.31% | 94.32% | 92.94% | 92.99% |
| MLP | subject_body_author，hidden=(100,)，alpha=0.001 | 95.93% | 95.98% | 93.22% | 93.28% |
| LinearSVC | raw，C=1 | 96.21% | 96.25% | 93.08% | 93.16% |

LinearSVC 的最优验证 Accuracy 比 MLP 高 1/369（约 0.271 个百分点），但固定配置后的 test Accuracy 比 MLP 低 1/737（约 0.136 个百分点）。test 正确数量分别为 SGD SVM=685、MLP=687、LinearSVC=686。这次实验没有支持“LinearSVC 在测试集上高于 MLP”的结论，三者接近的数值也不能仅凭这一套划分认定具有统计显著性。

四种文本版本的验证集结果如下，每格为“超参数均值 / 最佳配置”，两个统计量均未使用 test：

| 文本版本 | SGD SVM | MLP | LinearSVC |
| --- | ---: | ---: | ---: |
| raw | 92.05% / 93.22% | 95.48% / 95.93% | 95.57% / 96.21% |
| body_only | 91.06% / 91.33% | 93.22% / 93.77% | 93.50% / 94.31% |
| subject_body | 93.13% / 94.31% | 94.76% / 95.66% | 95.30% / 95.93% |
| subject_body_author | 92.59% / 93.50% | 95.48% / 95.93% | 95.21% / 95.93% |

完整数值、配置、日志、预测与混淆矩阵见 `results/linear_svm_split_B/`，状态见 `run_manifest.json`。原数据重复、标签冲突及历史 test 访问的局限仍需写进报告。新增流程先冻结验证集选择再评价 test，但不能据此将旧流程描述为从未访问 test。
