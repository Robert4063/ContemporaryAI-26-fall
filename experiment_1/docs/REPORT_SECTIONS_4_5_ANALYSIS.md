# 第四、五部分重排与图表分析

本文件为补充分析与历史修订记录，最终正文以 reports/实验一报告.ipynb 为准；备份及旧校验基准仅在本地归档中保留。

更新：2026-10-15。以用户当前修改后的 `reports/实验一报告.ipynb` 为主文件；标题、仓库说明、引言和前三部分按单元完整保留。修改前备份为 `.local/archive/reports/archive/实验一报告_第四五部分修改前.ipynb`，保留校验基准见 `.local/archive/reports/assets/report_revision_baseline.json`。

## 第四部分：loss、层数、准确率与模型差异

图1使用同一B划分、subject_body_author版本，对比SGD SVM的三个C与MLP在两种固定α下的三种h。完整版为 `results/evaluation/figures/svm_mlp_split_B_loss_combined.png`，结合既有MLP隐藏层曲线数据，并增加SGD曲线与9组配置关键轮次表；报告使用同一数据的紧凑排版 `assets/report_loss_comparison.png`。

SGD的C从1增至10，训练集诊断loss从0.041853降到0.024179，训练Accuracy由98.85%升至99.30%，验证却从93.50%降至92.14%。更低的训练目标、更高的训练准确率，均不能代替验证结果；SGD曲线还受到原脚本重复warm_start fit这一实现的影响。

MLP固定α=0.001，从h=(50,)加宽到(100,)，loss末值为0.006358/0.006319，val Accuracy由95.66%到95.93%，只多对1条。加深至(100,50)后loss为0.002633，train仍为100%，val却降到94.31%，比(100,)少对6条；α=0.0001时加宽反而少对1条。这与泛化不随训练拟合改善的风险一致，尚不能仅凭单种子记录断言稳定因果。

图2 `assets/report_accuracy_comparison.png` 也置于第四部分。它比较B划分各模型、各文本版本的最佳验证配置；这是最优值，不能与第五部分的均值混为一谈。MLP在各版本上高于原SGD基线，但新增LinearSVC与MLP接近，说明不能将SGD较弱直接归因于线性模型能力不足。LinearSVC使用squared_hinge，SGD使用hinge，训练方式也不同，差距不能只归为求解器。

各模型整体最优配置按val锁定，test Accuracy为SGD 92.94%、MLP 93.22%、LinearSVC 93.08%。LinearSVC验证多对1条、测试少对1条；模型参数约20万与MLP的200万也需纳入取舍。不能用test排名反过来改变最终无标签预测模型。

SVM/MLP的loss定义不同，不能按绝对值比较模型好坏。所有记录均为300轮，没有逐轮val曲线或不同epochs实验，不能推断超过300轮的收益或最佳停止轮次。完整思考题分析继续见 `EXPERIMENT_THINKING_Q1_Q2.md`。

## 第五部分：SVM与MLP版本效果合并

用户确认将 `version_effect_svm.png` 与 `version_effect_mlp.png` 合并。合并后完整版为 `results/evaluation/figures/version_effect_svm_mlp_combined.png`，报告版为 `assets/report_version_effect_comparison.png`。四面板按列区分SGD SVM/MLP、按行区分验证Accuracy/宏F1，A/B/C采用固定颜色和标记。

核查发现原 `plot_version_effect()` 采用groupby的字母排序（body_only先于raw），却使用raw、body的固定横轴标签，导致前两个点错标。已将每个分组显式reindex为 `raw, body_only, subject_body, subject_body_author`，重绘两张原图后再生成合并图。原始数值、训练数据与模型均未改变。

所有点为对应model×split×version的超参数均值：SGD每组3配置、MLP每组6配置。摘要为 `results/evaluation/version_effect_svm_mlp_summary.csv`（24组、两个均值指标），不使用test。

| B划分文本版本 | SGD平均验证Accuracy | MLP平均验证Accuracy |
| --- | ---: | ---: |
| raw | 92.05% | 95.48% |
| body_only | 91.06% | 93.22% |
| subject_body | 93.13% | 94.76% |
| subject_body_author | 92.59% | 95.48% |

从raw改为正文，B的SGD下降约0.99个百分点，MLP下降约2.26个百分点；A/C也都下降。说明去掉邮件头并不只去掉噪声，也可能丢失标题、来源等有用线索。

补回标题后，B的两模型均回升；继续加入作者，SGD反而下降，MLP回到raw的均值。A的MLP加作者略低于标题+正文，C略升，说明作者效果依赖划分和模型，不能宣称普遍有益。宏F1与Accuracy总体同趋势，但均值只描述这一组搜索范围，不是性能的无偏估计。

A/B共用test，C的test为其子集，三个划分也不是三次独立复验；B的MLP均值较高不能证明某划分比例普遍最优。已有重复文本、标签冲突、旧流程test访问和单种子等局限继续保留。

## 同步与记录

- Notebook是唯一正文主源，Markdown由它导出，PDF通过 `scripts/reports/export_report.py` 生成。
- 因第四部分跨页，Markdown分页注释不代表章节结束；图2仍位于第四部分，图3在第五部分。
- 前三部分的内容与单元元数据校验不变；为满足两页上限，仅调整版面并压缩第六、七部分，保留关键局限及预测说明。
- 本次没有重训或新增test预测，prediction.csv保持不变。
