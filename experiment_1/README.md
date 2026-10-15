# 实验一：文本分类

使用 TF-IDF 将新闻组邮件转为文本向量，比较 SGD SVM、MLP 与 LinearSVC，最后对无标签文本输出 0-9 中的一个类别。原始标注数据7368条，无标签数据2457条，随机种子固定为42。

本目录对应仓库的 `lab_1` 分支。建议先阅读[最终报告](reports/实验一报告.pdf)，再查看实验代码和结果。报告以 [Notebook](reports/实验一报告.ipynb) 为唯一正文主源，[Markdown](reports/实验一报告.md) 与之同步；PDF为两页，采用论文字体和黑色三线表。

## 实验做了什么

- 比较三种分层划分：A为80:10:10，B为85:5:10，C为90:5:5。
- 比较四种文本版本：原文、正文、标题+正文、作者+标题+正文。
- 原实验运行36组SGD SVM和72组MLP；仅在B划分补充12组LinearSVC，共120组配置。
- TF-IDF使用 `min_df=2`、`max_df=0.95`、`max_features=20000`，词表与IDF只在训练集拟合。
- 按验证Accuracy、宏F1、配置名依次排序选模，测试分数不参与排序。模型及训练细节见[补充实验记录](docs/LINEAR_SVM_SPLIT_B.md)。

## 主要结果与提交文件

下表为B划分中每种模型按验证集锁定的配置；测试指标来自已保存的最终评价，不用于选模。

| 模型 | 选中配置 | 验证Accuracy | 测试Accuracy | 测试宏F1 |
| --- | --- | ---: | ---: | ---: |
| SGD SVM | 标题+正文，C=1 | 94.31% | 92.94% | 92.99% |
| MLP | 作者+标题+正文，h=(100,)，α=0.001 | 95.93% | 93.22% | 93.28% |
| LinearSVC | 原文，C=1 | 96.21% | 93.08% | 93.16% |

最终预测采用验证Accuracy最高的 **LinearSVC（B、raw、C=1）**。根目录的 [prediction.csv](prediction.csv) 共2457行，无表头，每行一个0-9标签，顺序与原始无标签CSV一致；文件SHA256为 `e064cb6c39d721229a45d10010f23697773f1e74fe72cb66209e01076e775819`。选模依据、模型及输入哈希保存在 [results/predictions/](results/predictions/)。

## 按评分要求查找材料

[实验要求PDF](docs/实验一要求.pdf#page=18)最后一页给出的评分为：代码实现30分、模型性能10分、实验报告50分、创新10分。详细对应关系见[提交要求与评分清单](docs/SUBMISSION_GUIDE.md)。

| 评分项 | 本仓库对应材料 |
| --- | --- |
| 代码实现30分 | `scripts/`、`run_baseline.py`、依赖版本及本README运行命令 |
| 模型性能10分 | 验证选模表、三模型测试指标，以及报告中的模型复杂度讨论 |
| 实验报告50分 | `reports/`中的最终报告，含设置、loss、准确率、结构与清洗效果分析 |
| 创新10分 | 三种划分、四种邮件信息保留方式、MLP结构对比，以及B划分LinearSVC补充实验；实际评价以课程标准为准 |

## 目录用途

| 目录或文件 | 用途及保留原因 |
| --- | --- |
| `data-Project1/` | 两份原始CSV，是重现划分、训练和预测的输入 |
| `data-processed/` | 预处理生成目录；Git仅保留说明，CSV缓存可由代码重建 |
| `models/` | Git仅保留最终LinearSVC的模型、向量器和配置，支持直接预测；其他119组模型仅留在本地 |
| `results/baseline/` | 原108组配置的train/val指标，保留完整搜索记录 |
| `results/loss/` | 原108组真实300轮loss数组，用于重画曲线和检查收敛 |
| `results/linear_svm_split_B/` | 新增12组指标、验证比较、锁定选模记录、三模型最终测试结果和混淆矩阵 |
| `results/evaluation/` | 各划分与版本的对比图、曲线明细、摘要和来源哈希，支撑分析与重画 |
| `results/predictions/` | 最终预测的验证排序表与来源记录 |
| `reports/` | 最终Notebook、同文Markdown、两页PDF、引用图片及修订记录 |
| `docs/` | 原始课程要求、评分对应表、补充实验、思考题分析与已知问题 |
| `scripts/` | 数据预处理、训练、预测、绘图、报告导出和提交核查代码 |
| `TODO.md` | 仍待处理的问题，不把尚未完成的实验或流程修复标为已完成 |

`.local/`保存历史交接、旧草稿、日志、失效工具和整理清单，由`.gitignore`排除。预处理缓存、非最终模型、虚拟环境和Python缓存也不上传。所有完整模型仍在本地，整理没有删除训练成果或改变报告正文。

## 安装环境

实验使用Python 3.10.21、scikit-learn 1.7.1。为兼容随仓库提供的模型，请使用Python 3.10与固定依赖。在本README所在目录运行，以下示例为PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如果从仓库根目录开始，先进入 `ex3`。Linux/macOS激活虚拟环境使用 `source .venv/bin/activate`。

## 直接复现最终预测

```powershell
python scripts/verify_submission.py
python scripts/predict_unlabeled.py --output prediction_reproduced.csv
```

这条预测命令仅加载最终模型并处理原始无标签数据，不重新训练，也不依赖预处理缓存。已有缓存存在时会额外核对文本和顺序。输出应与提交的 `prediction.csv` 逐字节一致。

## 重新训练完整实验

完整实验包含108组基线和12组补充模型，MLP均训练300轮，运行时间明显长于直接预测。请在**单独复制的实验目录**中重训，先将已提交的模型和结果改名留作参考，避免旧哈希缓存与新训练产物混用：

```powershell
Rename-Item -LiteralPath models -NewName models.reference
Rename-Item -LiteralPath results -NewName results.reference
python scripts/preprocessing/preprocess.py
python run_baseline.py --sequential
python scripts/baselines/run_linear_svm_split_B.py
python scripts/predict_unlabeled.py --output prediction_reproduced.csv
```

`run_baseline.py`不加`--sequential`时只列出任务。单独重训某个配置可直接调用：

```powershell
python scripts/baselines/train_svm.py --split split_B --version subject_body --C 1
python scripts/baselines/train_mlp.py --split split_B --version subject_body_author --hidden 100 --alpha 0.001
```

LinearSVC补充脚本会核对B划分四种版本的基线TF-IDF，因此应先完成基线。旧基线训练脚本曾计算test指标；该历史事实见[已知问题](docs/KNOWN_ISSUES.md)，不能声称历史实验全程封存测试集。重训会生成新的模型文件和耗时记录，原始归档哈希仅代表本次提交的产物。

## 重画图表与导出报告

已有指标与loss随仓库保留，重画图表无需重新训练：

```powershell
python scripts/plots/plot_loss_curves.py
python scripts/plots/plot_summary_figures.py
python scripts/plots/plot_mlp_hidden_size_split_B.py
python scripts/reports/make_report_figures.py
python scripts/reports/export_report.py
```

PDF导出使用ReportLab，当前配置读取Windows字体目录中的宋体、黑体和Times New Roman。训练、预测、绘图不依赖这些字体；其他系统可直接阅读已提交PDF，若需重新导出则需在导出脚本中设置相应字体路径。Notebook全为Markdown单元，不需要执行训练。导出器会同步Markdown，并在PDF超过两页时报错。

已知重复文本、数据划分依赖、历史测试访问、SGD元数据和单种子限制集中记录在[已知问题](docs/KNOWN_ISSUES.md)与[TODO](TODO.md)。提交核查工具只检查文件、指标、来源哈希与报告同步，不把这些检查当成无泄露证明。
