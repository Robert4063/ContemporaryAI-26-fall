# experiment_1:文本分类实验

输入文本 → TF-IDF 特征 → 模型训练 → 验证集调参 → 测试集评价 → 预测无标签数据 → 可视化。

- 任务:对邮件/新闻组风格的英文文本做 10 分类
- 数据:`train_data.csv` 共 7368 条(列 `text`、`target`),`test_data_unlabeled.csv` 共 2457 条无标签文本
- 模型:Naive Bayes (NB)、Logistic Regression (LR)、线性 SVM (SVM)、Decision Tree (DT)、MLP
- 随机种子固定为 42

## 目录总览

```text
experiment_1/
├── run_baseline.py        统一训练入口
├── data-Project1/         原始数据(课程提供,只读)
├── data-processed/        预处理后的数据(由脚本生成)
├── scripts/               正式流程代码
│   ├── preprocessing/     数据预处理
│   ├── baselines/         模型训练、保存、预测
│   └── plots/             画图
├── models/                已保存的最优模型(由脚本生成)
├── results/               所有实验输出(由脚本生成)
│   ├── baseline/          各组合的准确率/F1 等指标
│   ├── loss/              训练 loss 曲线数据
│   ├── plots/             单模型曲线图
│   ├── evaluation/figures/ 汇总对比图与最优配置表
│   └── predictions/       对无标签数据的预测结果
├── exploration/           探索与诊断脚本(不属于正式流程)
├── docs/                  实验要求与调参说明
└── .vscode/               编辑器配置
```

标注"由脚本生成"的目录可以删除后重新跑出来。`data-Project1/` 是源数据,请不要改动。

## 各目录说明

### `run_baseline.py`

统一训练入口。它依次调用 `scripts/baselines/train_*.py`,把结果写入 `results/baseline/` 和 `results/loss/`。

```powershell
python run_baseline.py --model all --split all --version all
python run_baseline.py --model mlp --split B --version subject_body_author
```

参数说明:`--model` 可选 `nb / lr / svm / dt / mlp / all`;`--split` 可选 `A / B / C / all`;`--version` 可选 `raw / body_only / subject_body / subject_body_author / all`。注意 `--version` 默认只跑 `raw`,要跑全部版本需显式写 `all`。

### `data-Project1/`:原始数据

| 文件 | 作用 |
| --- | --- |
| `train_data.csv` | 有标签训练数据,列为 `text`、`target` |
| `test_data_unlabeled.csv` | 无标签数据,只有 `text`,用于最终预测 |
| `predictions.csv` | 课程给的提交格式示例:无表头,每行一个标签 |
| `draft_main.py` | 课程给的数据读取示例,不参与流程 |
| `.DS_Store` | macOS 系统文件,可忽略 |

### `data-processed/`:预处理后的数据

由 `scripts/preprocessing/preprocess.py` 生成,不要手改。

- `split_A/`、`split_B/`、`split_C/`:三种划分比例,均按类别分层抽样

  | 划分 | train | val | test |
  | --- | --- | --- | --- |
  | A | 80%(5894) | 10%(737) | 10%(737) |
  | B | 85%(6262) | 5%(369) | 10%(737) |
  | C | 90%(6630) | 5%(369) | 5%(369) |

- 每个 `split_X/` 里有 16 个文件:`{train, val, test, unlabeled}_{版本}.csv`
- 4 种文本版本:
  - `raw`:原始文本,不清洗
  - `body_only`:去掉邮件头,只留正文
  - `subject_body`:标题 + 正文
  - `subject_body_author`:作者 + 标题 + 正文
- `unlabeled_*.csv` 是同样处理过的无标签数据,没有 `target` 列
- `split_meta.csv`:三种划分的样本数与比例

### `scripts/`:正式流程代码

所有脚本都从自身位置推算项目根目录,在哪里运行都可以。

**`scripts/preprocessing/`**
- `preprocess.py`:读取 `data-Project1/`,划分数据集,生成 4 种文本版本,输出到 `data-processed/`

**`scripts/baselines/`**

| 脚本 | 作用 |
| --- | --- |
| `train_nb.py` `train_lr.py` `train_svm.py` `train_dt.py` `train_mlp.py` | 各模型的训练与验证集调参,写出指标和 loss 数据 |
| `merge_results.py` | 把并行跑出的单组合 CSV 合并成 `{模型}_all_splits_all_versions.csv` |
| `train_save_models.py` | 按验证集选每个模型的最优配置,在 train+val 上重训并保存到 `models/` |
| `save_mlp_model.py` | 只针对 MLP 的保存脚本,不依赖合并文件 |
| `predict_from_models.py` | 加载 `models/` 里的模型预测无标签数据,默认自动选验证集最优模型 |
| `predict.py` | 不用已保存模型,直接重新训练后预测 |

**`scripts/plots/`**:所有画图脚本,详见该目录下的 `README.md`

| 脚本 | 输出 |
| --- | --- |
| `plot_4charts.py` | 4 张核心对比图,存到 `results/evaluation/figures/` |
| `plot_loss.py` | 每个(模型, 划分)一张曲线图,存到 `results/plots/` |
| `plot_mlp_versions.py` | 固定 MLP 结构,比较 4 种文本版本 |
| `plot_mlp_loss_compare.py` | MLP 训练 loss 在不同划分、预处理、超参下的对比,存到 `results/evaluation/figures/MLP_train_loss_*.png` |
| `plot_evaluation.py` | 按编号 `01_` 到 `09_` 输出 LR/MLP loss 曲线、各模型超参对比柱状图、模型综合对比、指标雷达图(每个划分各一张),存到 `results/evaluation/figures/`,并生成 `best_config_summary.csv` |

### `models/`:已保存的模型

由 `train_save_models.py` 生成。

- `{NB,LR,SVM,DT,MLP}_best.joblib`:`(vectorizer, classifier)` 元组,即 TF-IDF 加分类器
- `{模型}_config.csv`:该模型的最优配置
- `best_config.csv`:五个模型的最优配置汇总(模型、划分、文本版本、验证/测试 accuracy 与 F1)

### `results/`:实验输出

| 子目录 | 内容 |
| --- | --- |
| `baseline/` | `{模型}_{划分}_{版本}.csv`:各超参数的 train/val/test 准确率和 F1;`*_all_splits_all_versions.csv` 是合并表 |
| `loss/` | 同样命名的 CSV,保存训练 loss 曲线数据。LR、MLP 是逐轮真实 loss;NB、SVM、DT 没有逐轮 loss,记录的是随超参数变化的准确率 |
| `plots/` | `plot_loss.py`、`plot_mlp_versions.py` 生成的单模型曲线图 |
| `evaluation/figures/` | 汇总图与表。`plot_4charts.py` 生成 `chart1` 到 `chart4`;`plot_evaluation.py` 生成 `01_` 到 `09_` 编号图和 `best_config_summary.csv`;`plot_mlp_loss_compare.py` 生成 `MLP_train_loss_*.png`。当前目录里只有 `chart1` 到 `chart4` 和 `best_config_summary.csv`,其余图需要运行对应脚本才会出现 |
| `predictions/` | `predictions_{模型}.csv`:各模型的预测;`predictions.csv`:最终提交文件,无表头,每行一个标签 |

### `exploration/`:探索与诊断脚本

用于理解数据和做对照实验,**不属于正式流程**,删掉不影响主流程运行。

| 脚本 | 作用 |
| --- | --- |
| `tfidf_explore.py` | 查看 TF-IDF 的词表、维度、稀疏度 |
| `compare_splits.py` | 同一模型下对比划分 A 和 B 的表现 |
| `author_analysis.py` | 分析作者与类别的关联强度 |
| `author_overlap.py` | 检查训练集与无标签集的作者重叠情况 |
| `check_authors.py` `check_test_authors.py` | 诊断作者名提取是否正确 |
| `scripts/` | 早期的预测与保存脚本。其中 `predict.py`、`train_save_models.py` 与 `scripts/baselines/` 下的同名文件内容完全相同;`predict_from_models.py` 与正式版本有差异。正式版本都在 `scripts/baselines/` |
| `results/` | 探索脚本的输出目录,目前为空 |

### `docs/`:文档

| 文件 | 内容 |
| --- | --- |
| `实验一要求.pdf` | 实验原始要求 |
| `requests.md` | 实验要求的文字版:流程、数据划分规则、报告结构 |
| `TUNING.md` | SVM、LR、MLP 的超参数调参指南 |

## 复现步骤

依赖:Python 3、`pandas`、`numpy`、`scikit-learn`、`scipy`、`matplotlib`、`joblib`。

```powershell
# 1. 预处理:生成 data-processed/
python scripts/preprocessing/preprocess.py

# 2. 训练所有模型并在验证集上调参
python run_baseline.py --model all --split all --version all

# 3. 如果是分终端并行跑的,先合并结果
python scripts/baselines/merge_results.py

# 4. 保存每个模型的最优配置
python scripts/baselines/train_save_models.py --all

# 5. 预测无标签数据,输出到 results/predictions/
python scripts/baselines/predict_from_models.py

# 6. 画图
python scripts/plots/plot_4charts.py
python scripts/plots/plot_evaluation.py
```

## 当前最优结果

来自 `models/best_config.csv`,每个模型按验证集准确率选出最优配置:

| 模型 | 划分 | 文本版本 | 验证集 Acc | 测试集 Acc | 测试集 F1 |
| --- | --- | --- | --- | --- | --- |
| SVM | B | subject_body_author | 0.9648 | 0.9403 | 0.9408 |
| MLP | B | subject_body_author | 0.9621 | 0.9362 | 0.9367 |
| LR | B | subject_body_author | 0.9593 | 0.9335 | 0.9341 |
| NB | B | subject_body | 0.9621 | 0.9254 | 0.9266 |
| DT | B | subject_body_author | 0.7371 | 0.6689 | 0.6713 |

## 数据划分原则

- 训练集只用于拟合参数
- 验证集用于选超参数和选模型
- 测试集只用于最终评价,不参与训练和调参
- 保存最终模型时在 train+val 上重训,测试集始终不参与

## 已知注意事项

- `scripts/baselines/save_mlp_model.py` 里把项目根目录写死成了 `d:/AI/projects/experiment_1`,换机器或换路径需要先改这一行。
- `exploration/scripts/` 下的三个脚本是 `scripts/baselines/` 的早期版本,以后只维护后者。
- `run_baseline.py` 结束时提示的 `predict_from_models.py --auto` 并不存在这个参数。正确写法是 `--model auto`(默认值,即不带参数运行),或 `--model all` 用全部模型分别预测。
