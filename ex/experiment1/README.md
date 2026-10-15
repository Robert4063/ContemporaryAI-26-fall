# 实验一：文本分类

## 项目结构

```
experiment1/
├── train.py                      # 主训练脚本
├── requirements.txt              # 依赖包
├── README.md                     # 说明文档
├── models/
│   ├── __init__.py
│   └── model_definitions.py      # 模型定义（SVM/MLP）
├── utils/
│   ├── __init__.py
│   ├── data_preprocessing.py     # 数据预处理工具
│   └── visualization.py          # 可视化模块
└── outputs/
    ├── models/                   # 保存的最优模型
    ├── logs/                     # 实验日志和结果
    └── figures/                   # 生成的图表
```

## 实验设计

### 数据划分 (分层抽样，seed=42)

| 划分配置 | 训练集 | 验证集 | 测试集 |
|---------|--------|--------|--------|
| 80_10_10 | 80% | 10% | 10% |
| 85_5_10 | 85% | 5% | 10% |
| 90_5_5 | 90% | 5% | 5% |

### 文本预处理方法

1. **raw**: 保留原文（仅基本清洗）
2. **body_only**: 移除已知头字段，保留正文
3. **subject_body**: 保留标题和正文
4. **subject_body_author**: 保留标题、正文和作者

### TF-IDF参数

- min_df=2
- max_df=0.95
- max_features=20000
- norm='l2'
- sublinear_tf=True

### 模型配置

#### Linear SVM (SGD训练)
- C ∈ {0.1, 1, 10}
- max_iter=5000

#### MLP
- hidden_layer_sizes ∈ {(50,), (100,), (100, 50)}
- alpha ∈ {0.0001, 0.001}
- max_iter=300
- solver='adam', learning_rate=0.001

## 运行方式

### 1. 安装依赖

```bash
cd experiment1
pip install -r requirements.txt
```

### 2. 运行训练

```bash
cd experiment1
python train.py
```

脚本将：
1. 加载数据并进行预处理
2. 运行单epoch测试预估时间
3. 询问是否继续训练
4. 执行108组实验配置
5. 保存模型、可视化和结果

## 输出内容

- **模型文件**: `outputs/models/best_svm_model.pkl`, `best_mlp_model.pkl`
- **结果文件**: `outputs/logs/experiment_results.json`, `experiment_results.csv`
- **图表**:
  - `loss_curves_*.png` - Loss曲线
  - `model_comparison.png` - 模型对比图
  - `heatmap_*.png` - 热力图
  - `split_comparison.png` - 划分配置对比
  - `class_distribution.png` - 类别分布图
