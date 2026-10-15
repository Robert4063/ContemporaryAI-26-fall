# 模型文件

Git仅保留 `linear_svm_split_B_raw_C1.0/`，这是全部120组配置中验证Accuracy最高的最终预测模型。目录中包含：

- `model.pkl`：训练完成的LinearSVC分类器。
- `vectorizer.pkl`：只在B训练集拟合的TF-IDF向量器。
- `config.json`：参数、数据哈希、收敛信息与模型/指标哈希。

从实验根目录运行 `python scripts/predict_unlabeled.py --output prediction_reproduced.csv` 即可直接预测，不需要重新训练。随仓库保存的模型使用Python 3.10.21、scikit-learn 1.7.1。

本地其余119组模型保留用于回溯，由根目录 `.gitignore` 排除。它们的指标、loss和补充实验结果仍随Git保存；完整重训方法见根目录README。
