"""
scripts/baselines/train_svm.py
SVM 训练脚本

使用 SGDClassifier(loss='hinge') 实现线性 SVM
- alpha = 1 / (C * n_train) 在 train 时计算，保存模型时也用这个公式
- 用 partial_fit 逐 epoch 训练并记录真实 hinge loss
- 300 个 epoch

重要: TF-IDF 只在 train 上 fit，val/test 只做 transform
最终模型 = 在 train 上用最优超参训练得到的那个模型，不重训
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_PROCESSED = PROJECT_ROOT / "data-processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RANDOM_STATE = 42

# 超参数搜索空间
C_VALUES = [0.1, 1, 10]
N_EPOCHS = 300


def compute_alpha(C, n_train):
    """计算 alpha = 1 / (C * n_train)
    这是唯一定义 alpha 公式的地方
    """
    return 1.0 / (C * n_train)


def compute_hinge_loss(y_true, y_decision):
    """计算真实 hinge loss (多分类版本)
    
    对于多分类，使用 one-vs-rest 的 hinge loss
    对每个样本，正确类别的得分应该比其他类别高至少 1
    """
    # y_decision 是 (n_samples, n_classes) 矩阵
    # 对于每个样本，提取真实类别的得分
    n_samples = len(y_true)
    true_scores = y_decision[np.arange(n_samples), y_true]
    
    # 计算与其他类别的 margin
    # loss = mean over all classes of max(0, 1 - (true_score - other_score))
    margins = true_scores[:, np.newaxis] - y_decision  # (n_samples, n_classes)
    
    # 对于真实类别，margin 是 0，我们排除它
    mask = np.ones_like(margins, dtype=bool)
    mask[np.arange(n_samples), y_true] = False
    
    # 计算 hinge loss
    losses = np.maximum(0, 1 - margins[mask])
    return np.mean(losses)


def load_data(split, version):
    """加载指定划分和版本的数据"""
    data_dir = DATA_PROCESSED / f"split_{split}" / version
    
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    unlabeled_df = pd.read_csv(data_dir / "unlabeled.csv")
    
    return {
        'train': {
            'text': train_df['text'].fillna('').astype(str).tolist(),
            'target': train_df['target'].values
        },
        'val': {
            'text': val_df['text'].fillna('').astype(str).tolist(),
            'target': val_df['target'].values
        },
        'test': {
            'text': test_df['text'].fillna('').astype(str).tolist(),
            'target': test_df['target'].values
        },
        'unlabeled': {
            'text': unlabeled_df['text'].fillna('').astype(str).tolist()
        }
    }


def create_vectorizer():
    """创建 TF-IDF 向量化器"""
    return TfidfVectorizer(
        min_df=2,
        max_df=0.95,
        max_features=20000
    )


def train_svm_epoch_by_epoch(X_train, y_train, C, n_train, classes, n_epochs=N_EPOCHS):
    """逐 epoch 训练 SVM，记录每个 epoch 的 loss"""
    
    alpha = compute_alpha(C, n_train)
    
    # 初始化分类器
    clf = SGDClassifier(
        loss='hinge',
        penalty='l2',
        alpha=alpha,
        random_state=RANDOM_STATE,
        learning_rate='optimal',
        eta0=0.0,
        warm_start=True,
        max_iter=1
    )
    
    train_losses = []
    
    for epoch in range(n_epochs):
        # 随机打乱数据
        indices = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        # 批量 partial_fit (每次处理一批样本)
        batch_size = 500
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            clf.partial_fit(X_batch, y_batch, classes=classes)
        
        # 计算训练集上的真实 hinge loss
        y_decision = clf.decision_function(X_train)
        loss = compute_hinge_loss(y_train, y_decision)
        train_losses.append(loss)
        
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch + 1}/{n_epochs}, Hinge Loss: {loss:.6f}")
    
    return clf, train_losses


def evaluate(clf, X, y, classes):
    """评估模型"""
    y_pred = clf.predict(X)
    
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    precision = precision_score(y, y_pred, average='macro', zero_division=0)
    recall = recall_score(y, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y, y_pred)
    
    return {
        'accuracy': acc,
        'f1_macro': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm.tolist()
    }, y_pred


def train_and_evaluate(split, version, C, log_file=None):
    """训练并评估一个配置"""
    print(f"\n{'='*50}")
    print(f"SVM 训练: split={split}, version={version}, C={C}")
    print(f"{'='*50}")
    
    # 创建输出目录
    config_dir = MODELS_DIR / "svm" / f"split_{split}" / version / f"C_{C}"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = RESULTS_DIR / "baseline" / "svm" / f"split_{split}" / version
    results_dir.mkdir(parents=True, exist_ok=True)
    
    loss_dir = RESULTS_DIR / "loss" / "svm" / f"split_{split}" / version
    loss_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    data = load_data(split, version)
    n_train = len(data['train']['text'])
    
    print(f"训练集: {n_train}, 验证集: {len(data['val']['text'])}, 测试集: {len(data['test']['text'])}")
    
    # 创建并拟合 TF-IDF (只在 train 上 fit)
    print("\n[1/5] 拟合 TF-IDF (只在 train 上)...")
    vectorizer = create_vectorizer()
    X_train = vectorizer.fit_transform(data['train']['text'])  # fit 在 train 上
    X_val = vectorizer.transform(data['val']['text'])           # val 只 transform
    X_test = vectorizer.transform(data['test']['text'])         # test 只 transform
    
    print(f"TF-IDF 词表大小: {len(vectorizer.vocabulary_)}")
    print(f"实际特征维度: {X_train.shape[1]}")
    
    # 转换为密集数组以便逐样本训练
    X_train_dense = X_train.toarray()
    X_val_dense = X_val.toarray()
    X_test_dense = X_test.toarray()
    
    # 获取类别
    classes = np.unique(data['train']['target'])
    
    # 训练 SVM
    print(f"\n[2/5] 训练 SVM (C={C}, alpha={compute_alpha(C, n_train):.6f})...")
    start_time = time.time()
    
    clf, train_losses = train_svm_epoch_by_epoch(
        X_train_dense, data['train']['target'], C, n_train, classes
    )
    
    train_time = time.time() - start_time
    print(f"训练完成, 耗时: {train_time:.1f}秒")
    
    # 保存训练 loss
    loss_file = loss_dir / f"loss_C_{C}.json"
    with open(loss_file, 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'config': {'C': C, 'split': split, 'version': version}
        }, f)
    
    # 评估
    print("\n[3/5] 评估模型...")
    
    train_metrics, _ = evaluate(clf, X_train_dense, data['train']['target'], classes)
    val_metrics, _ = evaluate(clf, X_val_dense, data['val']['target'], classes)
    test_metrics, test_preds = evaluate(clf, X_test_dense, data['test']['target'], classes)
    
    print(f"\n训练集: Acc={train_metrics['accuracy']:.4f}, F1={train_metrics['f1_macro']:.4f}")
    print(f"验证集: Acc={val_metrics['accuracy']:.4f}, F1={val_metrics['f1_macro']:.4f}")
    print(f"测试集: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1_macro']:.4f}")
    
    # 保存模型 (vectorizer, classifier) 元组
    print("\n[4/5] 保存模型...")
    model = (vectorizer, clf)
    model_file = config_dir / "model.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    # 保存结果
    print("\n[5/5] 保存结果...")
    results = {
        'config': {
            'model': 'svm',
            'split': split,
            'version': version,
            'C': C,
            'alpha': compute_alpha(C, n_train),
            'n_epochs': N_EPOCHS,
            'n_train': n_train,
            'vocab_size': len(vectorizer.vocabulary_)
        },
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'train_time': train_time
    }
    
    results_file = results_dir / f"results_C_{C}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到 {results_file}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SVM 训练脚本')
    parser.add_argument('--split', type=str, default='all',
                        choices=['A', 'B', 'C', 'all'])
    parser.add_argument('--version', type=str, default='all',
                        choices=['raw', 'body_only', 'subject_body', 'subject_body_author', 'all'])
    parser.add_argument('--C', type=float, default=None,
                        help='SVM C 参数 (默认: 所有)')
    
    args = parser.parse_args()
    
    splits = ['A', 'B', 'C'] if args.split == 'all' else [args.split]
    versions = ['raw', 'body_only', 'subject_body', 'subject_body_author'] if args.version == 'all' else [args.version]
    C_values = [args.C] if args.C else C_VALUES
    
    all_results = []
    
    for split in splits:
        for version in versions:
            for C in C_values:
                try:
                    result = train_and_evaluate(split, version, C)
                    all_results.append(result)
                except Exception as e:
                    print(f"\n[ERROR] 训练失败: split={split}, version={version}, C={C}")
                    print(f"错误: {e}")
                    import traceback
                    traceback.print_exc()
    
    # 保存汇总结果
    if all_results:
        summary_df = pd.DataFrame([{
            'model': r['config']['model'],
            'split': r['config']['split'],
            'version': r['config']['version'],
            'C': r['config']['C'],
            'val_accuracy': r['val_metrics']['accuracy'],
            'val_f1_macro': r['val_metrics']['f1_macro'],
            'test_accuracy': r['test_metrics']['accuracy'],
            'test_f1_macro': r['test_metrics']['f1_macro']
        } for r in all_results])
        
        summary_file = RESULTS_DIR / "baseline" / "svm" / "summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n汇总结果已保存到 {summary_file}")


if __name__ == "__main__":
    main()
