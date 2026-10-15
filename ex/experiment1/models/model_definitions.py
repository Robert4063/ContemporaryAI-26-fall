# -*- coding: utf-8 -*-
"""
模型定义模块
包含：
1. SGD训练的Linear SVM
2. MLP分类器
3. 训练过程记录器
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score
import time


class LossHistory:
    """记录训练过程中的loss值"""
    
    def __init__(self):
        self.losses = []
        self.epochs = []
    
    def append(self, epoch: int, loss: float):
        self.epochs.append(epoch)
        self.losses.append(loss)
    
    def get_data(self) -> Tuple[List[int], List[float]]:
        return self.epochs, self.losses


class LinearSVMwithLogging(BaseEstimator, ClassifierMixin):
    """
    使用SGD训练的Linear SVM
    支持记录每个epoch的loss
    """
    
    def __init__(self, C: float = 1.0, max_iter: int = 1000, 
                 tol: float = 1e-4, random_state: int = 42,
                 early_stopping: bool = False, n_iter_no_change: int = 10,
                 verbose: int = 0):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.verbose = verbose
        
        # 存储loss历史
        self.loss_history = LossHistory()
        
        # 底层模型
        self.model = SGDClassifier(
            loss='hinge',  # SVM loss
            penalty='l2',
            alpha=1.0 / C,  # 正则化参数
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            early_stopping=early_stopping,
            n_iter_no_change=n_iter_no_change,
            learning_rate='optimal',
            verbose=verbose,
            n_jobs=-1
        )
        
        # 存储训练时间
        self.train_time_ = 0
        self.val_scores_ = []
        self.test_score_ = None
    
    def fit(self, X, y) -> 'LinearSVMwithLogging':
        """训练模型并记录loss"""
        import sklearn
        self.loss_history = LossHistory()
        self.val_scores_ = []
        
        # 自定义训练循环以记录loss
        if self.verbose > 0:
            print(f"  [SVM] Starting training with C={self.C}, max_iter={self.max_iter}")
        
        start_time = time.time()
        
        # 使用partial_fit进行逐epoch训练
        classes = np.unique(y)
        
        # 初始化模型
        self.model.partial_fit(X[:1], y[:1], classes=classes)
        
        for epoch in range(self.max_iter):
            # 训练一个epoch
            try:
                self.model.partial_fit(X, y, classes=classes)
            except StopIteration:
                break
            
            # 计算Hinge Loss（铰链损失）- 支持多分类
            decision = self.model.decision_function(X)
            
            if decision.ndim == 1 or decision.shape[1] == 1:
                # 二分类情况
                y_binary = np.where(y == classes[0], 1, -1)
                hinge_losses = np.maximum(0, 1 - y_binary * decision)
            else:
                # 多分类情况：使用多分类Hinge Loss (One-vs-All)
                # 对于每个样本，计算其对应类别的hinge loss
                n_samples = X.shape[0]
                hinge_losses_per_class = np.zeros((n_samples, len(classes)))
                
                for c_idx, c in enumerate(classes):
                    y_binary = np.where(y == c, 1, -1)
                    hinge_losses_per_class[:, c_idx] = np.maximum(0, 1 - y_binary * decision[:, c_idx])
                
                # 每个样本的loss是所有类别中的最大值
                hinge_losses = np.max(hinge_losses_per_class, axis=1)
            
            epoch_loss = np.mean(hinge_losses)
            
            self.loss_history.append(epoch + 1, epoch_loss)
            
            if self.verbose > 0 and (epoch + 1) % 100 == 0:
                print(f"  [SVM] Epoch {epoch + 1}/{self.max_iter}, Loss: {epoch_loss:.4f}")
        
        self.train_time_ = time.time() - start_time
        
        if self.verbose > 0:
            print(f"  [SVM] Training completed in {self.train_time_:.2f}s")
        
        return self
    
    def partial_fit(self, X, y, classes=None) -> 'LinearSVMwithLogging':
        """部分拟合"""
        self.model.partial_fit(X, y, classes=classes)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y=None):
        return self.model.score(X, y)
    
    def get_loss_curve(self) -> Tuple[List[int], List[float]]:
        """获取loss曲线数据"""
        return self.loss_history.get_data()
    
    def set_params(self, **params):
        """设置参数并重建底层模型"""
        for key, value in params.items():
            setattr(self, key, value)
        
        self.model = SGDClassifier(
            loss='hinge',
            penalty='l2',
            alpha=1.0 / self.C,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            early_stopping=self.early_stopping,
            n_iter_no_change=self.n_iter_no_change,
            learning_rate='optimal',
            verbose=self.verbose,
            n_jobs=-1
        )
        return self


class MLPClassifierWithLogging(MLPClassifier):
    """
    MLP分类器，带loss记录功能
    """
    
    def __init__(self, hidden_layer_sizes: Tuple = (100,), 
                 activation: str = 'relu',
                 solver: str = 'adam',
                 alpha: float = 0.0001,
                 batch_size: str = 'auto',
                 learning_rate: str = 'constant',
                 learning_rate_init: float = 0.001,
                 max_iter: int = 300,
                 shuffle: bool = True,
                 random_state: int = 42,
                 verbose: bool = False,
                 warm_start: bool = False,
                 momentum: float = 0.9,
                 nesterovs_momentum: bool = True,
                 early_stopping: bool = False,
                 validation_fraction: float = 0.1,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 epsilon: float = 1e-8):
        
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon
        )
        
        # 存储额外信息
        self.val_scores_ = []
        self.test_score_ = None
        self.train_time_ = 0
    
    def fit(self, X, y) -> 'MLPClassifierWithLogging':
        """训练MLP并记录过程"""
        import time
        start_time = time.time()
        
        # 使用sklearn内置的训练，同时记录validation_scores
        self.val_scores_ = []
        
        # sklearn的MLPClassifier会在训练时记录loss_curve_和validation_scores_
        super().fit(X, y)
        
        self.train_time_ = time.time() - start_time
        
        return self
    
    def get_loss_curve(self) -> Tuple[List[int], List[float]]:
        """获取loss曲线数据"""
        if hasattr(self, 'loss_curve_'):
            epochs = list(range(1, len(self.loss_curve_) + 1))
            return epochs, self.loss_curve_
        return [], []


def create_svm_configs() -> List[Dict]:
    """创建SVM配置列表"""
    return [
        {'C': 0.1, 'max_iter': 5000},
        {'C': 1, 'max_iter': 5000},
        {'C': 10, 'max_iter': 5000},
    ]


def create_mlp_configs() -> List[Dict]:
    """创建MLP配置列表"""
    hidden_layer_options = [(50,), (100,), (100, 50)]
    alpha_options = [0.0001, 0.001]
    
    configs = []
    for h in hidden_layer_options:
        for a in alpha_options:
            configs.append({
                'hidden_layer_sizes': h,
                'alpha': a,
                'max_iter': 300
            })
    return configs


def train_and_evaluate(model: BaseEstimator, 
                       X_train, y_train,
                       X_val, y_val,
                       X_test, y_test,
                       model_name: str,
                       verbose: int = 1) -> Dict:
    """
    训练模型并评估
    
    Returns:
        包含训练结果的字典
    """
    result = {
        'model_name': model_name,
        'val_acc': 0.0,
        'val_f1': 0.0,
        'test_acc': 0.0,
        'test_f1': 0.0,
        'train_time': 0.0,
        'loss_curve': ([], [])
    }
    
    # 训练
    start_time = time.time()
    model.fit(X_train, y_train)
    result['train_time'] = time.time() - start_time
    
    # 验证集评估
    y_val_pred = model.predict(X_val)
    result['val_acc'] = accuracy_score(y_val, y_val_pred)
    result['val_f1'] = f1_score(y_val, y_val_pred, average='macro')
    
    # 测试集评估
    y_test_pred = model.predict(X_test)
    result['test_acc'] = accuracy_score(y_test, y_test_pred)
    result['test_f1'] = f1_score(y_test, y_test_pred, average='macro')
    
    # 获取loss曲线
    if hasattr(model, 'get_loss_curve'):
        result['loss_curve'] = model.get_loss_curve()
    
    if verbose > 0:
        print(f"  {model_name}: Val Acc={result['val_acc']:.4f}, "
              f"Val F1={result['val_f1']:.4f}, "
              f"Test Acc={result['test_acc']:.4f}, "
              f"Time={result['train_time']:.2f}s")
    
    return result


if __name__ == '__main__':
    # 简单测试
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # 生成测试数据
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 测试SVM
    print("Testing LinearSVMwithLogging...")
    svm = LinearSVMwithLogging(C=1.0, max_iter=100, verbose=1)
    svm.fit(X_train, y_train)
    print(f"Test accuracy: {svm.score(X_test, y_test):.4f}")
    
    # 测试MLP
    print("\nTesting MLPClassifierWithLogging...")
    mlp = MLPClassifierWithLogging(hidden_layer_sizes=(50,), max_iter=100, verbose=True)
    mlp.fit(X_train, y_train)
    print(f"Test accuracy: {mlp.score(X_test, y_test):.4f}")
