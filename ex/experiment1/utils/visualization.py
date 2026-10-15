# -*- coding: utf-8 -*-
"""
可视化模块
包含各种图表绘制函数
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from typing import List, Dict, Tuple, Optional
import pandas as pd


def plot_loss_curves(results: Dict[str, List[Dict]], 
                     output_dir: str,
                     model_type: str = 'all'):
    """
    绘制loss曲线对比图
    
    Args:
        results: 结果字典 {config_name: [result_dicts]}
        output_dir: 输出目录
        model_type: 'svm', 'mlp', 或 'all'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每种配置组合绘制loss曲线
    for config_name, result_list in results.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 收集所有loss曲线
        svm_curves = []
        mlp_curves = []
        
        for result in result_list:
            model_name = result.get('model_name', '')
            loss_data = result.get('loss_curve', ([], []))
            epochs, losses = loss_data
            
            if 'svm' in model_name.lower() or 'SVM' in model_name:
                svm_curves.append((model_name, epochs, losses))
            elif 'mlp' in model_name.lower() or 'MLP' in model_name:
                mlp_curves.append((model_name, epochs, losses))
        
        # 绘制SVM loss曲线
        ax1 = axes[0]
        for name, epochs, losses in svm_curves:
            if epochs and losses:
                ax1.plot(epochs, losses, label=name, linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (negative accuracy)', fontsize=12)
        ax1.set_title('SVM Loss Curve', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制MLP loss曲线
        ax2 = axes[1]
        for name, epochs, losses in mlp_curves:
            if epochs and losses:
                ax2.plot(epochs, losses, label=name, linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('MLP Loss Curve', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"loss_curves_{config_name}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")


def plot_combined_loss_curves(all_results: List[Dict],
                              output_dir: str):
    """
    绘制所有模型的综合loss曲线对比图
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 按模型类型分组
    svm_results = [r for r in all_results if 'svm' in r.get('model_name', '').lower()]
    mlp_results = [r for r in all_results if 'mlp' in r.get('model_name', '').lower()]
    
    # SVM loss曲线
    if svm_results:
        fig, ax = plt.subplots(figsize=(12, 6))
        for result in svm_results:
            name = result['model_name']
            epochs, losses = result.get('loss_curve', ([], []))
            if epochs and losses:
                ax.plot(epochs, losses, label=name, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (negative accuracy)', fontsize=12)
        ax.set_title('SVM Training Loss Curves', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = "svm_all_loss_curves.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    # MLP loss曲线
    if mlp_results:
        fig, ax = plt.subplots(figsize=(12, 6))
        for result in mlp_results:
            name = result['model_name']
            epochs, losses = result.get('loss_curve', ([], []))
            if epochs and losses:
                ax.plot(epochs, losses, label=name, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('MLP Training Loss Curves', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = "mlp_all_loss_curves.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")


def plot_model_comparison(results: List[Dict],
                         output_dir: str,
                         metric: str = 'val_acc',
                         title: str = 'Model Comparison'):
    """
    绘制模型对比柱状图
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 按验证集准确率排序
    sorted_results = sorted(results, key=lambda x: x.get(metric, 0), reverse=True)
    
    model_names = [r['model_name'] for r in sorted_results]
    val_accs = [r['val_acc'] for r in sorted_results]
    test_accs = [r['test_acc'] for r in sorted_results]
    
    # 创建对比图
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, val_accs, width, label='Validation Accuracy', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_accs, width, label='Test Accuracy', color='coral', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    filename = "model_comparison.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_heatmap(data: np.ndarray,
                row_labels: List[str],
                col_labels: List[str],
                output_path: str,
                title: str = 'Heatmap',
                cmap: str = 'YlOrRd',
                fmt: str = '.3f'):
    """
    绘制热力图
    
    Args:
        data: 2D numpy数组
        row_labels: 行标签
        col_labels: 列标签
        output_path: 输出文件路径
        title: 图表标题
        cmap: 颜色映射
        fmt: 数值格式
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    
    # 设置刻度
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=10)
    
    # 添加数值
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title(title, fontsize=14, pad=20)
    
    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Accuracy', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def plot_heatmap_by_config(results: List[Dict],
                          split_config: str,
                          output_dir: str):
    """
    为每种划分配置绘制热力图
    
    Args:
        results: 结果列表
        split_config: 划分配置名称
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 筛选该划分配置的结果
    config_results = [r for r in results if split_config in r.get('config_name', '')]
    
    if not config_results:
        return
    
    # 提取预处理方法和模型类型
    preproc_methods = ['raw', 'body_only', 'subject_body', 'subject_body_author']
    model_names = sorted(list(set([r['model_name'] for r in config_results])))
    
    # 创建矩阵
    val_acc_matrix = np.zeros((len(model_names), len(preproc_methods)))
    test_acc_matrix = np.zeros((len(model_names), len(preproc_methods)))
    
    for result in config_results:
        model_name = result['model_name']
        preproc = result.get('preprocess', 'raw')
        
        if model_name in model_names and preproc in preproc_methods:
            i = model_names.index(model_name)
            j = preproc_methods.index(preproc)
            val_acc_matrix[i, j] = result['val_acc']
            test_acc_matrix[i, j] = result['test_acc']
    
    # 绘制验证集热力图
    val_output = os.path.join(output_dir, f"heatmap_val_{split_config}.png")
    plot_heatmap(val_acc_matrix, model_names, preproc_methods,
                val_output, f'Validation Accuracy - {split_config}')
    
    # 绘制测试集热力图
    test_output = os.path.join(output_dir, f"heatmap_test_{split_config}.png")
    plot_heatmap(test_acc_matrix, model_names, preproc_methods,
                test_output, f'Test Accuracy - {split_config}')


def plot_split_comparison(results_by_split: Dict[str, List[Dict]],
                         output_dir: str):
    """
    绘制不同划分配置的对比图
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有模型名称
    all_models = set()
    for results in results_by_split.values():
        for r in results:
            all_models.add(r['model_name'])
    model_names = sorted(list(all_models))
    
    # 绘制每个模型在不同划分下的表现
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(model_names))
    width = 0.25
    
    colors = ['steelblue', 'coral', 'forestgreen']
    split_names = list(results_by_split.keys())
    
    for i, (split_name, results) in enumerate(results_by_split.items()):
        # 计算该划分下每个模型的平均准确率
        model_accs = []
        for model in model_names:
            model_results = [r for r in results if r['model_name'] == model]
            if model_results:
                avg_acc = np.mean([r['val_acc'] for r in model_results])
                model_accs.append(avg_acc)
            else:
                model_accs.append(0)
        
        offset = (i - 1) * width
        ax.bar(x + offset, model_accs, width, label=split_name, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Average Validation Accuracy', fontsize=12)
    ax.set_title('Performance by Split Configuration', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename = "split_comparison.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def save_results_table(results: List[Dict], output_path: str):
    """
    保存结果为CSV表格
    """
    if not results:
        return
    
    # 转换为DataFrame
    df_data = []
    for r in results:
        df_data.append({
            'config_name': r.get('config_name', ''),
            'preprocess': r.get('preprocess', ''),
            'model_name': r['model_name'],
            'val_acc': r['val_acc'],
            'val_f1': r['val_f1'],
            'test_acc': r['test_acc'],
            'test_f1': r['test_f1'],
            'train_time': r.get('train_time', 0)
        })
    
    df = pd.DataFrame(df_data)
    df.to_csv(output_path, index=False)
    print(f"  Saved results table: {os.path.basename(output_path)}")


def plot_class_distribution(y_train, y_val, y_test, 
                           labels: List[str],
                           output_dir: str):
    """
    绘制类别分布对比图
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    datasets = [('Train', y_train), ('Validation', y_val), ('Test', y_test)]
    
    for ax, (name, y) in zip(axes, datasets):
        unique, counts = np.unique(y, return_counts=True)
        ax.bar(range(len(unique)), counts, color='steelblue', alpha=0.8)
        ax.set_xlabel('Class', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{name} Set Distribution', fontsize=12)
        ax.set_xticks(range(len(unique)))
        if len(unique) <= 10:
            ax.set_xticklabels(unique, fontsize=9)
    
    plt.tight_layout()
    filename = "class_distribution.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


if __name__ == '__main__':
    # 测试可视化功能
    output_dir = './test_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成测试数据
    results = [
        {'model_name': 'SVM_C0.1', 'val_acc': 0.85, 'test_acc': 0.83},
        {'model_name': 'SVM_C1', 'val_acc': 0.87, 'test_acc': 0.85},
        {'model_name': 'MLP_50', 'val_acc': 0.86, 'test_acc': 0.84},
    ]
    
    # 测试柱状图
    plot_model_comparison(results, output_dir, metric='val_acc')
    
    # 测试热力图
    data = np.random.rand(5, 4)
    labels = ['SVM_C0.1', 'SVM_C1', 'MLP_50', 'MLP_100', 'MLP_100_50']
    cols = ['raw', 'body_only', 'subject_body', 'subject_body_author']
    plot_heatmap(data, labels, cols, os.path.join(output_dir, 'test_heatmap.png'))
    
    print("Visualization test completed!")
