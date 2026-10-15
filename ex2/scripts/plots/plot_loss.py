"""
scripts/plots/plot_loss.py
Loss 曲线绘制脚本

输出:
- 每个 (模型, 划分) 一张 loss 图
- MLP 版本对比图 (固定划分A, hidden=(100,50))
"""

import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
from pathlib import Path
import numpy as np

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
LOSS_DIR = PROJECT_ROOT / "results" / "loss"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# 颜色方案
VERSION_COLORS = {
    'raw': '#1f77b4',
    'body_only': '#ff7f0e',
    'subject_body': '#2ca02c',
    'subject_body_author': '#d62728'
}

VERSION_LABELS = {
    'raw': 'Raw',
    'body_only': 'Body Only',
    'subject_body': 'Subject+Body',
    'subject_body_author': 'Subject+Body+Author'
}


def plot_model_split_loss(model, split):
    """绘制单个 (模型, 划分) 的 loss 图"""
    split_dir = LOSS_DIR / model / f"split_{split}"
    
    if not split_dir.exists():
        print(f"[WARNING] 目录不存在: {split_dir}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    has_data = False
    for version_dir in split_dir.iterdir():
        if not version_dir.is_dir():
            continue
        
        version = version_dir.name
        
        for loss_file in version_dir.glob("loss_*.json"):
            try:
                with open(loss_file, 'r') as f:
                    data = json.load(f)
                
                losses = data['train_losses']
                config = data.get('config', {})
                
                # 获取超参标签
                if model == 'svm':
                    C = config.get('C', '')
                    label = f"{VERSION_LABELS.get(version, version)} (C={C})"
                else:
                    hidden = config.get('hidden_layer_sizes', '')
                    alpha = config.get('alpha', '')
                    label = f"{VERSION_LABELS.get(version, version)} (h={hidden}, α={alpha})"
                
                color = VERSION_COLORS.get(version, '#888888')
                epochs = range(1, len(losses) + 1)
                
                ax.plot(epochs, losses, label=label, color=color, alpha=0.8, linewidth=1.5)
                has_data = True
                
            except Exception as e:
                print(f"[ERROR] 读取 loss 文件失败: {loss_file}")
                print(f"  错误: {e}")
    
    if not has_data:
        print(f"[WARNING] 无数据可绘制: model={model}, split={split}")
        plt.close()
        return
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'{model.upper()} - Split {split} Training Loss', fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 如果 loss 跨多个数量级，使用对数坐标
    all_losses = []
    for loss_file in split_dir.glob("*/*/loss_*.json"):
        try:
            with open(loss_file, 'r') as f:
                data = json.load(f)
                all_losses.extend(data['train_losses'])
        except:
            pass
    
    if all_losses:
        loss_min, loss_max = min(all_losses), max(all_losses)
        if loss_max / loss_min > 100:
            ax.set_yscale('log')
            ax.set_ylabel('Loss (log scale)', fontsize=12)
    
    plt.tight_layout()
    
    output_file = PLOTS_DIR / f"loss_{model}_split_{split}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] 保存 loss 图: {output_file}")


def plot_mlp_version_comparison():
    """绘制 MLP 版本对比图 (固定 split A, hidden=(100,50), alpha=1e-3)"""
    print("\n" + "="*60)
    print("绘制 MLP 版本对比图")
    print("="*60)
    
    split = 'A'
    model = 'mlp'
    target_hidden = (100, 50)
    target_alpha = 0.001
    
    split_dir = LOSS_DIR / model / f"split_{split}"
    
    if not split_dir.exists():
        print(f"[WARNING] 目录不存在: {split_dir}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for version_dir in split_dir.iterdir():
        if not version_dir.is_dir():
            continue
        
        version = version_dir.name
        
        for loss_file in version_dir.glob("loss_*.json"):
            try:
                with open(loss_file, 'r') as f:
                    data = json.load(f)
                
                config = data.get('config', {})
                
                # 筛选目标配置
                if tuple(config.get('hidden_layer_sizes', ())) != target_hidden:
                    continue
                if abs(config.get('alpha', 0) - target_alpha) > 1e-6:
                    continue
                
                losses = data['train_losses']
                color = VERSION_COLORS.get(version, '#888888')
                epochs = range(1, len(losses) + 1)
                
                ax.plot(epochs, losses, 
                       label=VERSION_LABELS.get(version, version),
                       color=color, linewidth=2)
                
            except Exception as e:
                pass
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('MLP Version Comparison (Split A, hidden=(100,50), α=0.001)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 检查是否需要对数坐标
    plt.tight_layout()
    
    output_file = PLOTS_DIR / "mlp_version_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] 保存版本对比图: {output_file}")


def main():
    print("="*60)
    print("绘制 Loss 曲线")
    print("="*60)
    
    for model in ['svm', 'mlp']:
        for split in ['A', 'B', 'C']:
            print(f"\n处理 {model} / split_{split}...")
            plot_model_split_loss(model, split)
    
    # MLP 版本对比图
    plot_mlp_version_comparison()
    
    print("\n" + "="*60)
    print("Loss 曲线绘制完成!")
    print(f"输出目录: {PLOTS_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
