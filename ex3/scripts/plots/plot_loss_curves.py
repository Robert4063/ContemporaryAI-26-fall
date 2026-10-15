"""
绘制 Loss 曲线
- 每个(模型, 划分)一张图：不同超参不同线型，不同文本版本不同颜色
- MLP 版本对比图：固定 split_A, hidden=(100,50), 同一 alpha, 四个版本四种颜色
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import pandas as pd

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOSS_DIR = PROJECT_ROOT / "results" / "loss"
PLOT_DIR = PROJECT_ROOT / "results" / "evaluation" / "figures"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# 配置
VERSIONS = ["raw", "body_only", "subject_body", "subject_body_author"]
VERSION_COLORS = {
    "raw": "#1f77b4",  # blue
    "body_only": "#ff7f0e",  # orange
    "subject_body": "#2ca02c",  # green
    "subject_body_author": "#d62728",  # red
}

# SVM 参数
SVM_C_VALUES = [0.1, 1.0, 10.0]
SVM_LINESTYLES = {0.1: "-", 1.0: "--", 10.0: ":"}

# MLP 参数
MLP_HIDDEN_SIZES = [(50,), (100,), (100, 50)]
MLP_ALPHAS = [0.0001, 0.001]
MLP_HIDDEN_LINESTYLES = {
    (50,): "-",
    (100,): "--",
    (100, 50): ":"
}
MLP_ALPHA_LINESTYLES = {
    0.0001: "-",
    0.001: "--"
}

def load_loss(filename):
    """加载 loss 数据"""
    path = LOSS_DIR / f"{filename}.npy"
    if path.exists():
        return np.load(path)
    return None

def plot_svm_losses():
    """绘制 SVM Loss 曲线"""
    for split in ["split_A", "split_B", "split_C"]:
        for version in VERSIONS:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for C in SVM_C_VALUES:
                filename = f"svm_{split}_{version}_C{C}"
                loss = load_loss(filename)
                if loss is not None:
                    iterations = np.arange(len(loss))
                    ax.plot(iterations, loss, 
                           label=f"C={C}", 
                           linestyle=SVM_LINESTYLES[C],
                           linewidth=2)
            
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss (Hinge, log scale)")
            ax.set_yscale("log")
            ax.set_title(f"SVM - {split} ({version})")
            ax.legend()
            ax.grid(True, alpha=0.3, which='both')
            
            plot_path = PLOT_DIR / f"svm_{split}_{version}_loss.png"
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {plot_path}")

def plot_mlp_losses_by_split_version():
    """绘制 MLP Loss 曲线 - 按 split 和 version 分组"""
    for split in ["split_A", "split_B", "split_C"]:
        for version in VERSIONS:
            fig, ax = plt.subplots(figsize=(12, 7))
            
            for hidden in MLP_HIDDEN_SIZES:
                for alpha in MLP_ALPHAS:
                    hidden_str = f"h{hidden[0]}" if len(hidden) == 1 else f"h{'_'.join(map(str, hidden))}"
                    alpha_str = str(alpha)  # 保持原格式 0.0001 或 0.001
                    filename = f"mlp_{split}_{version}_{hidden_str}_a{alpha_str}"
                    
                    loss = load_loss(filename)
                    if loss is not None:
                        iterations = np.arange(len(loss))
                        label = f"{hidden_str}, α={alpha}"
                        linestyle = MLP_HIDDEN_LINESTYLES[hidden]
                        
                        ax.plot(iterations, loss, 
                               label=label, 
                               linestyle=linestyle,
                               linewidth=2)
            
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss (log scale)")
            ax.set_yscale("log")
            ax.set_title(f"MLP - {split} ({version})")
            ax.legend(fontsize=9, ncol=2)
            ax.grid(True, alpha=0.3, which='both')
            
            plot_path = PLOT_DIR / f"mlp_{split}_{version}_loss.png"
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {plot_path}")

def plot_mlp_version_comparison():
    """绘制 MLP 版本对比图
    固定 split_A, hidden=(100,50), 同一 alpha, 四个版本四种颜色
    """
    split = "split_A"
    hidden = (100, 50)
    hidden_str = "h100_50"
    
    for alpha in MLP_ALPHAS:
        alpha_str = str(alpha)  # 保持原格式 0.0001 或 0.001
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for version in VERSIONS:
            filename = f"mlp_{split}_{version}_{hidden_str}_a{alpha_str}"
            loss = load_loss(filename)
            if loss is not None:
                iterations = np.arange(len(loss))
                ax.plot(iterations, loss, 
                       label=version, 
                       color=VERSION_COLORS[version],
                       linewidth=2.5)
        
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss (log scale)")
        ax.set_yscale("log")
        ax.set_title(f"MLP Version Comparison - {split} (hidden={hidden}, alpha={alpha})")
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        
        plot_path = PLOT_DIR / f"mlp_{split}_version_comparison_h100_50_a{alpha_str}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {plot_path}")

def main():
    print("Generating Loss Curves...")
    print("\n=== SVM Loss Curves ===")
    plot_svm_losses()
    
    print("\n=== MLP Loss Curves (by split & version) ===")
    plot_mlp_losses_by_split_version()
    
    print("\n=== MLP Version Comparison ===")
    plot_mlp_version_comparison()
    
    print(f"\nAll plots saved to: {PLOT_DIR}")

if __name__ == "__main__":
    main()
