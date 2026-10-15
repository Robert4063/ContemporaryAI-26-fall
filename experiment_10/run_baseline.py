"""
run_baseline.py
===============
统一训练入口脚本，运行所有 baseline 模型。

支持的模型:
    nb   - Naive Bayes (Multinomial)
    lr   - Logistic Regression
    svm  - Linear SVM
    dt   - Decision Tree
    mlp  - Multi-Layer Perceptron

用法:
    python run_baseline.py --all          # 训练所有模型
    python run_baseline.py --model nb     # 只训练 NB
    python run_baseline.py --model mlp --epochs 500  # 自定义迭代次数
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts" / "baselines"

MODELS = {
    "nb":  "train_nb.py",
    "lr":  "train_lr.py",
    "svm": "train_svm.py",
    "dt":  "train_dt.py",
    "mlp": "train_mlp.py",
}


def run(script_name: str, split: str = "all", version: str = "all"):
    """运行单个模型的训练脚本"""
    cmd = [sys.executable, str(SCRIPTS / script_name),
           "--split", split, "--version", version]
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] {script_name} failed with exit code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--model", default="all",
                        choices=list(MODELS.keys()) + ["all"])
    parser.add_argument("--split", default="all")
    parser.add_argument("--version", default="raw",
                        help="数据集版本: raw, body_only, subject_body, subject_body_author, all")
    args = parser.parse_args()

    # 确定要运行的模型
    if args.model == "all":
        models_to_run = MODELS
    else:
        models_to_run = {args.model: MODELS[args.model]}

    print(f"\n{'#'*60}")
    print(f"# Baseline 训练开始")
    print(f"# 模型: {list(models_to_run.keys())}")
    print(f"# 划分: {args.split}")
    print(f"# 数据版本: {args.version}")
    print(f"{'#'*60}")

    for name, script in models_to_run.items():
        success = run(script, args.split, args.version)
        if not success:
            print(f"\n[WARNING] {name} 训练失败，继续下一个模型...")

    print(f"\n{'='*60}")
    print("Baseline 训练完成！")
    print(f"结果保存在: results/baseline/")
    print(f"Loss 曲线保存在: results/loss/")
    print("=" * 60)

    # 提示下一步
    print("\n下一步:")
    print("  1. 查看结果: results/baseline/*.csv")
    print("  2. 保存最优模型: python scripts/baselines/train_save_models.py --all")
    print("  3. 使用模型预测: python scripts/baselines/predict_from_models.py --auto")


if __name__ == "__main__":
    main()
