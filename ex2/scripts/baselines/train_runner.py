"""
scripts/baselines/train_runner.py
训练运行器 - 支持并行调度多个训练任务

用法:
python train_runner.py --model svm --split A --version raw
python train_runner.py --model mlp --split all --version all
"""

import subprocess
import sys
from pathlib import Path
import argparse

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
PYTHON_EXE = "D:\\AI\\envs\\experiment_1\\python.exe"


def run_training(model, split, version, log_suffix=""):
    """运行单个训练任务"""
    if model == 'svm':
        script = PROJECT_ROOT / "scripts" / "baselines" / "train_svm.py"
    else:
        script = PROJECT_ROOT / "scripts" / "baselines" / "train_mlp.py"
    
    log_file = f"tmp/{model}_split_{split}_{version}{log_suffix}.log"
    
    cmd = [
        PYTHON_EXE, str(script),
        "--split", split,
        "--version", version
    ]
    
    print(f"\n>>> 运行: {' '.join(cmd)}")
    print(f"    日志: {log_file}")
    
    # 创建日志目录
    (PROJECT_ROOT / "tmp").mkdir(exist_ok=True)
    
    # 运行命令
    with open(PROJECT_ROOT / log_file, 'w', encoding='utf-8') as f:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=f,
            stderr=subprocess.STDOUT
        )
    
    if result.returncode != 0:
        print(f"[ERROR] 训练失败，日志: {log_file}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='训练运行器')
    parser.add_argument('--model', type=str, required=True,
                        choices=['svm', 'mlp'])
    parser.add_argument('--split', type=str, default='all',
                        choices=['A', 'B', 'C', 'all'])
    parser.add_argument('--version', type=str, default='all',
                        choices=['raw', 'body_only', 'subject_body', 'subject_body_author', 'all'])
    
    args = parser.parse_args()
    
    splits = ['A', 'B', 'C'] if args.split == 'all' else [args.split]
    versions = ['raw', 'body_only', 'subject_body', 'subject_body_author'] if args.version == 'all' else [args.version]
    
    total = len(splits) * len(versions)
    current = 0
    failed = []
    
    print(f"模型: {args.model}")
    print(f"划分: {splits}")
    print(f"版本: {versions}")
    print(f"总任务数: {total}")
    
    for split in splits:
        for version in versions:
            current += 1
            print(f"\n[{current}/{total}] 训练 {args.model} - split_{split} - {version}")
            
            success = run_training(args.model, split, version)
            if not success:
                failed.append((args.model, split, version))
    
    if failed:
        print(f"\n[WARNING] {len(failed)} 个任务失败:")
        for m, s, v in failed:
            print(f"  - {m}, split_{s}, {v}")
    else:
        print(f"\n[OK] 所有 {total} 个任务完成!")


if __name__ == "__main__":
    main()
