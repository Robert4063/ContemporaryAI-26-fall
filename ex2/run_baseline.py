"""
run_baseline.py - 统一入口脚本
用法: python run_baseline.py [--model {svm,mlp,all}] [--split {A,B,C,all}] [--version {raw,body_only,subject_body,subject_body_author,all}] [--step {preprocess,train,plot,predict,all}] [--check_leak]

示例:
    python run_baseline.py --step preprocess           # 数据预处理
    python run_baseline.py --model svm --split A --version raw   # SVM, 划分A, raw版本
    python run_baseline.py --model all --split all --version all # 所有组合
    python run_baseline.py --step plot                 # 绘制图表
    python run_baseline.py --step predict              # 生成预测
    python run_baseline.py --check_leak                # 数据泄露检查
"""

import argparse
import subprocess
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.resolve()


def run_command(cmd, description, log_file=None):
    """运行命令并记录日志"""
    print(f"\n{'='*60}")
    print(f"[{description}]")
    print(f"{'='*60}")
    
    if log_file:
        log_path = PROJECT_ROOT / "tmp" / log_file
        log_path.parent.mkdir(exist_ok=True)
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Command: {' '.join(cmd)}\n\n")
        cmd = cmd + [">>", str(log_path)]
        full_cmd = " ".join(cmd)
        result = subprocess.run(full_cmd, shell=True, cwd=str(PROJECT_ROOT))
    else:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    
    if result.returncode != 0:
        print(f"[ERROR] {description} failed with return code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="实验一: 英文文本10分类基线实验")
    parser.add_argument("--model", type=str, default="all", 
                        choices=["svm", "mlp", "all"],
                        help="模型类型 (默认: all)")
    parser.add_argument("--split", type=str, default="all",
                        choices=["A", "B", "C", "all"],
                        help="数据划分 (默认: all)")
    parser.add_argument("--version", type=str, default="all",
                        choices=["raw", "body_only", "subject_body", "subject_body_author", "all"],
                        help="文本版本 (默认: all)")
    parser.add_argument("--step", type=str, default="all",
                        choices=["preprocess", "train", "plot", "predict", "all", "check_leak"],
                        help="实验步骤 (默认: all)")
    parser.add_argument("--check_leak", action="store_true",
                        help="运行数据泄露检查")
    
    args = parser.parse_args()
    
    python_exe = "D:\\AI\\envs\\experiment_1\\python.exe"
    
    # Step 1: 数据预处理
    if args.step in ["preprocess", "all"]:
        print("\n" + "="*60)
        print("步骤 1: 数据预处理")
        print("="*60)
        preprocess_script = PROJECT_ROOT / "scripts" / "preprocessing" / "preprocess.py"
        cmd = [python_exe, str(preprocess_script)]
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            print("[ERROR] 数据预处理失败")
            sys.exit(1)
    
    # Step 2: 训练
    if args.step in ["train", "all"]:
        print("\n" + "="*60)
        print("步骤 2: 模型训练")
        print("="*60)
        
        train_script = PROJECT_ROOT / "scripts" / "baselines" / "train_runner.py"
        
        if args.model in ["svm", "all"]:
            print("\n>>> 训练 SVM 模型")
            cmd = [python_exe, str(train_script), "--model", "svm"]
            if args.split != "all":
                cmd.extend(["--split", args.split])
            if args.version != "all":
                cmd.extend(["--version", args.version])
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if result.returncode != 0:
                print("[ERROR] SVM 训练失败")
                sys.exit(1)
        
        if args.model in ["mlp", "all"]:
            print("\n>>> 训练 MLP 模型")
            cmd = [python_exe, str(train_script), "--model", "mlp"]
            if args.split != "all":
                cmd.extend(["--split", args.split])
            if args.version != "all":
                cmd.extend(["--version", args.version])
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if result.returncode != 0:
                print("[ERROR] MLP 训练失败")
                sys.exit(1)
    
    # Step 3: 绘制图表
    if args.step in ["plot", "all"]:
        print("\n" + "="*60)
        print("步骤 3: 绘制图表")
        print("="*60)
        
        # 绘制 loss 曲线
        plot_loss_script = PROJECT_ROOT / "scripts" / "plots" / "plot_loss.py"
        cmd = [python_exe, str(plot_loss_script)]
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            print("[ERROR] Loss 曲线绘制失败")
            sys.exit(1)
        
        # 绘制评估图表
        plot_eval_script = PROJECT_ROOT / "scripts" / "plots" / "plot_evaluation.py"
        cmd = [python_exe, str(plot_eval_script)]
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            print("[ERROR] 评估图表绘制失败")
            sys.exit(1)
    
    # Step 4: 生成预测
    if args.step in ["predict", "all"]:
        print("\n" + "="*60)
        print("步骤 4: 生成最终预测")
        print("="*60)
        
        predict_script = PROJECT_ROOT / "scripts" / "baselines" / "generate_predictions.py"
        cmd = [python_exe, str(predict_script)]
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            print("[ERROR] 预测生成失败")
            sys.exit(1)
    
    # Step 5: 数据泄露检查
    if args.check_leak or args.step == "check_leak":
        print("\n" + "="*60)
        print("步骤 5: 数据泄露检查")
        print("="*60)
        
        check_leak_script = PROJECT_ROOT / "scripts" / "check_no_leak.py"
        cmd = [python_exe, str(check_leak_script)]
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            print("[WARNING] 数据泄露检查发现问题，请查看输出")
        else:
            print("[OK] 数据泄露检查通过")
    
    print("\n" + "="*60)
    print("全部步骤完成!")
    print("="*60)


if __name__ == "__main__":
    main()
