"""
scripts/run_final_pipeline.py
训练完成后自动运行的最终流程：合并结果、数据泄露检查、绘图、生成预测
"""

import subprocess
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
PYTHON_EXE = "D:\\AI\\envs\\experiment_1\\python.exe"

def run_step(script_path, description):
    """运行一个步骤"""
    print("\n" + "="*60)
    print(f"[步骤] {description}")
    print("="*60)
    
    cmd = [PYTHON_EXE, str(script_path)]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    
    if result.returncode != 0:
        print(f"[ERROR] {description} 失败")
        return False
    
    print(f"[OK] {description} 完成")
    return True

def main():
    print("="*60)
    print("最终流程 (训练完成后)")
    print("="*60)
    
    steps = [
        (PROJECT_ROOT / "scripts" / "merge_results.py", "合并训练结果"),
        (PROJECT_ROOT / "scripts" / "check_no_leak.py", "数据泄露检查"),
        (PROJECT_ROOT / "scripts" / "plots" / "plot_loss.py", "绘制 Loss 曲线"),
        (PROJECT_ROOT / "scripts" / "plots" / "plot_evaluation.py", "绘制评估图表"),
        (PROJECT_ROOT / "scripts" / "baselines" / "generate_predictions.py", "生成最终预测"),
    ]
    
    results = []
    for script, desc in steps:
        if not script.exists():
            print(f"[SKIP] {desc} - 脚本不存在: {script}")
            continue
        
        success = run_step(script, desc)
        results.append((desc, success))
        
        if not success:
            print(f"\n[ERROR] 流程中断，请检查错误")
            break
        
        time.sleep(1)  # 短暂延迟
    
    # 汇总
    print("\n" + "="*60)
    print("流程汇总")
    print("="*60)
    
    for desc, success in results:
        status = "✓ 完成" if success else "✗ 失败"
        print(f"{status}: {desc}")
    
    all_ok = all(success for _, success in results)
    
    if all_ok:
        print("\n✓ 所有流程完成!")
        return 0
    else:
        print("\n✗ 部分流程失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
