"""
scripts/wait_for_training.py  
等待所有训练完成
"""

import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
RESULTS_DIR = PROJECT_ROOT / "results" / "baseline"

# 期望的结果数
EXPECTED_SVM = 36  # 3 splits × 4 versions × 3 C values
EXPECTED_MLP = 72  # 3 splits × 4 versions × 3 hidden × 2 alpha

def count_results():
    svm = len(list((RESULTS_DIR / "svm").rglob("results_*.json"))) if (RESULTS_DIR / "svm").exists() else 0
    mlp = len(list((RESULTS_DIR / "mlp").rglob("results_*.json"))) if (RESULTS_DIR / "mlp").exists() else 0
    return svm, mlp

def main():
    print("等待所有训练完成...")
    print(f"期望结果: SVM {EXPECTED_SVM} + MLP {EXPECTED_MLP} = {EXPECTED_SVM + EXPECTED_MLP} 个")
    
    last_svm, last_mlp = 0, 0
    
    while True:
        svm, mlp = count_results()
        
        if svm != last_svm or mlp != last_mlp:
            print(f"\r[进行中] SVM: {svm}/{EXPECTED_SVM}, MLP: {mlp}/{EXPECTED_MLP}", end="", flush=True)
            last_svm, last_mlp = svm, mlp
        
        if svm >= EXPECTED_SVM and mlp >= EXPECTED_MLP:
            print(f"\n\n✓ 所有训练完成!")
            print(f"  SVM: {svm}/{EXPECTED_SVM}")
            print(f"  MLP: {mlp}/{EXPECTED_MLP}")
            return 0
        
        time.sleep(10)

if __name__ == "__main__":
    sys.exit(main())
