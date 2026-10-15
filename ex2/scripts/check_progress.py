"""
scripts/check_progress.py
检查训练进度
"""

from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
RESULTS_DIR = PROJECT_ROOT / "results" / "baseline"

def check_progress():
    print("="*60)
    print("训练进度检查")
    print("="*60)
    
    # 统计 SVM 结果
    svm_results = list((RESULTS_DIR / "svm").rglob("results_*.json")) if (RESULTS_DIR / "svm").exists() else []
    
    # 统计 MLP 结果
    mlp_results = list((RESULTS_DIR / "mlp").rglob("results_*.json")) if (RESULTS_DIR / "mlp").exists() else []
    
    print(f"\nSVM 结果: {len(svm_results)} / 36 (3 划分 × 4 版本 × 3 C值)")
    print(f"MLP 结果: {len(mlp_results)} / 72 (3 划分 × 4 版本 × 3 hidden × 2 alpha)")
    print(f"\n总进度: {len(svm_results) + len(mlp_results)} / 108")
    
    # 按划分和版本统计
    splits = ['A', 'B', 'C']
    versions = ['raw', 'body_only', 'subject_body', 'subject_body_author']
    
    print("\n详细进度:")
    for model in ['svm', 'mlp']:
        print(f"\n{model.upper()}:")
        for split in splits:
            for version in versions:
                split_dir = RESULTS_DIR / model / f"split_{split}" / version
                if split_dir.exists():
                    count = len(list(split_dir.glob("results_*.json")))
                    expected = 3 if model == 'svm' else 6
                    status = "✓" if count == expected else f"{count}/{expected}"
                    print(f"  split_{split}/{version}: {status}")
                else:
                    print(f"  split_{split}/{version}: 0/{3 if model == 'svm' else 6}")

if __name__ == "__main__":
    check_progress()
