"""
Unified entry point for baseline training.

Usage:
    # Train all models, splits, versions
    python run_baseline.py
    
    # Train specific model on all splits/versions
    python run_baseline.py --model svm
    
    # Train specific split
    python run_baseline.py --split split_A
    
    # Train specific version
    python run_baseline.py --version raw
    
    # Combine multiple constraints
    python run_baseline.py --model svm --split split_A --version raw
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "baselines"


# All hyperparameter combinations
MODELS = ["svm", "mlp"]
SPLITS = ["split_A", "split_B", "split_C"]
VERSIONS = ["raw", "body_only", "subject_body", "subject_body_author"]

SVM_C_VALUES = [0.1, 1.0, 10.0]
MLP_HIDDEN_SIZES = [(50,), (100,), (100, 50)]
MLP_ALPHAS = [1e-4, 1e-3]


def get_svm_tasks():
    """Get all SVM training tasks."""
    tasks = []
    for split in SPLITS:
        for version in VERSIONS:
            for C in SVM_C_VALUES:
                tasks.append({
                    "model": "svm",
                    "split": split,
                    "version": version,
                    "C": C,
                })
    return tasks


def get_mlp_tasks():
    """Get all MLP training tasks."""
    tasks = []
    for split in SPLITS:
        for version in VERSIONS:
            for hidden in MLP_HIDDEN_SIZES:
                for alpha in MLP_ALPHAS:
                    hidden_str = ",".join(map(str, hidden))
                    tasks.append({
                        "model": "mlp",
                        "split": split,
                        "version": version,
                        "hidden": hidden_str,
                        "alpha": alpha,
                    })
    return tasks


def filter_tasks(tasks, model=None, split=None, version=None):
    """Filter tasks based on constraints."""
    filtered = tasks
    
    if model:
        filtered = [t for t in filtered if t["model"] == model]
    
    if split:
        filtered = [t for t in filtered if t["split"] == split]
    
    if version:
        filtered = [t for t in filtered if t["version"] == version]
    
    return filtered


def run_svm_task(task):
    """Execute SVM training task."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "train_svm.py"),
        "--split", task["split"],
        "--version", task["version"],
        "--C", str(task["C"]),
    ]
    
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Task failed with return code {e.returncode}")
        return False


def run_mlp_task(task):
    """Execute MLP training task."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "train_mlp.py"),
        "--split", task["split"],
        "--version", task["version"],
        "--hidden", task["hidden"],
        "--alpha", str(task["alpha"]),
    ]
    
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Task failed with return code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Unified baseline training entry point")
    parser.add_argument("--model", choices=["svm", "mlp", "all"], default="all",
                       help="Model to train (default: all)")
    parser.add_argument("--split", choices=SPLITS + ["all"], default="all",
                       help="Data split to use (default: all)")
    parser.add_argument("--version", choices=VERSIONS + ["all"], default="all",
                       help="Text version to use (default: all)")
    parser.add_argument("--sequential", action="store_true",
                       help="Run tasks sequentially (default: provide tasks for parallel execution)")
    
    args = parser.parse_args()
    
    # Normalize 'all' inputs
    model_filter = None if args.model == "all" else args.model
    split_filter = None if args.split == "all" else args.split
    version_filter = None if args.version == "all" else args.version
    
    # Get all tasks
    all_tasks = get_svm_tasks() + get_mlp_tasks()
    
    # Filter tasks
    filtered_tasks = filter_tasks(
        all_tasks,
        model=model_filter,
        split=split_filter,
        version=version_filter,
    )
    
    print(f"\n{'='*80}")
    print(f"Baseline Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print(f"Total tasks: {len(filtered_tasks)}")
    
    if len(filtered_tasks) > 20:
        print(f"\nWARNING: Large number of tasks!")
        print(f"  SVM tasks: {len([t for t in filtered_tasks if t['model'] == 'svm'])}")
        print(f"  MLP tasks: {len([t for t in filtered_tasks if t['model'] == 'mlp'])}")
        print(f"\nRECOMMENDATION: Use multiple terminals for parallel training!")
        print(f"  Terminal 1: python run_baseline.py --model svm --sequential")
        print(f"  Terminal 2: python run_baseline.py --model mlp --sequential")
    
    if args.sequential:
        # Sequential execution
        print(f"\nRunning {len(filtered_tasks)} tasks sequentially...\n")
        
        passed = 0
        failed = 0
        
        for i, task in enumerate(filtered_tasks, 1):
            task_desc = f"[{i}/{len(filtered_tasks)}]"
            
            if task["model"] == "svm":
                print(f"{task_desc} Running: SVM {task['split']} {task['version']} C={task['C']}")
                success = run_svm_task(task)
            else:
                print(f"{task_desc} Running: MLP {task['split']} {task['version']} h={task['hidden']} a={task['alpha']}")
                success = run_mlp_task(task)
            
            if success:
                passed += 1
                print(f"  ✓ Success")
            else:
                failed += 1
                print(f"  ✗ Failed")
        
        print(f"\n{'='*80}")
        print(f"Baseline Training Complete")
        print(f"  Passed: {passed}/{len(filtered_tasks)}")
        print(f"  Failed: {failed}/{len(filtered_tasks)}")
        print(f"{'='*80}\n")
        
    else:
        # List tasks (without executing)
        print(f"\nTo run tasks, execute them as separate processes:")
        print(f"  (For parallel execution across multiple terminals)\n")
        
        # Group tasks by model for easy terminal assignment
        svm_tasks = [t for t in filtered_tasks if t["model"] == "svm"]
        mlp_tasks = [t for t in filtered_tasks if t["model"] == "mlp"]
        
        if svm_tasks:
            print(f"SVM tasks ({len(svm_tasks)}):")
            print(f"  python run_baseline.py --model svm --sequential [--split SPLIT] [--version VERSION]")
            print()
        
        if mlp_tasks:
            print(f"MLP tasks ({len(mlp_tasks)}):")
            print(f"  python run_baseline.py --model mlp --sequential [--split SPLIT] [--version VERSION]")
            print()
        
        # List sample tasks
        print(f"Sample tasks:")
        for i, task in enumerate(filtered_tasks[:5]):
            if task["model"] == "svm":
                print(f"  {i+1}. SVM | {task['split']} | {task['version']} | C={task['C']}")
            else:
                print(f"  {i+1}. MLP | {task['split']} | {task['version']} | h={task['hidden']} | a={task['alpha']}")
        
        if len(filtered_tasks) > 5:
            print(f"  ... and {len(filtered_tasks) - 5} more tasks")
        
        print()


if __name__ == "__main__":
    main()
