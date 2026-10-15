#!/usr/bin/env python
"""
并行运行所有训练任务
"""
import subprocess
import concurrent.futures
from pathlib import Path
import time

PYTHON_EXE = r'D:\AI\envs\experiment_1\python.exe'
PROJECT_ROOT = Path.cwd()
SPLITS = ['A', 'B', 'C']
VERSIONS = ['raw', 'body_only', 'subject_body', 'subject_body_author']
MODELS = ['svm', 'mlp']

# 为 SVM 和 MLP 创建不同的 C/alpha 值
SVM_PARAMS = [
    {'C': 0.1},
    {'C': 1.0},
    {'C': 10.0},
]

MLP_PARAMS = [
    {'hidden': 50, 'alpha': 0.0001},
    {'hidden': 50, 'alpha': 0.001},
    {'hidden': 100, 'alpha': 0.0001},
    {'hidden': 100, 'alpha': 0.001},
    {'hidden': 200, 'alpha': 0.0001},
    {'hidden': 200, 'alpha': 0.001},
]

def run_task(task):
    """运行单个训练任务"""
    cmd = [
        PYTHON_EXE,
        str(task['script']),
        '--split', task['split'],
        '--version', task['version']
    ]
    
    # 添加模型特定参数
    if 'c_value' in task:
        cmd.extend(['--C', str(task['c_value'])])
    if 'hidden_size' in task:
        cmd.extend(['--hidden_size', str(task['hidden_size'])])
    if 'alpha' in task:
        cmd.extend(['--alpha', str(task['alpha'])])
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=3600
    )
    
    return result.returncode == 0, task, result.stderr

# 构建任务列表
tasks = []

for split in SPLITS:
    for version in VERSIONS:
        # SVM 任务
        for c_idx, params in enumerate(SVM_PARAMS, 1):
            tasks.append({
                'model': 'svm',
                'split': split,
                'version': version,
                'script': PROJECT_ROOT / 'scripts' / 'baselines' / 'train_svm.py',
                'c_value': params['C'],
                'param_idx': c_idx,
            })
        
        # MLP 任务
        for mlp_idx, params in enumerate(MLP_PARAMS, 1):
            tasks.append({
                'model': 'mlp',
                'split': split,
                'version': version,
                'script': PROJECT_ROOT / 'scripts' / 'baselines' / 'train_mlp.py',
                'hidden_size': params['hidden'],
                'alpha': params['alpha'],
                'param_idx': mlp_idx,
            })

print(f'总任务数: {len(tasks)}')
print(f'SVM: {len(SPLITS)} × {len(VERSIONS)} × {len(SVM_PARAMS)} = {len(SPLITS) * len(VERSIONS) * len(SVM_PARAMS)}')
print(f'MLP: {len(SPLITS)} × {len(VERSIONS)} × {len(MLP_PARAMS)} = {len(SPLITS) * len(VERSIONS) * len(MLP_PARAMS)}')
print()

start_time = time.time()
completed = 0
failed = 0
errors = []

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(run_task, task) for task in tasks]
    
    for future in concurrent.futures.as_completed(futures):
        try:
            success, task, stderr = future.result()
            if success:
                completed += 1
                status = '✓'
            else:
                failed += 1
                status = '✗'
                errors.append((task, stderr))
            
            total_done = completed + failed
            print(f'[{total_done:3}/{len(tasks)}] {status} {task["model"]:3} split_{task["split"]} {task["version"]:16} param_{task["param_idx"]}')
        except Exception as e:
            failed += 1
            print(f'[ERROR] {e}')
            errors.append((None, str(e)))

elapsed = time.time() - start_time
print(f'\n完成: {completed}/{len(tasks)} ({failed} 失败)')
print(f'耗时: {elapsed:.1f} 秒')

if errors and len(errors) <= 5:
    print(f'\n错误详情:')
    for task, stderr in errors[:5]:
        if task:
            print(f"\n{task['model']} split_{task['split']} {task['version']}:")
        if stderr:
            print(stderr[:500])
