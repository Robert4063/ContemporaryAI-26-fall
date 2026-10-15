"""
scripts/merge_results.py
合并训练结果脚本

将分散在各配置目录的结果合并为汇总 CSV
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
RESULTS_DIR = PROJECT_ROOT / "results" / "baseline"


def merge_model_results(model_name):
    """合并单个模型的所有结果"""
    model_dir = RESULTS_DIR / model_name
    
    if not model_dir.exists():
        print(f"[WARNING] {model_name} 结果目录不存在: {model_dir}")
        return None
    
    results = []
    
    # 遍历所有划分和版本
    for split_dir in model_dir.glob("split_*"):
        if not split_dir.is_dir():
            continue
        
        split_name = split_dir.name.replace("split_", "")
        
        for version_dir in split_dir.iterdir():
            if not version_dir.is_dir():
                continue
            
            version_name = version_dir.name
            
            # 查找结果文件
            for result_file in version_dir.glob("results_*.json"):
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    config = data['config']
                    train_metrics = data['train_metrics']
                    val_metrics = data['val_metrics']
                    test_metrics = data['test_metrics']
                    
                    row = {
                        'model': config['model'],
                        'split': config['split'],
                        'version': config['version'],
                        'n_train': config['n_train'],
                        'vocab_size': config['vocab_size'],
                        'train_accuracy': train_metrics['accuracy'],
                        'train_f1_macro': train_metrics['f1_macro'],
                        'val_accuracy': val_metrics['accuracy'],
                        'val_f1_macro': val_metrics['f1_macro'],
                        'test_accuracy': test_metrics['accuracy'],
                        'test_f1_macro': test_metrics['f1_macro'],
                        'test_precision': test_metrics['precision'],
                        'test_recall': test_metrics['recall']
                    }
                    
                    # 添加模型特定参数
                    if model_name == 'svm':
                        row['C'] = config['C']
                        row['alpha'] = config['alpha']
                    elif model_name == 'mlp':
                        row['hidden_layer_sizes'] = str(config['hidden_layer_sizes'])
                        row['alpha'] = config['alpha']
                        row['n_iter'] = config.get('n_iter', 'N/A')
                    
                    results.append(row)
                    
                except Exception as e:
                    print(f"[ERROR] 读取结果文件失败: {result_file}")
                    print(f"  错误: {e}")
    
    if results:
        df = pd.DataFrame(results)
        return df
    return None


def merge_all_results():
    """合并所有模型结果"""
    print("="*60)
    print("合并训练结果")
    print("="*60)
    
    all_results = []
    
    for model in ['svm', 'mlp']:
        print(f"\n处理 {model}...")
        df = merge_model_results(model)
        if df is not None and len(df) > 0:
            all_results.append(df)
            print(f"  找到 {len(df)} 条结果")
        else:
            print(f"  无结果")
    
    if all_results:
        merged_df = pd.concat(all_results, ignore_index=True)
        
        # 保存合并结果
        output_file = RESULTS_DIR / "merged_results.csv"
        merged_df.to_csv(output_file, index=False)
        print(f"\n合并结果已保存到: {output_file}")
        print(f"总计 {len(merged_df)} 条结果")
        
        return merged_df
    else:
        print("\n[WARNING] 没有找到任何结果")
        return None


def select_best_configs():
    """根据验证集指标选择最优配置"""
    print("\n" + "="*60)
    print("选择最优配置")
    print("="*60)
    
    merged_file = RESULTS_DIR / "merged_results.csv"
    
    if not merged_file.exists():
        print("[ERROR] 合并结果文件不存在，请先运行 merge_results.py")
        return None
    
    df = pd.read_csv(merged_file)
    
    best_configs = []
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        # 按 val_accuracy 排序，并列时按 val_f1_macro
        model_df = model_df.sort_values(
            by=['val_accuracy', 'val_f1_macro'],
            ascending=[False, False]
        )
        
        best = model_df.iloc[0]
        best_configs.append(best)
    
    best_df = pd.DataFrame(best_configs)
    
    # 保存最优配置
    output_file = RESULTS_DIR / "best_config_summary.csv"
    best_df.to_csv(output_file, index=False)
    
    print(f"\n最优配置已保存到: {output_file}")
    print("\n最优配置汇总:")
    print(best_df[['model', 'split', 'version', 'val_accuracy', 'val_f1_macro', 
                   'test_accuracy', 'test_f1_macro']].to_string())
    
    return best_df


def main():
    # 合并结果
    merged_df = merge_all_results()
    
    if merged_df is not None:
        # 选择最优配置
        select_best_configs()
    
    print("\n完成!")


if __name__ == "__main__":
    main()
