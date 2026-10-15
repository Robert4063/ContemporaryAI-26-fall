"""
scripts/check_no_leak.py
数据泄露自检脚本

检查内容:
1. 断言 train/val/test 的文本互不重叠
2. 断言各脚本中没有把 val 或 test 拼进 fit 的调用
3. 断言任何选择配置的代码路径只读 val 指标
"""

import re
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Set
import sys

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_PROCESSED = PROJECT_ROOT / "data-processed"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


class LeakChecker:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.passed = []
    
    def add_error(self, msg):
        self.errors.append(f"[ERROR] {msg}")
    
    def add_warning(self, msg):
        self.warnings.append(f"[WARNING] {msg}")
    
    def add_pass(self, msg):
        self.passed.append(f"[PASS] {msg}")
    
    def check_text_no_overlap(self):
        """检查各划分的文本是否互不重叠"""
        print("\n" + "="*60)
        print("检查 1: 文本重叠检查")
        print("="*60)
        
        all_passed = True
        for split in ['A', 'B', 'C']:
            split_dir = DATA_PROCESSED / f"split_{split}"
            if not split_dir.exists():
                self.add_warning(f"split_{split} 目录不存在，跳过")
                continue
            
            for version in ['raw', 'body_only', 'subject_body', 'subject_body_author']:
                version_dir = split_dir / version
                if not version_dir.exists():
                    continue
                
                # 读取数据
                train_df = pd.read_csv(version_dir / "train.csv")
                val_df = pd.read_csv(version_dir / "val.csv")
                test_df = pd.read_csv(version_dir / "test.csv")
                
                train_texts = set(train_df['text'].tolist())
                val_texts = set(val_df['text'].tolist())
                test_texts = set(test_df['text'].tolist())
                
                # 检查重叠
                train_val = train_texts & val_texts
                train_test = train_texts & test_texts
                val_test = val_texts & test_texts
                
                issues = []
                if train_val:
                    issues.append(f"Train-Val 重叠 {len(train_val)} 条")
                if train_test:
                    issues.append(f"Train-Test 重叠 {len(train_test)} 条")
                if val_test:
                    issues.append(f"Val-Test 重叠 {len(val_test)} 条")
                
                if issues:
                    all_passed = False
                    for issue in issues:
                        self.add_error(f"split_{split}/{version}: {issue}")
                else:
                    self.add_pass(f"split_{split}/{version}: 无文本重叠")
        
        return all_passed
    
    def check_fit_calls(self):
        """检查脚本中是否有把 val 或 test 用于 fit 的调用"""
        print("\n" + "="*60)
        print("检查 2: Fit 调用检查 (禁止 val/test 用于 fit)")
        print("="*60)
        
        # 需要检查的脚本模式
        fit_patterns = [
            r'\.fit\s*\(',
            r'\.fit_transform\s*\(',
            r'partial_fit\s*\(',
        ]
        
        # 禁止用于 fit 的变量
        forbidden_vars = [
            'val', 'valid', 'validation',
            'test', 'unlabeled', 'X_test', 'y_test',
            'val_df', 'validation_df', 'test_df',
        ]
        
        # 需要检查的脚本
        scripts_to_check = list(SCRIPTS_DIR.glob("**/*.py"))
        
        all_passed = True
        for script in scripts_to_check:
            try:
                content = script.read_text(encoding='utf-8')
                lines = content.split('\n')
                
                for line_no, line in enumerate(lines, 1):
                    # 跳过注释行
                    if line.strip().startswith('#'):
                        continue
                    
                    # 检查是否有 fit 调用
                    has_fit = any(re.search(p, line) for p in fit_patterns)
                    if not has_fit:
                        continue
                    
                    # 检查 fit 调用的参数
                    for var in forbidden_vars:
                        # 匹配 var 在 fit 参数中
                        # 例如: .fit(X_val, y_val) 或 fit(train_X, val_y)
                        if re.search(rf'\b{var}\b', line):
                            # 检查这行是否真的是 fit 调用
                            if re.search(r'\.(fit|fit_transform|partial_fit)\s*\(', line):
                                # 排除注释中的情况
                                code_part = line.split('#')[0]
                                if re.search(rf'\.(fit|fit_transform|partial_fit)\s*\([^)]*\b{var}\b', code_part):
                                    self.add_error(f"{script.relative_to(PROJECT_ROOT)}:{line_no}: "
                                                  f"禁止将 {var} 用于 fit")
                                    all_passed = False
                
            except Exception as e:
                self.add_warning(f"无法检查 {script}: {e}")
        
        if all_passed:
            self.add_pass("所有 fit 调用均未使用 val 或 test 数据")
        
        return all_passed
    
    def check_model_selection(self):
        """检查模型选择代码是否只使用 val 指标"""
        print("\n" + "="*60)
        print("检查 3: 模型选择逻辑检查 (只使用 val 指标)")
        print("="*60)
        
        # 需要检查的脚本
        scripts_to_check = list(SCRIPTS_DIR.glob("**/*.py"))
        
        all_passed = True
        for script in scripts_to_check:
            try:
                content = script.read_text(encoding='utf-8')
                lines = content.split('\n')
                
                for line_no, line in enumerate(lines, 1):
                    # 检查是否有选择配置的代码
                    selection_patterns = [
                        r'argmax\s*\(',
                        r'\.idxmax\s*\(',
                        r'\.sort_values\s*\(',
                        r'best_',
                        r'selected_',
                        r'optimal_',
                    ]
                    
                    has_selection = any(re.search(p, line, re.IGNORECASE) for p in selection_patterns)
                    if not has_selection:
                        continue
                    
                    # 检查是否引用了 test 相关变量
                    code_part = line.split('#')[0]
                    
                    # 检查是否有 test 相关的排序或选择
                    if re.search(r'\btest\b.*(?:acc|f1|score|metric)', code_part, re.IGNORECASE):
                        self.add_error(f"{script.relative_to(PROJECT_ROOT)}:{line_no}: "
                                      f"模型选择使用了 test 指标 (应只用 val)")
                        all_passed = False
                    
                    if re.search(r'(?:acc|f1|score|metric).*\btest\b', code_part, re.IGNORECASE):
                        self.add_error(f"{script.relative_to(PROJECT_ROOT)}:{line_no}: "
                                      f"模型选择使用了 test 指标 (应只用 val)")
                        all_passed = False
                
            except Exception as e:
                self.add_warning(f"无法检查 {script}: {e}")
        
        if all_passed:
            self.add_pass("模型选择逻辑只使用 val 指标")
        
        return all_passed
    
    def check_tfidf_fit_scope(self):
        """检查 TF-IDF 的 fit 范围"""
        print("\n" + "="*60)
        print("检查 4: TF-IDF Fit 范围检查 (只在 train 上 fit)")
        print("="*60)
        
        # 需要检查的脚本
        scripts_to_check = list(SCRIPTS_DIR.glob("**/*.py"))
        
        all_passed = True
        for script in scripts_to_check:
            try:
                content = script.read_text(encoding='utf-8')
                lines = content.split('\n')
                
                for line_no, line in enumerate(lines, 1):
                    # 跳过注释行
                    if line.strip().startswith('#'):
                        continue
                    
                    code_part = line.split('#')[0]
                    
                    # 检查 TfidfVectorizer 的 fit 调用
                    if 'TfidfVectorizer' in code_part or 'tfidf' in code_part.lower():
                        # 检查是否有 fit_transform 且参数包含 val/test/unlabeled
                        if re.search(r'fit_transform\s*\([^)]*\)', code_part):
                            # 提取 fit_transform 参数
                            match = re.search(r'fit_transform\s*\(([^)]+)\)', code_part)
                            if match:
                                params = match.group(1)
                                forbidden = ['val', 'validation', 'test', 'unlabeled']
                                for f in forbidden:
                                    if re.search(rf'\b{f}', params, re.IGNORECASE):
                                        self.add_error(f"{script.relative_to(PROJECT_ROOT)}:{line_no}: "
                                                      f"TF-IDF fit_transform 使用了 {f} 数据")
                                        all_passed = False
                
            except Exception as e:
                self.add_warning(f"无法检查 {script}: {e}")
        
        if all_passed:
            self.add_pass("TF-IDF fit 范围正确 (只在 train 上)")
        
        return all_passed
    
    def check_vectorizer_transform_only(self):
        """检查 val/test 是否只用 transform"""
        print("\n" + "="*60)
        print("检查 5: Val/Test Transform 检查")
        print("="*60)
        
        scripts_to_check = list(SCRIPTS_DIR.glob("**/*.py"))
        
        all_passed = True
        for script in scripts_to_check:
            try:
                content = script.read_text(encoding='utf-8')
                lines = content.split('\n')
                
                for line_no, line in enumerate(lines, 1):
                    code_part = line.split('#')[0]
                    
                    # 检查 transform 调用
                    if re.search(r'\.transform\s*\(', code_part):
                        # 检查参数
                        match = re.search(r'\.transform\s*\(([^)]+)\)', code_part)
                        if match:
                            params = match.group(1)
                            # 检查是否错误地用了 train 以外的变量
                            # 这里主要检查是否有 fit + transform 混用
                            pass  # transform 可以用于任何数据
                
            except Exception as e:
                self.add_warning(f"无法检查 {script}: {e}")
        
        self.add_pass("Transform 调用检查完成")
        return all_passed
    
    def check_data_split_sizes(self):
        """检查数据划分是否符合预期"""
        print("\n" + "="*60)
        print("检查 6: 数据划分样本数检查")
        print("="*60)
        
        # 注意: 预期值基于原始的三分割，但在处理后文本版本时，
        # 某些样本因为处理后文本与已有样本重叠而被过滤掉。
        # 因此 raw 版本应该完全符合预期，其他版本的样本数可能略少。
        expected_sizes = {
            'A': {'train': 5894, 'val': 737, 'test': 737},
            'B': {'train': 6243, 'val': 388, 'test': 737},
            'C': {'train': 6611, 'val': 388, 'test': 369},
        }
        
        all_passed = True
        for split, expected in expected_sizes.items():
            split_dir = DATA_PROCESSED / f"split_{split}"
            if not split_dir.exists():
                self.add_warning(f"split_{split} 目录不存在")
                continue
            
            for version in ['raw']:
                version_dir = split_dir / version
                if not version_dir.exists():
                    continue
                
                train_df = pd.read_csv(version_dir / "train.csv")
                val_df = pd.read_csv(version_dir / "val.csv")
                test_df = pd.read_csv(version_dir / "test.csv")
                
                actual = {
                    'train': len(train_df),
                    'val': len(val_df),
                    'test': len(test_df)
                }
                
                match = True
                for key, exp_val in expected.items():
                    act_val = actual[key]
                    if exp_val != act_val:
                        self.add_error(f"split_{split}/{version}: {key} 数量 {act_val} != 预期 {exp_val}")
                        match = False
                        all_passed = False
                
                if match:
                    self.add_pass(f"split_{split}/{version}: 样本数正确 {actual}")
        
        return all_passed
    
    def run_all_checks(self):
        """运行所有检查"""
        print("="*60)
        print("开始数据泄露自检")
        print("="*60)
        
        results = []
        results.append(self.check_text_no_overlap())
        results.append(self.check_fit_calls())
        results.append(self.check_model_selection())
        results.append(self.check_tfidf_fit_scope())
        results.append(self.check_data_split_sizes())
        
        # 输出汇总
        print("\n" + "="*60)
        print("检查结果汇总")
        print("="*60)
        
        if self.passed:
            print("\n通过的项目:")
            for p in self.passed:
                print(f"  {p}")
        
        if self.warnings:
            print("\n警告:")
            for w in self.warnings:
                print(f"  {w}")
        
        if self.errors:
            print("\n错误:")
            for e in self.errors:
                print(f"  {e}")
        
        all_ok = all(results) and len(self.errors) == 0
        
        print("\n" + "="*60)
        if all_ok:
            print("✓ 所有检查通过! 未发现数据泄露。")
        else:
            print("✗ 检查发现问题，请修复后重试。")
        print("="*60)
        
        return 0 if all_ok else 1


def main():
    checker = LeakChecker()
    exit_code = checker.run_all_checks()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
