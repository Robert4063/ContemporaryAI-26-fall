"""
scripts/preprocessing/preprocess.py
数据预处理脚本：生成三种划分和四种文本版本

划分规则:
- split_A: 80/10/10 (train 5894, val 737, test 737)
- split_B: 85/5/10 (train 6262, val 369, test 737)  
- split_C: 90/5/5  (train 6630, val 369, test 369)

文本版本:
- raw: 原始文本，不做任何清洗
- body_only: 去掉邮件头，只留正文
- subject_body: Subject + 正文
- subject_body_author: From 作者 + Subject + 正文

重要: TF-IDF 词表只在 train 上 fit，val 和 test 只做 transform
"""

import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_RAW = PROJECT_ROOT / "data-Project1"
DATA_PROCESSED = PROJECT_ROOT / "data-processed"

# 随机种子
RANDOM_STATE = 42

# 邮件头行匹配模式 (用于清洗邮件头)
HEADER_LINE_PATTERNS = [
    r'^From:\s*.*$',
    r'^Subject:\s*.*$',
    r'^Organization:\s*.*$',
    r'^Lines:\s*\d+$',
    r'^Reply-To:\s*.*$',
    r'^NNTP-Posting-Host:\s*.*$',
    r'^Nntp-Posting-Host:\s*.*$',
    r'^X-Newsreader:\s*.*$',
    r'^In-Reply-To:\s*.*$',
    r'^References:\s*.*$',
    r'^Message-ID:\s*.*$',
    r'^Date:\s*.*$',
    r'^From:\s*',
    r'^To:\s*.*$',
    r'^Cc:\s*.*$',
    r'^Bcc:\s*.*$',
    r'^Distribution:\s*.*$',
    r'^Origin:\s*.*$',
    r'^Paths:\s*.*$',
    r'^Newsgroups:\s*.*$',
    r'^Keywords:\s*.*$',
    r'^Summary:\s*.*$',
    r'^Expires:\s*.*$',
    r'^Followup-To:\s*.*$',
    r'^Sender:\s*.*$',
    r'^Reply-To:\s*.*$',
    r'^Precedence:\s*.*$',
    r'^Lines:\s*.*$',
]

# 编译正则表达式
HEADER_COMPILED = [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in HEADER_LINE_PATTERNS]


def extract_author(text):
    """从邮件头提取作者名"""
    # 匹配 From: 行的格式
    match = re.search(r'^From:\s*(.+?)(?:\n|$)', text, re.MULTILINE)
    if match:
        author_line = match.group(1).strip()
        # 提取邮件地址中的用户名或名字
        email_match = re.search(r'<([^>]+)>', author_line)
        if email_match:
            return email_match.group(1).split('@')[0]
        # 如果没有邮件地址，尝试提取名字
        name_match = re.search(r'^"?([^"<]+)"?', author_line)
        if name_match:
            return name_match.group(1).strip()
        return author_line[:50]  # 截断过长内容
    return ""


def extract_subject(text):
    """从邮件头提取主题"""
    match = re.search(r'^Subject:\s*(.+?)(?:\n|$)', text, re.MULTILINE | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def clean_header_lines(text):
    """移除邮件头行，保留正文"""
    result = text
    for pattern in HEADER_COMPILED:
        result = pattern.sub('', result)
    
    # 移除空行和多余的空白
    lines = result.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    return '\n'.join(lines)


def create_text_versions(text):
    """为一个文本创建四种版本"""
    author = extract_author(text)
    subject = extract_subject(text)
    body = clean_header_lines(text)
    
    return {
        'raw': text,
        'body_only': body,
        'subject_body': f"{subject}\n{body}",
        'subject_body_author': f"{author}\n{subject}\n{body}"
    }


def stratified_split(df, test_size, val_size, random_state):
    """
    分层抽样划分：先切 test，再从剩余数据里切 val
    
    参数:
        df: DataFrame (必须包含 'text' 和 'target' 列)
        test_size: 测试集比例
        val_size: 验证集比例 (相对于剩余数据)
        random_state: 随机种子
    
    返回:
        train_df, val_df, test_df
        
    注意: 此函数在原始文本级别进行划分，
         之后的文本处理不会导致测试集泄露问题
    """
    # 计算划分比例
    n = len(df)
    n_test = int(n * test_size)
    n_train_val = n - n_test
    n_val = int(n_train_val * val_size)
    n_train = n_train_val - n_val
    
    # 先切出 test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size,
        stratify=df['target'],
        random_state=random_state
    )
    
    # 再从剩余数据里切 val
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        stratify=train_val_df['target'],
        random_state=random_state
    )
    
    return train_df, val_df, test_df


def verify_splits(train_df, val_df, test_df, split_name, target_ratios):
    """验证划分的样本数和类别分布"""
    print(f"\n--- 验证 {split_name} 划分 ---")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"总计: {len(train_df) + len(val_df) + len(test_df)}")
    
    # 检查类别分布
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        dist = df['target'].value_counts().sort_index()
        print(f"\n{name} 类别分布:")
        for cls, count in dist.items():
            ratio = count / len(df) * 100
            print(f"  Class {cls}: {count} ({ratio:.1f}%)")
    
    # 检查是否有文本重叠
    train_texts = set(train_df['text'].tolist())
    val_texts = set(val_df['text'].tolist())
    test_texts = set(test_df['text'].tolist())
    
    train_val_overlap = train_texts & val_texts
    train_test_overlap = train_texts & test_texts
    val_test_overlap = val_texts & test_texts
    
    print(f"\n文本重叠检查:")
    print(f"  Train-Val 重叠: {len(train_val_overlap)}")
    print(f"  Train-Test 重叠: {len(train_test_overlap)}")
    print(f"  Val-Test 重叠: {len(val_test_overlap)}")
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("  [WARNING] 发现文本重叠!")
        return False
    else:
        print("  [OK] 无文本重叠")
        return True


def main():
    print("="*60)
    print("数据预处理开始")
    print("="*60)
    
    # 创建输出目录
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    # 读取原始数据
    print("\n读取原始数据...")
    train_raw = pd.read_csv(DATA_RAW / "train_data.csv")
    test_unlabeled = pd.read_csv(DATA_RAW / "test_data_unlabeled.csv")
    
    print(f"训练集样本数: {len(train_raw)}")
    print(f"测试集样本数: {len(test_unlabeled)}")
    print(f"类别数: {train_raw['target'].nunique()}")
    print(f"类别分布:\n{train_raw['target'].value_counts().sort_index()}")
    
    # 定义三种划分
    splits = {
        'A': {'test_size': 0.10, 'val_size': 0.10},  # 80/10/10
        'B': {'test_size': 0.10, 'val_size': 0.0526}, # 85/5/10 (5/(85+5))
        'C': {'test_size': 0.05, 'val_size': 0.0526}, # 90/5/5 (5/(90+5))
    }
    
    split_results = {}
    all_valid = True
    
    # 创建每种划分
    for split_name, params in splits.items():
        print(f"\n{'='*60}")
        print(f"创建划分: split_{split_name}")
        print(f"{'='*60}")
        
        # 分层划分
        train_df, val_df, test_df = stratified_split(
            train_raw,
            test_size=params['test_size'],
            val_size=params['val_size'],
            random_state=RANDOM_STATE
        )
        
        # 验证划分
        is_valid = verify_splits(train_df, val_df, test_df, split_name, None)
        all_valid = all_valid and is_valid
        
        # 保存划分信息
        split_results[split_name] = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    
    # 创建文本版本
    print("\n" + "="*60)
    print("创建文本版本")
    print("="*60)
    
    text_versions = ['raw', 'body_only', 'subject_body', 'subject_body_author']
    
    for split_name, dfs in split_results.items():
        print(f"\n处理 split_{split_name}...")
        
        # 创建版本子目录
        split_dir = DATA_PROCESSED / f"split_{split_name}"
        split_dir.mkdir(exist_ok=True)
        
        # 处理每种文本版本
        for version in text_versions:
            version_dir = split_dir / version
            version_dir.mkdir(exist_ok=True)
            
            print(f"  生成 {version} 版本...")
            
            # 处理训练集
            train_texts_processed = []
            train_texts_to_check = set()
            for text in dfs['train']['text']:
                versions = create_text_versions(text)
                processed_text = versions[version]
                train_texts_processed.append(processed_text)
                train_texts_to_check.add(processed_text)
            
            train_df_out = pd.DataFrame({
                'text': train_texts_processed,
                'target': dfs['train']['target'].values
            })
            
            # 处理验证集 - 检查与训练集是否重叠
            val_texts_processed = []
            val_texts_to_check = set()
            val_idx_keep = []
            for idx, text in enumerate(dfs['val']['text']):
                versions = create_text_versions(text)
                processed_text = versions[version]
                # 如果与训练集重叠，跳过此样本
                if processed_text in train_texts_to_check:
                    print(f"    [WARNING] Val 样本 {idx} 与 Train 重叠，跳过")
                    continue
                val_texts_processed.append(processed_text)
                val_texts_to_check.add(processed_text)
                val_idx_keep.append(idx)
            
            val_df_out = pd.DataFrame({
                'text': val_texts_processed,
                'target': dfs['val']['target'].iloc[val_idx_keep].values
            })
            
            # 处理测试集 - 检查与训练集和验证集是否重叠
            test_texts_processed = []
            test_texts_to_check = set()
            test_idx_keep = []
            for idx, text in enumerate(dfs['test']['text']):
                versions = create_text_versions(text)
                processed_text = versions[version]
                # 如果与训练集或验证集重叠，跳过此样本
                if processed_text in train_texts_to_check or processed_text in val_texts_to_check:
                    print(f"    [WARNING] Test 样本 {idx} 与 Train/Val 重叠，跳过")
                    continue
                test_texts_processed.append(processed_text)
                test_texts_to_check.add(processed_text)
                test_idx_keep.append(idx)
            
            test_df_out = pd.DataFrame({
                'text': test_texts_processed,
                'target': dfs['test']['target'].iloc[test_idx_keep].values
            })
            
            # 处理无标签测试集 (只有 text 列)
            test_unlabeled_processed = []
            for text in test_unlabeled['text']:
                versions = create_text_versions(text)
                test_unlabeled_processed.append(versions[version])
            
            test_unlabeled_df_out = pd.DataFrame({
                'text': test_unlabeled_processed
            })
            
            # 保存文件
            train_df_out.to_csv(version_dir / "train.csv", index=False)
            val_df_out.to_csv(version_dir / "val.csv", index=False)
            test_df_out.to_csv(version_dir / "test.csv", index=False)
            test_unlabeled_df_out.to_csv(version_dir / "unlabeled.csv", index=False)
            
            print(f"    保存到: {version_dir}/")
    
    # 生成 split_meta.csv
    print("\n" + "="*60)
    print("生成 split_meta.csv")
    print("="*60)
    
    meta_records = []
    for split_name, dfs in split_results.items():
        total = len(dfs['train']) + len(dfs['val']) + len(dfs['test'])
        for split_type, df in [('train', dfs['train']), ('val', dfs['val']), ('test', dfs['test'])]:
            meta_records.append({
                'split': split_name,
                'subset': split_type,
                'count': len(df),
                'ratio': len(df) / total
            })
    
    meta_df = pd.DataFrame(meta_records)
    meta_df.to_csv(DATA_PROCESSED / "split_meta.csv", index=False)
    print(meta_df.to_string())
    
    # 创建可读样例 (用于人工检查，不用于训练)
    print("\n" + "="*60)
    print("生成可读样例 (用于人工检查)")
    print("="*60)
    
    sample_dir = DATA_PROCESSED / "samples_for_inspection"
    sample_dir.mkdir(exist_ok=True)
    
    # 每个划分选一个样本
    for split_name, dfs in split_results.items():
        sample = {
            'text': dfs['train']['text'].iloc[0],
            'target': dfs['train']['target'].iloc[0]
        }
        
        # 添加人类可读的 Subject: / Body: 标注
        annotated = sample['text']
        # 在 Subject: 行后添加标记
        annotated = re.sub(
            r'(Subject:\s*[^\n]+)',
            r'\1 [SUBJECT_END]',
            annotated,
            flags=re.IGNORECASE | re.MULTILINE
        )
        annotated = annotated.replace('[SUBJECT_END]', '\n--- Subject End ---\n')
        annotated = '\n--- Body Start ---\n' + annotated
        
        # 保存
        with open(sample_dir / f"split_{split_name}_sample.txt", 'w', encoding='utf-8') as f:
            f.write(f"Target: {sample['target']}\n")
            f.write(f"{'='*60}\n\n")
            f.write(annotated)
    
    print(f"样例已保存到: {sample_dir}")
    
    # 汇总统计
    print("\n" + "="*60)
    print("预处理完成!")
    print("="*60)
    print(f"\n输出目录: {DATA_PROCESSED}")
    print(f"\n文件结构:")
    for split_name in split_results.keys():
        print(f"  split_{split_name}/")
        for version in text_versions:
            print(f"    {version}/")
            print(f"      train.csv, val.csv, test.csv, unlabeled.csv")
    
    print(f"\n总共生成了:")
    print(f"  3 种划分 × 4 种文本版本 × 4 个文件 = 48 个数据文件")
    print(f"  加上 split_meta.csv 和样例文件")
    
    if not all_valid:
        print("\n[WARNING] 部分划分的验证未通过，请检查!")
        sys.exit(1)


if __name__ == "__main__":
    main()
