"""
快速诊断作者提取问题
"""
import pandas as pd
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(ROOT / "data-Project1" / "train_data.csv")

def extract_author_v2(text: str) -> str:
    """从原始文本提取作者名，兼容多种 From: 格式"""
    if not isinstance(text, str):
        return 'UNKNOWN'
    lines = text.split('\n')
    for line in lines:
        if line.lower().startswith('from:'):
            content = line[5:].strip()
            # 格式1: Name <email@domain>
            m = re.match(r'^([^<]+)\s*<[^>]+>', content)
            if m:
                return m.group(1).strip()
            # 格式2: email@domain (Name)
            m = re.match(r'^[^@\s]+@[^@\s]+\s*\(([^)]+)\)', content)
            if m:
                return m.group(1).strip()
            # 格式3: 只有邮箱
            m = re.match(r'^([^@\s]+@[^@\s]+)$', content)
            if m:
                return m.group(1).strip()
            # 格式4: 纯名字
            if content and '@' not in content:
                return content.strip()[:50]
            return content.strip()[:50]
    return 'UNKNOWN'

df['author'] = df['text'].apply(extract_author_v2)

print(f'总文档: {len(df)}')
print(f'不同作者: {df["author"].nunique()}')
print(f'UNKNOWN 数量: {(df["author"] == "UNKNOWN").sum()} ({100*(df["author"]=="UNKNOWN").mean():.1f}%)')
print()
print('作者数量前 20:')
print(df['author'].value_counts().head(20))
