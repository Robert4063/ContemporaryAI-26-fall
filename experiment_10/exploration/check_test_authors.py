import pandas as pd, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(ROOT / "data-Project1" / "test_data_unlabeled.csv")

def extract_author(text):
    if not isinstance(text, str): return 'UNKNOWN'
    for line in text.split('\n'):
        if line.lower().startswith('from:'):
            content = line[5:].strip()
            m = re.match(r'^([^<]+)\s*<[^>]+>', content)
            if m: return m.group(1).strip()
            m = re.match(r'^[^@\s]+@[^@\s]+\s*\(([^)]+)\)', content)
            if m: return m.group(1).strip()
            m = re.match(r'^([^@\s]+@[^@\s]+)$', content)
            if m: return m.group(1).strip()
            if '@' not in content: return content.strip()[:50]
            return content.strip()[:50]
    return 'UNKNOWN'

authors = df['text'].apply(extract_author)
print(f'测试集总文档: {len(df)}')
print(f'不同作者数: {authors.nunique()}')
print(f'UNKNOWN: {(authors=="UNKNOWN").sum()} ({(authors=="UNKNOWN").mean()*100:.1f}%)')
print()
print('测试集作者分布 TOP 20:')
print(authors.value_counts().head(20))
