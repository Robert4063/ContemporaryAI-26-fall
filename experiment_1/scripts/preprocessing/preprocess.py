"""
Data preprocessing and splitting script for text classification experiment.

Functionality:
1. Load train data from data-Project1/train_data.csv
2. Load unlabeled data from data-Project1/test_data_unlabeled.csv
3. Split train data into train/val/test with stratified sampling
4. Extract 4 text versions: raw, body_only, subject_body, subject_body_author
5. Generate 3 split schemes: A (80/10/10), B (85/5/10), C (90/5/5)
6. Output processed data to data-processed/

CRITICAL: Data isolation is maintained throughout:
- Test split first, then validation from remaining
- All splits use random_state=42 for reproducibility
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data-Project1"
DATA_PROCESSED = PROJECT_ROOT / "data-processed"


# Header line patterns to identify email headers
HEADER_LINE_PATTERNS = [
    r"^From:",
    r"^Subject:",
    r"^Organization:",
    r"^Lines:",
    r"^Reply-To:",
    r"^NNTP-Posting-Host:",
    r"^Nntp-Posting-Host:",
    r"^Distribution:",
    r"^Newsgroups:",
    r"^Path:",
    r"^Message-ID:",
    r"^Date:",
    r"^X-.*:",
]

HEADER_REGEX = re.compile("|".join(HEADER_LINE_PATTERNS), re.IGNORECASE)


def extract_from_field(text):
    """Extract author name from 'From:' field."""
    match = re.search(r"^From:\s*([^\n]*?)\s*(?:\(|<|$)", text, re.MULTILINE)
    if match:
        author = match.group(1).strip()
        # Handle format like "name (email)" or "email (name)"
        author = re.sub(r"\s*\(.*?\)\s*", " ", author).strip()
        return author if author and author != "From:" else ""
    return ""


def extract_subject_field(text):
    """Extract text from 'Subject:' field."""
    match = re.search(r"^Subject:\s*([^\n]*)", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return ""


def split_email(text):
    """
    Split email into header and body.
    Header ends when we encounter an empty line after header lines.
    """
    lines = text.split("\n")
    header_lines = []
    body_lines = []
    found_blank = False
    
    for i, line in enumerate(lines):
        if not found_blank:
            if line.strip() == "":
                found_blank = True
            elif HEADER_REGEX.match(line):
                header_lines.append(line)
            else:
                # Non-header line before blank -> this is where header ends
                if header_lines:  # We've seen headers
                    found_blank = True
                    body_lines.append(line)
                else:  # No headers seen yet, treat as body
                    body_lines.append(line)
        else:
            body_lines.append(line)
    
    header_text = "\n".join(header_lines)
    body_text = "\n".join(body_lines).strip()
    
    return header_text, body_text


def extract_text_versions(text):
    """
    Extract 4 versions of text:
    - raw: original text
    - body_only: just body
    - subject_body: Subject + body
    - subject_body_author: From author + Subject + body
    """
    header, body = split_email(text)
    
    raw = text.strip()
    body_only = body
    
    subject = extract_subject_field(text)
    subject_body = (subject + "\n" + body).strip() if subject else body
    
    author = extract_from_field(text)
    if author:
        subject_body_author = (author + "\n" + subject + "\n" + body).strip() \
                             if subject else (author + "\n" + body).strip()
    else:
        subject_body_author = subject_body
    
    return {
        "raw": raw,
        "body_only": body_only,
        "subject_body": subject_body,
        "subject_body_author": subject_body_author,
    }


def perform_splits(df, random_state=42):
    """
    Perform 3 stratified splits:
    - split_A: 80/10/10 (train=5894, val=737, test=737)
    - split_B: 85/5/10 (train=6262, val=369, test=737) - SHARES test with A
    - split_C: 90/5/5 (train=6630, val=369, test=369)
    
    Return dict: split_name -> {'train': df, 'val': df, 'test': df}
    """
    splits = {}
    
    # Split A: 80/10/10
    # First: 90/10 split (train+val vs test) = (6631, 737)
    train_val_a, test_a = train_test_split(
        df, test_size=737/7368, random_state=random_state,
        stratify=df["target"]
    )
    # Then: 80/10 split of train_val = (5894, 737)
    train_a, val_a = train_test_split(
        train_val_a, test_size=737/6631, random_state=random_state,
        stratify=train_val_a["target"]
    )
    splits["split_A"] = {
        "train": train_a,
        "val": val_a,
        "test": test_a,
    }
    
    # Split B: 85/5/10 - SHARES test with split_A
    # Use same test set as split_A
    test_b = test_a.copy()
    remaining_b = train_val_a.copy()  # The 90% that wasn't in test_a
    # From remaining, split into 85/5 (train 6262 + val 369)
    # We want train:val = 85:5 = 17:1
    train_b, val_b = train_test_split(
        remaining_b, test_size=369/6631, random_state=random_state,
        stratify=remaining_b["target"]
    )
    splits["split_B"] = {
        "train": train_b,
        "val": val_b,
        "test": test_b,
    }
    
    # Split C: 90/5/5 - DIFFERENT test set
    # First: 95/5 split (train+val vs test)
    train_val_c, test_c = train_test_split(
        df, test_size=369/7368, random_state=random_state,
        stratify=df["target"]
    )
    # Then: split train+val into 90/5 (train 6630 + val 369)
    train_c, val_c = train_test_split(
        train_val_c, test_size=369/6999, random_state=random_state,
        stratify=train_val_c["target"]
    )
    splits["split_C"] = {
        "train": train_c,
        "val": val_c,
        "test": test_c,
    }
    
    return splits


def save_split_versions(splits, text_versions_dict, unlabeled_texts):
    """
    Save each split with each text version.
    
    For each split and version, save train/val/test CSVs with columns:
    - text: the processed text (no label column for unlabeled)
    - target: the label (not present in unlabeled)
    
    text_versions_dict: dict mapping original_index -> {version_name -> text}
    """
    version_names = ["raw", "body_only", "subject_body", "subject_body_author"]
    split_meta = []
    
    for split_name, split_data in splits.items():
        for version_name in version_names:
            split_dir = DATA_PROCESSED / split_name / version_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each subset
            for subset_name in ["train", "val", "test"]:
                subset_df = split_data[subset_name].copy()
                # IMPORTANT: subset_df has been reset with new indices 0-N
                # But we need the ORIGINAL indices from the split to access text_versions_dict
                
                # Get original indices
                original_indices = split_data[subset_name].index.tolist()
                
                # Map texts using original indices
                texts = [text_versions_dict[orig_idx][version_name] for orig_idx in original_indices]
                
                # Create output dataframe
                output_df = pd.DataFrame({
                    "text": texts,
                    "target": subset_df["target"].values
                })
                
                output_file = split_dir / f"{subset_name}.csv"
                output_df.to_csv(output_file, index=False, quoting=1)  # quoting=1 for QUOTE_ALL
                
                # Record metadata
                split_meta.append({
                    "split": split_name,
                    "version": version_name,
                    "subset": subset_name,
                    "count": len(output_df),
                })
            
            # Save unlabeled data (no target column)
            # unlabeled_texts is already in {idx -> {version -> text}} format
            unlabeled_indices = list(range(len(unlabeled_texts)))
            unlabeled_texts_list = [unlabeled_texts[idx][version_name] for idx in unlabeled_indices]
            
            unlabeled_df = pd.DataFrame({"text": unlabeled_texts_list})
            
            output_file = split_dir / "unlabeled.csv"
            unlabeled_df.to_csv(output_file, index=False, quoting=1)
            
            split_meta.append({
                "split": split_name,
                "version": version_name,
                "subset": "unlabeled",
                "count": len(unlabeled_df),
            })
    
    return pd.DataFrame(split_meta)


def save_readable_examples(splits, text_versions_dict):
    """
    Save readable example files showing Subject/Body structure.
    Format: "Subject: ... / Body: ..."
    This is for human inspection only, NOT used for training.
    """
    readable_dir = DATA_PROCESSED / "readable_examples"
    readable_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name in list(splits.keys())[:1]:  # Just save for split_A as example
        for subset_name in ["train"]:
            subset_indices = splits[split_name][subset_name].index
            examples = []
            
            for idx in subset_indices[:10]:  # First 10 examples
                text = text_versions_dict[idx]["raw"]
                subject = extract_subject_field(text)
                _, body = split_email(text)
                body_preview = body[:200].replace("\n", " ")
                
                examples.append(f"Subject: {subject}\nBody: {body_preview}...\n")
            
            output_file = readable_dir / f"{split_name}_{subset_name}_examples.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(examples))


def main():
    print("=" * 80)
    print("Starting data preprocessing...")
    print("=" * 80)
    
    # Load data
    print("\n[1/6] Loading data...")
    train_df = pd.read_csv(DATA_RAW / "train_data.csv")
    unlabeled_df = pd.read_csv(DATA_RAW / "test_data_unlabeled.csv")
    
    print(f"  Train data shape: {train_df.shape}")
    print(f"  Unlabeled data shape: {unlabeled_df.shape}")
    print(f"  Train targets: {sorted(train_df['target'].unique())}")
    
    # Extract text versions for train data
    print("\n[2/6] Extracting text versions for train data...")
    text_versions_train = {}
    for idx, row in train_df.iterrows():
        text_versions_train[idx] = extract_text_versions(row["text"])
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(train_df)}")
    
    # Extract text versions for unlabeled data
    print("\n[3/6] Extracting text versions for unlabeled data...")
    text_versions_unlabeled = {}
    for idx, row in unlabeled_df.iterrows():
        text_versions_unlabeled[idx] = extract_text_versions(row["text"])
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(unlabeled_df)}")
    
    # Perform stratified splits
    print("\n[4/6] Performing stratified splits...")
    splits = perform_splits(train_df, random_state=42)
    for split_name, split_data in splits.items():
        print(f"  {split_name}:")
        for subset_name, subset_df in split_data.items():
            print(f"    {subset_name}: {len(subset_df)}")
    
    # Save split versions
    print("\n[5/6] Saving split versions...")
    split_meta = save_split_versions(splits, text_versions_train, text_versions_unlabeled)
    
    # Save metadata
    print("\n[6/6] Saving metadata...")
    split_meta_summary = split_meta.groupby(["split", "subset"])["count"].sum().reset_index()
    split_meta_summary.to_csv(DATA_PROCESSED / "split_meta.csv", index=False)
    
    # Save readable examples
    save_readable_examples(splits, text_versions_train)
    
    print("\n" + "=" * 80)
    print("Preprocessing complete!")
    print(f"Output directory: {DATA_PROCESSED}")
    print("=" * 80)


if __name__ == "__main__":
    main()
