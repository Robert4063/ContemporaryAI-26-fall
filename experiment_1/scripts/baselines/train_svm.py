"""
SVM baseline training script using SGDClassifier with hinge loss.

CRITICAL: This script ONLY fits on train data:
- TfidfVectorizer fitted on train, transformed for val/test
- SGDClassifier fitted on train, evaluated on val/test
- Alpha value computed from C and n_train

Usage:
    python train_svm.py --split split_A --version raw --C 1.0
"""

import argparse
import re
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data-processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"


# Alpha formula: alpha = 1 / (C * n_train)
# This is the ONLY place where alpha is defined for consistency
def compute_alpha(C, n_train):
    """
    Compute alpha value for SGDClassifier based on C value.
    Formula: alpha = 1 / (C * n_train)
    """
    return 1.0 / (C * n_train)


def load_data(split_name, version_name):
    """
    Load train, val, test data for given split and version.
    CRITICAL: Return original text, not pre-processed.
    """
    split_dir = DATA_PROCESSED / split_name / version_name
    
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")
    test_df = pd.read_csv(split_dir / "test.csv")
    
    return train_df, val_df, test_df


def train_svm(split_name, version_name, C=1.0, max_iter=300, random_state=42):
    """
    Train SVM model on train data, evaluate on val/test.
    
    Returns:
        results: dict with train/val/test metrics
        vectorizer: fitted TfidfVectorizer
        model: fitted SGDClassifier
        loss_history: list of loss values per epoch
    """
    
    print(f"\n{'='*80}")
    print(f"Training SVM | split={split_name} | version={version_name} | C={C}")
    print(f"{'='*80}")
    
    # Load data
    print(f"[1/5] Loading data...")
    train_df, val_df, test_df = load_data(split_name, version_name)
    X_train_text = train_df["text"].fillna("").values  # Replace NaN with empty string
    y_train = train_df["target"].values
    X_val_text = val_df["text"].fillna("").values
    y_val = val_df["target"].values
    X_test_text = test_df["text"].fillna("").values
    y_test = test_df["target"].values
    
    print(f"  Train: {len(X_train_text)} samples")
    print(f"  Val:   {len(X_val_text)} samples")
    print(f"  Test:  {len(X_test_text)} samples")
    
    # Fit TfidfVectorizer ONLY on train data
    print(f"[2/5] Fitting TfidfVectorizer on train data...")
    vectorizer = TfidfVectorizer(
        min_df=2,
        max_df=0.95,
        max_features=20000,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_val_tfidf = vectorizer.transform(X_val_text)  # ONLY transform, not fit
    X_test_tfidf = vectorizer.transform(X_test_text)  # ONLY transform, not fit
    
    vocab_size = len(vectorizer.get_feature_names_out())
    print(f"  Vocabulary size: {vocab_size}")
    
    # Compute alpha value
    n_train = len(X_train_text)
    alpha = compute_alpha(C, n_train)
    print(f"\n[3/5] Training SGDClassifier...")
    print(f"  C = {C}")
    print(f"  alpha = 1/(C*n_train) = 1/({C}*{n_train}) = {alpha:.10f}")
    print(f"  max_iter = {max_iter}")
    
    # Create and train model with warm_start to track progress
    model = SGDClassifier(
        loss="hinge",
        penalty="l2",
        alpha=alpha,
        max_iter=1,
        random_state=random_state,
        n_jobs=-1,
        warm_start=True,
        tol=None,
    )
    
    # Track loss history by training incrementally
    loss_history = []
    classes = np.unique(y_train)
    
    print(f"  Starting training...")
    for epoch in range(max_iter):
        # Fit one more iteration
        model.fit(X_train_tfidf, y_train)
        
        # Compute hinge loss manually
        decision = model.decision_function(X_train_tfidf)
        
        # Multi-class hinge loss: max(0, 1 - (score_correct - max_score_wrong))
        margins = np.zeros(len(y_train))
        for i, y_true in enumerate(y_train):
            correct_score = decision[i, y_true]
            # Get all other class scores
            other_indices = [j for j in range(len(classes)) if j != y_true]
            if other_indices:
                max_wrong = np.max(decision[i, other_indices])
                margins[i] = correct_score - max_wrong
            else:
                margins[i] = correct_score
        
        hinge_loss = np.maximum(0, 1 - margins).mean()
        loss_history.append(hinge_loss)
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{max_iter} | Hinge Loss: {hinge_loss:.6f}")
    
    print(f"  Training complete! Final loss: {loss_history[-1]:.6f}")
    
    # Evaluate on train/val/test
    print(f"\n[4/5] Evaluating on train/val/test...")
    
    y_train_pred = model.predict(X_train_tfidf)
    y_val_pred = model.predict(X_val_tfidf)
    y_test_pred = model.predict(X_test_tfidf)
    
    def compute_metrics(y_true, y_pred, set_name):
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        
        print(f"  {set_name}:")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Macro-F1:  {macro_f1:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        
        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "precision": precision,
            "recall": recall,
        }
    
    results = {
        "train": compute_metrics(y_train, y_train_pred, "TRAIN"),
        "val": compute_metrics(y_val, y_val_pred, "VAL"),
        "test": compute_metrics(y_test, y_test_pred, "TEST"),
    }
    
    # Get confusion matrix for test set
    cm = confusion_matrix(y_test, y_test_pred)
    results["confusion_matrix"] = cm
    
    print(f"\n[5/5] Saving results...")
    
    return results, vectorizer, model, loss_history, y_val_pred, y_test_pred


def save_results(split_name, version_name, C, results, vectorizer, model, loss_history):
    """
    Save results, model, and loss history.
    """
    
    # Save loss history
    loss_file = RESULTS_DIR / "loss" / f"svm_{split_name}_{version_name}_C{C}.npy"
    loss_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(loss_file, np.array(loss_history))
    print(f"  Loss history saved: {loss_file}")
    
    # Save model and vectorizer
    model_dir = MODELS_DIR / f"svm_{split_name}_{version_name}_C{C}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    with open(model_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # Save config
    config = {
        "model": "svm",
        "split": split_name,
        "version": version_name,
        "C": C,
        "alpha": compute_alpha(C, len(results["train"])),  # placeholder
        "max_iter": 300,
    }
    
    with open(model_dir / "config.pkl", "wb") as f:
        pickle.dump(config, f)
    
    print(f"  Model saved: {model_dir}")
    
    # Save metrics
    metrics_file = RESULTS_DIR / "baseline" / f"svm_{split_name}_{version_name}_C{C}_metrics.csv"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    metrics_data = []
    for split_type in ["train", "val"]:
        metrics_data.append({
            "model": "svm",
            "split": split_name,
            "version": version_name,
            "C": C,
            "eval_set": split_type,
            "accuracy": results[split_type]["accuracy"],
            "macro_f1": results[split_type]["macro_f1"],
            "precision": results[split_type]["precision"],
            "recall": results[split_type]["recall"],
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(metrics_file, index=False)
    print(f"  Metrics saved: {metrics_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="split_A", help="split_A, split_B, or split_C")
    parser.add_argument("--version", default="raw", help="raw, body_only, subject_body, or subject_body_author")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization parameter C")
    parser.add_argument("--max-iter", type=int, default=300, help="Max iterations")
    
    args = parser.parse_args()
    
    results, vectorizer, model, loss_history, _, _ = train_svm(
        args.split, args.version, C=args.C, max_iter=args.max_iter
    )
    
    save_results(args.split, args.version, args.C, results, vectorizer, model, loss_history)
    
    print(f"\n{'='*80}")
    print("Training complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
