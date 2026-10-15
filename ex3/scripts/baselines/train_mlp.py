"""
MLP baseline training script using MLPClassifier.

CRITICAL: This script ONLY fits on train data:
- TfidfVectorizer fitted on train, transformed for val/test
- MLPClassifier fitted on train, evaluated on val/test
- No early stopping, full 300 iterations

Usage:
    python train_mlp.py --split split_A --version raw --hidden 100 --alpha 0.001
"""

import argparse
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data-processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"


def load_data(split_name, version_name):
    """
    Load train, val, test data for given split and version.
    """
    split_dir = DATA_PROCESSED / split_name / version_name
    
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")
    test_df = pd.read_csv(split_dir / "test.csv")
    
    return train_df, val_df, test_df


def parse_hidden_layers(hidden_str):
    """
    Parse hidden layer string like "100" or "100,50" into tuple.
    """
    if "," in hidden_str:
        return tuple(map(int, hidden_str.split(",")))
    else:
        return (int(hidden_str),)


def train_mlp(split_name, version_name, hidden_layer_sizes=(100,), alpha=0.001, 
              max_iter=300, random_state=42):
    """
    Train MLP model on train data, evaluate on val/test.
    
    Returns:
        results: dict with train/val/test metrics
        vectorizer: fitted TfidfVectorizer
        model: fitted MLPClassifier
        loss_history: list of loss values per iteration
    """
    
    print(f"\n{'='*80}")
    print(f"Training MLP | split={split_name} | version={version_name}")
    print(f"            | hidden={hidden_layer_sizes} | alpha={alpha}")
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
    
    print(f"\n[3/5] Training MLPClassifier...")
    print(f"  hidden_layer_sizes = {hidden_layer_sizes}")
    print(f"  alpha = {alpha}")
    print(f"  max_iter = {max_iter}")
    print(f"  early_stopping = False")
    print(f"  tol = 0")
    print(f"  n_iter_no_change = {max_iter + 1}")
    
    # Create and train model
    # CRITICAL: No early stopping to ensure full loss curve
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=False,
        tol=0,
        n_iter_no_change=max_iter + 1,
        verbose=True,
    )
    
    # Train
    model.fit(X_train_tfidf, y_train)
    
    # Get loss history (MLP stores this automatically)
    loss_history = model.loss_curve_
    
    print(f"  Training complete!")
    print(f"  Total iterations: {model.n_iter_}")
    print(f"  Final loss: {loss_history[-1]:.6f}")
    
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


def save_results(split_name, version_name, hidden_layer_sizes, alpha, 
                 results, vectorizer, model, loss_history):
    """
    Save results, model, and loss history.
    """
    
    hidden_str = "_".join(map(str, hidden_layer_sizes))
    
    # Save loss history
    loss_file = RESULTS_DIR / "loss" / f"mlp_{split_name}_{version_name}_h{hidden_str}_a{alpha}.npy"
    loss_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(loss_file, np.array(loss_history))
    print(f"  Loss history saved: {loss_file}")
    
    # Save model and vectorizer
    model_dir = MODELS_DIR / f"mlp_{split_name}_{version_name}_h{hidden_str}_a{alpha}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    with open(model_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # Save config
    config = {
        "model": "mlp",
        "split": split_name,
        "version": version_name,
        "hidden_layer_sizes": hidden_layer_sizes,
        "alpha": alpha,
        "max_iter": 300,
    }
    
    with open(model_dir / "config.pkl", "wb") as f:
        pickle.dump(config, f)
    
    print(f"  Model saved: {model_dir}")
    
    # Save metrics
    metrics_file = RESULTS_DIR / "baseline" / f"mlp_{split_name}_{version_name}_h{hidden_str}_a{alpha}_metrics.csv"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    metrics_data = []
    for split_type in ["train", "val"]:
        metrics_data.append({
            "model": "mlp",
            "split": split_name,
            "version": version_name,
            "hidden_layer_sizes": str(hidden_layer_sizes),
            "alpha": alpha,
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
    parser.add_argument("--hidden", default="100", help="Hidden layer sizes, comma-separated (e.g., '100' or '100,50')")
    parser.add_argument("--alpha", type=float, default=0.001, help="L2 regularization parameter")
    parser.add_argument("--max-iter", type=int, default=300, help="Max iterations")
    
    args = parser.parse_args()
    
    hidden_layer_sizes = parse_hidden_layers(args.hidden)
    
    results, vectorizer, model, loss_history, _, _ = train_mlp(
        args.split, args.version, 
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=args.alpha, 
        max_iter=args.max_iter
    )
    
    save_results(args.split, args.version, hidden_layer_sizes, args.alpha, 
                 results, vectorizer, model, loss_history)
    
    print(f"\n{'='*80}")
    print("Training complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
