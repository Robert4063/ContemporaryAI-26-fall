"""
save_mlp_model.py
=================
直接从已有的 MLP 训练结果 CSV 中找到最优配置,
在 train+val 上重新训练并保存模型到 models/MLP_best.joblib。
不依赖合并文件,直接读取各个 MLP_*.csv 结果文件。
"""
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

ROOT      = Path("d:/AI/projects/experiment_1")
DATA_ROOT = ROOT / "data-processed"
BASELINE  = ROOT / "results" / "baseline"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. 从各 CSV 找最优配置 ───────────────────────────────────────
parts = sorted(BASELINE.glob("MLP_*.csv"))
parts = [p for p in parts if "all_splits_all_versions" not in p.name]
df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
df = df.sort_values(["val_acc", "val_f1"], ascending=False)
best = df.iloc[0]
split, version = best["split"], best["version"]
hidden = tuple(int(x) for x in
               str(best["hidden_layer_sizes"]).strip("()").split(",") if x.strip())
alpha  = float(best["alpha"])

print(f"最优配置: split={split}  version={version}  hidden={hidden}  alpha={alpha}")
print(f"  val_acc={best['val_acc']:.4f}  test_acc={best['test_acc']:.4f}")

# ── 2. 加载 train+val 数据 ──────────────────────────────────────
base  = DATA_ROOT / f"split_{split}"
train = pd.read_csv(base / f"train_{version}.csv")
val   = pd.read_csv(base / f"val_{version}.csv")
full  = pd.concat([train, val], ignore_index=True)
X = full["text"].astype(str).tolist()
y = full["target"].values

vec = TfidfVectorizer(min_df=2, max_df=0.95, max_features=20000)
X_vec = vec.fit_transform(X)
print(f"训练 MLP (train+val 共 {len(full)} 条, {X_vec.shape[1]} 特征)...")

# ── 3. 训练并保存 ───────────────────────────────────────────────
clf = MLPClassifier(
    hidden_layer_sizes=hidden, alpha=alpha,
    max_iter=300, tol=0.0, n_iter_no_change=301,
    early_stopping=False, random_state=42,
)
clf.fit(X_vec, y)
print(f"训练完成 (n_iter={clf.n_iter_})")

joblib.dump((vec, clf), MODEL_DIR / "MLP_best.joblib")
print(f"已保存 models/MLP_best.joblib")

# ── 4. 保存配置 ────────────────────────────────────────────────
cfg = pd.DataFrame([{
    "model": "MLP",
    "split": split,
    "version": version,
    "hidden_layer_sizes": str(hidden),
    "alpha": alpha,
    "val_acc": best["val_acc"],
    "test_acc": best["test_acc"],
    "val_f1": best["val_f1"],
    "test_f1": best["test_f1"],
}])
cfg.to_csv(MODEL_DIR / "MLP_config.csv", index=False)
print(f"已保存 models/MLP_config.csv")
