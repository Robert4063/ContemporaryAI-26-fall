"""Select the saved model by validation metrics and predict unlabeled rows.

No fitting or test-set evaluation is performed. Default output: prediction.csv
in the project root. Provenance is stored in results/predictions/.
"""

import argparse
import hashlib
import json
import pickle
import runpy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "results" / "predictions"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output):
    metric_paths = sorted((ROOT / "results" / "baseline").glob("*_metrics.csv"))
    metric_paths += sorted((ROOT / "results" / "linear_svm_split_B" / "metrics").glob("*_metrics.csv"))
    candidates = []
    for path in metric_paths:
        frame = pd.read_csv(path, float_precision="round_trip")
        validation = frame.loc[frame["eval_set"] == "val"].copy()
        if len(validation) != 1:
            raise ValueError(f"Expected one validation result: {path}")
        validation["config_id"] = path.name.removesuffix("_metrics.csv")
        validation["metrics_path"] = path.relative_to(ROOT).as_posix()
        validation["metrics_sha256"] = digest(path)
        candidates.append(validation)
    ranking = pd.concat(candidates, ignore_index=True).sort_values(
        ["accuracy", "macro_f1", "config_id"], ascending=[False, False, True]
    )
    if len(ranking) != 120 or ranking["config_id"].duplicated().any():
        raise ValueError("Expected the complete grid: 108 original + 12 LinearSVC configurations")
    best = ranking.iloc[0]
    model_dir = ROOT / "models" / best["config_id"]
    model_path = model_dir / "model.pkl"
    vectorizer_path = model_dir / "vectorizer.pkl"
    if best["model"] == "linear_svm":
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        for path in [model_path, vectorizer_path]:
            if digest(path) != config["artifact_sha256"][path.name]:
                raise ValueError(f"Saved model checksum mismatch: {path}")

    raw_path = ROOT / "data-Project1" / "test_data_unlabeled.csv"
    raw = pd.read_csv(raw_path, keep_default_na=False)
    if list(raw.columns) != ["text"] or len(raw) != 2457:
        raise ValueError("Expected 2457 unlabeled rows and one text column")
    extract = runpy.run_path(str(ROOT / "scripts" / "preprocessing" / "preprocess.py"))["extract_text_versions"]
    texts = [extract(text)[best["version"]] for text in raw["text"]]
    processed_path = ROOT / "data-processed" / best["split"] / best["version"] / "unlabeled.csv"
    # The GitHub checkout excludes regenerable processed CSVs. Prediction always
    # transforms the original input in order; additionally verify the cache when
    # it exists, without requiring 198 MiB of duplicated text for inference.
    if processed_path.exists():
        processed = pd.read_csv(processed_path, keep_default_na=False)
        if processed["text"].tolist() != texts:
            raise ValueError("Preprocessed unlabeled rows do not match the original order/text transformation")

    with vectorizer_path.open("rb") as stream:
        vectorizer = pickle.load(stream)
    with model_path.open("rb") as stream:
        model = pickle.load(stream)
    if not np.array_equal(model.classes_, np.arange(10)):
        raise ValueError("Unexpected class labels")
    predictions = model.predict(vectorizer.transform(texts))
    if predictions.shape != (2457,) or not np.isin(predictions, np.arange(10)).all():
        raise ValueError("Predictions must be 2457 labels in the range 0-9")
    content = "".join(f"{int(label)}\n" for label in predictions).encode("ascii")
    if output.exists() and output.read_bytes() != content:
        raise FileExistsError(f"Refusing to overwrite a different prediction file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)
    if output.read_text(encoding="ascii").splitlines() != [str(int(p)) for p in predictions]:
        raise ValueError("Written predictions failed read-back verification")

    PROVENANCE.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(PROVENANCE / "model_selection_val.csv", index=False)
    provenance = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "sklearn_version": sklearn.__version__,
        "selection_rule": "validation accuracy desc, macro_f1 desc, config_id asc; test metrics excluded",
        "configurations_compared": len(ranking),
        "config_id": best["config_id"], "model": best["model"],
        "split": best["split"], "version": best["version"],
        "val_accuracy": float(best["accuracy"]), "val_macro_f1": float(best["macro_f1"]),
        "input_file": raw_path.relative_to(ROOT).as_posix(), "input_sha256": digest(raw_path),
        "processed_input_file": processed_path.relative_to(ROOT).as_posix() if processed_path.exists() else None,
        "processed_input_sha256": digest(processed_path) if processed_path.exists() else None,
        "processed_cache_verified": processed_path.exists(),
        "model_sha256": digest(model_path), "vectorizer_sha256": digest(vectorizer_path),
        "output_file": str(output), "output_sha256": digest(output),
        "rows": 2457, "header": False, "labels": list(range(10)),
        "label_counts": {str(i): int(np.count_nonzero(predictions == i)) for i in range(10)},
        "original_sample_order_verified": True, "refitted": False,
    }
    (PROVENANCE / "prediction_manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "prediction.csv")
    main(parser.parse_args().output.resolve())
