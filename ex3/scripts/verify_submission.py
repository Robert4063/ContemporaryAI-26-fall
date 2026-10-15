"""Check submission assets and provenance; this is not a data leakage proof.

Run from any working directory. Verification is read-only unless --output is
given. No training, model fitting or test-driven model selection is performed.
"""

import argparse
import ast
import csv
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    # Some source emails exceed the csv module's default 128 KiB field limit.
    csv.field_size_limit(max(csv.field_size_limit(), path.stat().st_size))
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify(root):
    root = root.resolve()
    for name in ["README.md", "requirements.txt", ".gitignore", ".gitattributes",
                 "docs/实验一要求.pdf", "docs/SUBMISSION_GUIDE.md", "docs/KNOWN_ISSUES.md"]:
        require((root / name).is_file(), f"Missing required file: {name}")

    train = rows(root / "data-Project1/train_data.csv")
    unlabeled = rows(root / "data-Project1/test_data_unlabeled.csv")
    require(len(train) == 7368 and len(unlabeled) == 2457, "Unexpected raw dataset size")
    require(set(train[0]) == {"text", "target"} and set(unlabeled[0]) == {"text"}, "Unexpected CSV header")
    require({r["target"] for r in train} == {str(i) for i in range(10)}, "Unexpected training labels")

    baseline = sorted((root / "results/baseline").glob("*_metrics.csv"))
    linear = sorted((root / "results/linear_svm_split_B/metrics").glob("*_metrics.csv"))
    require(len(baseline) == 108 and len(linear) == 12, "Expected 108 baseline + 12 supplemental metrics")
    candidates = []
    for path in baseline + linear:
        records = rows(path)
        require(len(records) == 2 and {r["eval_set"] for r in records} == {"train", "val"}, f"Bad metric rows: {path}")
        for record in records:
            for key in ["accuracy", "macro_f1", "precision", "recall"]:
                require(0 <= float(record[key]) <= 1, f"Invalid {key}: {path}")
        record = next(r for r in records if r["eval_set"] == "val")
        record["config_id"] = path.name.removesuffix("_metrics.csv")
        candidates.append(record)
    ranking = sorted(candidates, key=lambda r: (-float(r["accuracy"]), -float(r["macro_f1"]), r["config_id"]))
    config_ids = [r["config_id"] for r in ranking]
    require(len(set(config_ids)) == 120, "Duplicate metric configuration")
    saved_ranking = rows(root / "results/predictions/model_selection_val.csv")
    require([r["config_id"] for r in saved_ranking] == config_ids, "Saved validation ranking differs")
    for saved, actual in zip(saved_ranking, ranking):
        require(float(saved["accuracy"]) == float(actual["accuracy"]) and
                float(saved["macro_f1"]) == float(actual["macro_f1"]), "Saved validation score differs")

    losses = sorted((root / "results/loss").glob("*.npy"))
    require({p.stem for p in losses} == {p.name.removesuffix("_metrics.csv") for p in baseline}, "Loss/metric names differ")
    for path in losses:
        loss = np.load(path, allow_pickle=False)
        require(loss.shape == (300,) and np.isfinite(loss).all(), f"Invalid loss history: {path}")

    manifest = json.loads((root / "results/predictions/prediction_manifest.json").read_text(encoding="utf-8"))
    require(ranking[0]["config_id"] == manifest["config_id"], "Prediction model is not the validation winner")
    require(digest(root / manifest["input_file"]) == manifest["input_sha256"], "Unlabeled input hash differs")
    model_dir = root / "models" / manifest["config_id"]
    require(digest(model_dir / "model.pkl") == manifest["model_sha256"], "Prediction model hash differs")
    require(digest(model_dir / "vectorizer.pkl") == manifest["vectorizer_sha256"], "Vectorizer hash differs")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    for name, expected in config["artifact_sha256"].items():
        path = root / "results/linear_svm_split_B/metrics" / (manifest["config_id"] + "_metrics.csv") if name == "metrics.csv" else model_dir / name
        require(digest(path) == expected, f"Selected model provenance differs: {name}")
    predictions = (root / "prediction.csv").read_text(encoding="ascii").splitlines()
    require(len(predictions) == 2457 and set(predictions) <= {str(i) for i in range(10)}, "Invalid prediction format")
    require(digest(root / "prediction.csv") == manifest["output_sha256"], "Prediction hash differs")

    final_tests = rows(root / "results/linear_svm_split_B/test_final_comparison.csv")
    require({r["model"] for r in final_tests} == {"svm", "mlp", "linear_svm"} and len(final_tests) == 3, "Missing final test comparison")
    for record in final_tests:
        final_path = root / "results/linear_svm_split_B/test" / (record["model"] + "_final.json")
        final = json.loads(final_path.read_text(encoding="utf-8"))
        require(final["signature"]["config_id"] == record["config_id"], "Final test configuration differs")
        require(float(record["test_accuracy"]) == final["metrics"]["accuracy"], "Final test accuracy differs")
        for name, expected in final["output_sha256"].items():
            require(digest(final_path.parent / name) == expected, f"Final test output hash differs: {name}")

    notebook_path = root / "reports/实验一报告.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    require(all(c["cell_type"] == "markdown" for c in notebook["cells"]), "Expected Markdown-only report")
    source = "\n\n".join("".join(c["source"]) for c in notebook["cells"]) + "\n"
    require(notebook_path.with_suffix(".md").read_text(encoding="utf-8") == source, "Markdown differs from notebook")
    for image in re.findall(r"!\[\]\(([^)]+)\)", source):
        require((notebook_path.parent / image).is_file(), f"Missing report image: {image}")
    pdf = PdfReader(notebook_path.with_suffix(".pdf"))
    require(1 <= len(pdf.pages) <= 2, "PDF exceeds the user limit of two pages")
    qa = json.loads((root / "reports/assets/report_qa.json").read_text(encoding="utf-8"))
    require(digest(notebook_path) == qa["notebook_sha256"] and
            digest(notebook_path.with_suffix(".md")) == qa["markdown_sha256"] and
            digest(notebook_path.with_suffix(".pdf")) == qa["pdf_sha256"], "Report changed since visual review")

    scripts = sorted((root / "scripts").rglob("*.py")) + [root / "run_baseline.py"]
    for path in scripts:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in root.rglob("*.md"):
        if any(part in {".local", ".git", ".venv", "tmp"} for part in path.relative_to(root).parts):
            continue
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            target = target.split("#")[0].strip("<>")
            if target and not re.match(r"[a-zA-Z]+:", target):
                require((path.parent / unquote(target)).exists(), f"Broken Markdown link in {path}: {target}")

    return {"status": "passed", "training_rows": len(train), "unlabeled_rows": len(unlabeled),
            "baseline_configurations": len(baseline), "linear_svm_configurations": len(linear),
            "loss_histories": len(losses), "loss_rounds": 300, "final_test_models": len(final_tests),
            "prediction_config": manifest["config_id"], "prediction_sha256": digest(root / "prediction.csv"),
            "report_pages": len(pdf.pages), "notebook_markdown_synchronized": True,
            "report_hashes_match_visual_review": True, "python_files_parsed": len(scripts),
            "scope": "Asset/provenance consistency only; not a no-leakage or statistical-significance proof."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="Optional JSON verification record")
    args = parser.parse_args()
    result = verify(args.root)
    content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(content)
