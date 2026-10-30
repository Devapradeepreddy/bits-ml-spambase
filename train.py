"""Reproduce the entire experiment offline from the bundled UCI source data.

Usage: python train.py
       python train.py --skip-cv   # faster, removes any stale CV output
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from src.core import ROOT, SEED, TARGET, FEATURES, MODEL_FILES, METRICS, build_models, feature_groups, validate_frame, evaluate


def run(skip_cv=False):
    for directory in ["model", "results", "assets"]:
        (ROOT / directory).mkdir(exist_ok=True)
    raw_path = ROOT / "data" / "spambase.data"
    data = pd.read_csv(raw_path, header=None, names=FEATURES + [TARGET])
    X, y = validate_frame(data)
    groups = feature_groups(X)
    # The first of five folds is held out once; all preprocessing lives in pipelines.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    overlap = set(groups[train_idx]) & set(groups[test_idx])
    assert not overlap, "Duplicate feature groups leaked across the holdout boundary."
    data.iloc[test_idx].to_csv(ROOT / "test_data.csv", index=False)
    membership = pd.DataFrame({"source_row_0_based": range(len(data)), "partition": "train"})
    membership.loc[test_idx, "partition"] = "test"
    membership.to_csv(ROOT / "results" / "split_membership.csv", index=False)
    pd.DataFrame({"feature": FEATURES, "dtype": [str(X[c].dtype) for c in FEATURES],
                  "description": ["Percentage frequency" if i < 54 else "Capital-run statistic" for i in range(57)]}).to_csv(ROOT / "data" / "feature_schema.csv", index=False)
    schema = {"features": FEATURES, "target": TARGET, "target_labels": {"0": "Non-spam", "1": "Spam"},
              "positive_class": 1, "missing_feature_values": "training-fitted median imputation"}
    (ROOT / "model" / "schema.json").write_text(json.dumps(schema, indent=2) + "\n")

    rows, details, cv_rows = [], {}, []
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in build_models().items():
        print(f"Training {name} ...", flush=True)
        if not skip_cv:
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
            scores = cross_validate(model, X_train, y_train, groups=groups[train_idx], cv=cv,
                                    scoring={"accuracy": "accuracy", "f1": "f1", "auc": "roc_auc"},
                                    n_jobs=1, error_score="raise")
            cv_rows.append({"Model": name, **{
                f"{metric}_{stat}": float(getattr(np, stat)(scores[f"test_{metric}"]))
                for metric in ["accuracy", "f1", "auc"] for stat in ["mean", "std"]
            }})
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        result = evaluate(model, X_test, y_test)
        joblib.dump(model, ROOT / "model" / MODEL_FILES[name], compress=3)
        rows.append({"Model": name, **result["metrics"]})
        details[name] = {k: v for k, v in result.items() if k not in ["predictions", "scores"]}
        details[name]["fit_seconds"] = fit_seconds
        predictions = pd.DataFrame({"source_row_0_based": test_idx, "actual": y_test.to_numpy(),
                                    "predicted": result["predictions"], result["score_name"]: result["scores"]})
        slug = MODEL_FILES[name].removesuffix(".joblib")
        predictions.to_csv(ROOT / "results" / f"{slug}_predictions.csv", index=False)
        cmfig, cmax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(np.array(result["confusion_matrix"]), display_labels=["Non-spam", "Spam"]).plot(ax=cmax, cmap="Blues", colorbar=False)
        cmax.set_title(name)
        cmfig.tight_layout()
        cmfig.savefig(ROOT / "assets" / f"{slug}_confusion.png", dpi=160)
        plt.close(cmfig)
        RocCurveDisplay.from_predictions(y_test, result["scores"], name=name, ax=ax)
        print("  " + ", ".join(f"{k}={v:.4f}" for k, v in result["metrics"].items()), flush=True)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af", label="Chance")
    ax.set_title("ROC curves on the same held-out test set")
    fig.tight_layout()
    fig.savefig(ROOT / "assets" / "roc_curves.png", dpi=170)
    plt.close(fig)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(ROOT / "results" / "metrics.csv", index=False, float_format="%.10f")
    (ROOT / "results" / "evaluation.json").write_text(json.dumps(details, indent=2, allow_nan=False) + "\n")
    if cv_rows:
        pd.DataFrame(cv_rows).to_csv(ROOT / "results" / "cross_validation.csv", index=False)
    else:
        (ROOT / "results" / "cross_validation.csv").unlink(missing_ok=True)
    feature_label_counts = data.groupby(FEATURES, dropna=False)[TARGET].nunique()
    manifest = {
        "dataset": "UCI Spambase", "source_url": "https://archive.ics.uci.edu/dataset/94/spambase",
        "doi": "10.24432/C53G6X", "license": "CC BY 4.0", "seed": SEED,
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "test_csv_sha256": hashlib.sha256((ROOT / "test_data.csv").read_bytes()).hexdigest(),
        "rows": len(data), "features": len(FEATURES), "missing_feature_cells": int(X.isna().sum().sum()),
        "duplicate_feature_rows_beyond_first": int(X.duplicated().sum()),
        "feature_groups_with_conflicting_labels": int((feature_label_counts > 1).sum()),
        "train_rows": len(train_idx), "test_rows": len(test_idx),
        "train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        "shared_feature_groups": len(overlap),
        "split_method": "First fold of StratifiedGroupKFold(5, shuffle=True, random_state=42); exact feature duplicates grouped",
        "cross_validation": "5-fold StratifiedGroupKFold on training partition only" if not skip_cv else "skipped",
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_environment": "Development runtime; not BITS Virtual Lab",
    }
    (ROOT / "results" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(metrics))
    colors = ["#0e7490", "#2563eb", "#d97706"]
    for i, metric in enumerate(["Accuracy", "F1", "AUC"]):
        ax.bar(x + (i - 1) * .23, metrics[metric], width=.23, label=metric, color=colors[i])
    ax.set_xticks(x, ["Logistic\nRegression", "Decision\nTree", "KNN", "Naive\nBayes", "SVM"])
    ax.set_ylim(0, 1.07)
    ax.set_ylabel("Score")
    ax.set_title("Held-out test comparison - positive class: spam")
    ax.legend(loc="lower left", ncol=3)
    fig.tight_layout()
    fig.savefig(ROOT / "assets" / "model_comparison.png", dpi=170)
    plt.close(fig)
    print(f"Complete: {len(train_idx)} train / {len(test_idx)} test; {len(overlap)} shared feature groups.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()
    run(skip_cv=args.skip_cv)
