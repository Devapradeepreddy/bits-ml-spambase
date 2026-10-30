"""Build the required README from measured outputs, without inventing live URLs."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.core import ROOT, METRICS


def markdown_table(frame, decimals=4):
    headers = list(frame.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.{decimals}f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def run(config_path=None):
    path = Path(config_path) if config_path else ROOT / "submission_config.json"
    config = json.loads(path.read_text())
    manifest = json.loads((ROOT / "results" / "manifest.json").read_text())
    details = json.loads((ROOT / "results" / "evaluation.json").read_text())
    metrics = pd.read_csv(ROOT / "results" / "metrics.csv")
    winner = metrics.sort_values(["F1", "AUC"], ascending=False).iloc[0]
    observations = []
    for _, row in metrics.iterrows():
        name = row.Model
        tn, fp = details[name]["confusion_matrix"][0]
        fn, tp = details[name]["confusion_matrix"][1]
        description = (
            f"Accuracy {row.Accuracy:.4f}; spam precision {row.Precision:.4f}, recall {row.Recall:.4f}, "
            f"F1 {row.F1:.4f}. {fp} normal emails flagged and {fn} spam emails missed."
        )
        if name == "Logistic Regression":
            description += " The scaled linear baseline provides a useful reference for the nonlinear methods."
        elif name == "Decision Tree":
            description += " Its depth and minimum leaf size are limited; the observed results reflect this fixed configuration."
        elif name == "KNN":
            description += " Distance-weighted voting uses seven neighbors after scaling, so feature magnitude does not set the distance by itself."
        elif name == "Naive Bayes":
            description += " It catches many spam examples but also produces more false alarms here. Gaussian feature shapes and conditional independence are modeling assumptions, not verified explanations for the gap."
        else:
            description += " The scaled RBF model achieves the highest F1 in this run. ROC AUC uses its decision margins, not probabilities."
        observations.append({"ML Model Name": name, "Observation about model performance": description})
    observations.append({"ML Model Name": "Overall winner for this dataset", "Observation about model performance":
                         f"{winner.Model} by held-out spam F1 ({winner.F1:.4f}); accuracy {winner.Accuracy:.4f}, AUC {winner.AUC:.4f}. This is a descriptive comparison on this split, not a universal ranking or a tuned deployment decision."})
    github = config.get("github_url") or "Pending publication - replace with the actual public GitHub repository URL."
    streamlit = config.get("streamlit_url") or "Pending deployment - replace with the actual live Streamlit app URL."
    cv_path = ROOT / "results" / "cross_validation.csv"
    cv_text = "Training-only cross-validation was skipped in the latest run."
    if cv_path.exists():
        cv = pd.read_csv(cv_path)
        cv_out = pd.DataFrame({"Model": cv.Model,
                               "Accuracy mean +/- SD": [f"{a:.4f} +/- {b:.4f}" for a, b in zip(cv.accuracy_mean, cv.accuracy_std)],
                               "F1 mean +/- SD": [f"{a:.4f} +/- {b:.4f}" for a, b in zip(cv.f1_mean, cv.f1_std)],
                               "AUC mean +/- SD": [f"{a:.4f} +/- {b:.4f}" for a, b in zip(cv.auc_mean, cv.auc_std)]})
        cv_text = markdown_table(cv_out)
    text = f"""# SpamScope - five-model email classification comparison

M.Tech (AIML), Machine Learning - Assignment 2

Live application: {streamlit}

Public source: {github}

## a. Problem statement

Classify an email as spam (1) or non-spam (0) from 57 precomputed numeric
features. Implement Logistic Regression, Decision Tree, K-Nearest Neighbors,
Gaussian Naive Bayes and Support Vector Machine on the same dataset and compare
Accuracy, ROC AUC, Precision, Recall, F1 and MCC on one common held-out test set.
Provide an interactive Streamlit application that evaluates saved models on
labeled uploaded test CSVs without retraining them.

## b. Dataset description

- Dataset: UCI Spambase, a public binary classification dataset.
- Source: https://archive.ics.uci.edu/dataset/94/spambase
- Citation: Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999). Spambase [Dataset]. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C53G6X
- License: CC BY 4.0; see data/ATTRIBUTION.md.
- Instances: {manifest['rows']:,}; input features: {manifest['features']} (meets at least 500 instances and 12 features).
- Features: 48 word frequencies, 6 character frequencies and 3 capital-run statistics.
- Target: spam; 0 = non-spam, 1 = spam. Total counts: 2,788 non-spam and 1,813 spam.
- Missing feature cells in the original data: {manifest['missing_feature_cells']}.
- Duplicate feature rows beyond the first occurrence: {manifest['duplicate_feature_rows_beyond_first']}.
- Identical-feature groups containing conflicting class labels: {manifest['feature_groups_with_conflicting_labels']}.
- Group-aware partition: {manifest['train_rows']:,} training rows and {manifest['test_rows']:,} test rows; shared exact feature groups: {manifest['shared_feature_groups']}.
- Training class counts: {manifest['train_class_counts']['0']} non-spam / {manifest['train_class_counts']['1']} spam.
- Test class counts: {manifest['test_class_counts']['0']} non-spam / {manifest['test_class_counts']['1']} spam.

The target is excluded from all input features. Exact duplicate feature vectors
remain together using the first fold from StratifiedGroupKFold with five folds,
shuffle=True and random_state=42. No source rows are dropped. Groups are based
on features, not labels, so contradictory copies also remain together. This
prevents exact-copy overlap, while near-duplicate or collection-source effects
can still exist. Data originated in 1999 and contain person-specific indicators;
the results are not evidence of performance on present-day email.

## c. GitHub repository and live application

GitHub repository: {github}

Live Streamlit application: {streamlit}

The repository includes app.py, requirements.txt, README.md, test_data.csv,
training source, all five saved models, preprocessing pipelines, original data,
result tables, figures, tests and deployment instructions. The app entrypoint
is app.py. Deploy with Python 3.12 and the pinned requirements.txt.

## d. Models used and evaluation metrics

All metrics below are measured on the same held-out test rows. Precision,
Recall and F1 use spam (1) as the positive class. ROC AUC uses continuous spam
probabilities for four models and decision scores for SVM. It is not computed
from hard class predictions. MCC lies between -1 and 1; the other reported
metrics lie between 0 and 1.

{markdown_table(metrics[['Model'] + METRICS])}

### Preprocessing and fixed model settings

| Model | Configuration |
| --- | --- |
| Logistic Regression | Training median imputation, StandardScaler, C=1, max_iter=4000, seed=42. |
| Decision Tree | Training median imputation, max_depth=12, min_samples_leaf=3, seed=42. |
| KNN | Training median imputation, StandardScaler, n_neighbors=7, weights=distance. |
| Naive Bayes | Training median imputation, GaussianNB with var_smoothing=1e-9. |
| Support Vector Machine | Training median imputation, StandardScaler, SVC with RBF kernel, C=3, gamma=scale, seed=42. |

Imputers and scalers are fitted within each training fold and the final training
partition only. Uploaded test data never fit or update any preprocessing or
classifier. Saved Pipeline objects contain every required transformation.
Settings are fixed rather than selected by repeated holdout trials.

### Training-only five-fold cross-validation

{cv_text}

The folds are group-aware. Standard deviations describe variation across these
five folds; they are not confidence intervals. These training-only scores are
separate from the mandatory held-out comparison table above.

## e. Observations on model performance

{markdown_table(pd.DataFrame(observations))}

This experiment prioritizes spam F1 for the descriptive winner and uses AUC as
a tie-breaker. A real deployment may prioritize reducing false alarms instead.
Neither this split nor the cross-validation establishes a universal best model.

![Held-out model comparison](assets/model_comparison.png)

![Held-out ROC curves](assets/roc_curves.png)

## Application features

- Labeled test CSV upload with required-column, numeric-value and binary-target validation.
- A dropdown for all five classifiers and an included test-data example.
- Six evaluation metrics; AUC is shown as unavailable for a one-class CSV.
- Confusion matrix, classification report and ROC curve.
- A comparison table calculated on the active test rows for every saved model.
- Downloadable sample data, predictions and comparison metrics.
- Median imputation for individual blank feature values using training-fitted medians.
- No retraining, API keys or remote model calls in the app.

The app accepts extracted Spambase numeric features, not raw email text. User
uploads must be truly held out to support generalization claims. Only trusted
local model artifacts are deserialized; no uploaded model files are accepted.

## Run, reproduce and check

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

```bash
python train.py
python -m unittest discover -s tests -p 'test_*.py' -v
python tests/smoke_app.py
python evaluate.py --csv test_data.csv --model all
python scripts/update_readme.py
```

The raw dataset is bundled, so training does not require a dataset download.
Use Python 3.12 and scikit-learn=={manifest['scikit_learn']} for artifact compatibility.
Detailed operating instructions are in START_HERE.md; file explanations and
metric definitions are in EXPLANATION.md.

## Evidence and reproducibility

- Dataset SHA-256: {manifest['raw_sha256']}
- Held-out CSV SHA-256: {manifest['test_csv_sha256']}
- Python: {manifest['python']}; scikit-learn: {manifest['scikit_learn']}; NumPy: {manifest['numpy']}; pandas: {manifest['pandas']}.
- Split membership: results/split_membership.csv.
- Predictions: results/*_predictions.csv; confusion counts: results/evaluation.json.
- Full provenance and creation time: results/manifest.json.

The initial packaged execution was performed in a development runtime. It is
not BITS Virtual Lab execution proof. The student must execute the assignment
in the required lab and supply a genuine screenshot for the final PDF.

## Deployment and submission

Use Streamlit Community Cloud with this repository, branch main, entrypoint
app.py and Python 3.12. See START_HERE.md for the exact sequence and official
documentation links. A repository or app URL is pending until it has really
been created and opened; no placeholder is a valid submission link.

The final PDF must contain, in order, the working GitHub link, live Streamlit
link, genuine BITS Virtual Lab execution screenshot and the complete README.
Run scripts/build_submission.py with completed metadata and your screenshot.
Personal screenshots and the final PDF belong in private_submission/, which
is excluded from Git. A draft PDF with missing items is not ready to submit.
"""
    (ROOT / "README.md").write_text(text, encoding="utf-8")
    print("Updated README.md from the current measured results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    run(parser.parse_args().config)
