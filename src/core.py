"""One feature contract and evaluation path shared by training, app and checks."""
from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
TARGET = "spam"
WORD_NAMES = (
    "make address all 3d our over remove internet order mail receive will people "
    "report addresses free business email you credit your font 000 money hp hpl "
    "george 650 lab labs telnet 857 data 415 85 technology 1999 parts pm direct cs "
    "meeting original project re edu table conference"
).split()
FEATURES = [f"word_freq_{word}" for word in WORD_NAMES] + [
    f"char_freq_{char}" for char in [";", "(", "[", "!", "$", "#"]
] + ["capital_run_length_average", "capital_run_length_longest", "capital_run_length_total"]
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def build_models() -> dict[str, Pipeline]:
    """Fixed configurations; no tuning against the held-out test set."""
    def pipe(estimator, scale=False):
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        return Pipeline(steps + [("classifier", estimator)])

    return {
        "Logistic Regression": pipe(LogisticRegression(C=1.0, max_iter=4000, random_state=SEED), True),
        "Decision Tree": pipe(DecisionTreeClassifier(max_depth=12, min_samples_leaf=3, random_state=SEED)),
        "KNN": pipe(KNeighborsClassifier(n_neighbors=7, weights="distance"), True),
        "Naive Bayes": pipe(GaussianNB(var_smoothing=1e-9)),
        "Support Vector Machine": pipe(SVC(C=3.0, kernel="rbf", gamma="scale", random_state=SEED), True),
    }


def feature_groups(frame: pd.DataFrame) -> np.ndarray:
    """Group exact duplicate feature vectors, irrespective of their target labels."""
    # Hashes are grouping IDs, not cryptographic claims; fit never uses them.
    return pd.util.hash_pandas_object(frame.loc[:, FEATURES], index=False).to_numpy()


def validate_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Require the documented 57 features and binary target; reorder safely."""
    if frame.empty:
        raise ValueError("The CSV has no data rows.")
    if frame.columns.duplicated().any():
        raise ValueError("The CSV contains duplicate column names.")
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in frame.columns]
    extra = [str(c) for c in frame.columns if c not in required]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if extra:
        raise ValueError("Unexpected columns: " + ", ".join(extra) + ". Use the supplied CSV schema.")
    try:
        X = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="raise")
        y = pd.to_numeric(frame[TARGET], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Features must be numeric; spam must contain 0 or 1.") from exc
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("The spam target must contain only 0 (non-spam) or 1 (spam), with no blanks.")
    values = X.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Features contain infinity. Supply finite numeric values or blank cells.")
    if np.isnan(values).all(axis=0).any():
        raise ValueError("At least one feature is completely blank. Supply at least one value per feature.")
    if np.any(values < 0):
        raise ValueError("Spambase frequencies and capital-run features cannot be negative.")
    if np.any(values[:, :54] > 100):
        raise ValueError("Word and character frequency percentages must be between 0 and 100.")
    return X, y.astype(int)


def read_test_csv(source) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Check raw headers before pandas can silently rename duplicate columns."""
    if isinstance(source, (str, Path)):
        data = Path(source).read_bytes()
    elif isinstance(source, bytes):
        data = source
    else:
        data = source.getvalue() if hasattr(source, "getvalue") else source.read()
    if len(data) > 20 * 1024 * 1024:
        raise ValueError("CSV exceeds the 20 MB upload limit.")
    try:
        decoded = data.decode("utf-8-sig")
        headers = next(csv.reader(io.StringIO(decoded)), [])
        if len(headers) != len(set(headers)):
            raise ValueError("The CSV contains duplicate column names.")
        frame = pd.read_csv(io.StringIO(decoded))
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("Upload a valid UTF-8, comma-separated CSV with a header row.") from exc
    if len(frame) > 10000:
        raise ValueError("Upload at most 10,000 test rows.")
    X, y = validate_frame(frame)
    return frame, X, y


def positive_scores(model, X: pd.DataFrame) -> tuple[np.ndarray, str]:
    """ROC AUC requires a score, never the thresholded prediction labels."""
    if hasattr(model, "predict_proba"):
        index = int(np.flatnonzero(model.classes_ == 1)[0])
        return model.predict_proba(X)[:, index], "spam_probability"
    return np.asarray(model.decision_function(X)), "spam_decision_score"


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    predicted = model.predict(X)
    scores, score_name = positive_scores(model, X)
    auc = float(roc_auc_score(y, scores)) if y.nunique() == 2 else None
    values = {
        "Accuracy": float(accuracy_score(y, predicted)),
        "AUC": auc,
        "Precision": float(precision_score(y, predicted, pos_label=1, zero_division=0)),
        "Recall": float(recall_score(y, predicted, pos_label=1, zero_division=0)),
        "F1": float(f1_score(y, predicted, pos_label=1, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, predicted)),
    }
    return {
        "metrics": values,
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y, predicted, labels=[0, 1], target_names=["Non-spam (0)", "Spam (1)"],
            zero_division=0, output_dict=True,
        ),
        "predictions": predicted,
        "scores": scores,
        "score_name": score_name,
    }
