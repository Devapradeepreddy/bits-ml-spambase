"""Streamlit inference and evaluation UI. No fitting occurs in this application."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from src.core import ROOT, FEATURES, MODEL_FILES, METRICS, read_test_csv, evaluate

st.set_page_config(page_title="SpamScope | ML comparison", page_icon="✉", layout="wide")
st.markdown("""<style>
.block-container {padding-top:4.5rem; max-width:1280px;}
[data-testid="stMetric"] {background:#f0f7fb; border:1px solid #d8e8f1; border-radius:10px; padding:12px;}
[data-testid="stMetricValue"] {color:#164e63;}
.eyebrow {color:#0e7490; font-weight:700; letter-spacing:.14em; font-size:.8rem;}
.hero-note {color:#526575; font-size:1.05rem; margin-bottom:1.5rem;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    # Only local, trusted repository artifacts are deserialized. No model upload UI.
    manifest = json.loads((ROOT / "results" / "manifest.json").read_text())
    models = {name: joblib.load(ROOT / "model" / file) for name, file in MODEL_FILES.items()}
    return models, manifest


try:
    models, manifest = load_artifacts()
except Exception as exc:
    st.error("Saved model artifacts could not be loaded. Install requirements.txt and run python train.py once.")
    st.code(str(exc))
    st.stop()

st.markdown('<div class="eyebrow">BITS WILP · MACHINE LEARNING ASSIGNMENT 2</div>', unsafe_allow_html=True)
st.title("SpamScope")
st.markdown('<div class="hero-note">One dataset. Five classifiers. Evidence you can inspect.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Evaluate saved models")
    choice = st.selectbox("Model", list(MODEL_FILES), key="model_choice")
    source = st.radio("Test data source", ["Included holdout test data", "Upload test CSV"], key="data_source")
    uploaded = None
    if source == "Upload test CSV":
        uploaded = st.file_uploader("Upload labeled test CSV", type=["csv"], key="test_csv")
    st.download_button("Download sample test CSV", data=(ROOT / "test_data.csv").read_bytes(),
                       file_name="test_data.csv", mime="text/csv")
    st.caption("CSV requires 57 numeric features and the spam target (0 or 1). Upload test data only.")
    st.divider()
    st.markdown("**UCI Spambase**")
    st.caption(f"{manifest['rows']:,} source rows · {manifest['features']} features\n\nPositive class: spam (1)")
    st.caption("Saved pipelines handle preprocessing. This app does not retrain models.")

if source == "Upload test CSV" and uploaded is None:
    st.info("Upload a labeled test CSV using the sidebar. Download the sample to see the required columns.")
    st.stop()
payload = uploaded.getvalue() if uploaded is not None else (ROOT / "test_data.csv").read_bytes()
try:
    frame, X, y = read_test_csv(payload)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

result = evaluate(models[choice], X, y)
is_reference = hashlib.sha256(payload).hexdigest() == manifest["test_csv_sha256"]
st.caption(f"{choice}  /  {len(frame):,} evaluated rows  /  {'Included held-out test set' if is_reference else 'User-supplied data; independence from training is not verified'}")
if not is_reference:
    st.info("Scores describe the uploaded rows. They estimate generalization only if those rows were held out from model training.")
if y.nunique() < 2:
    st.warning("This CSV contains one target class. ROC AUC is undefined; other metrics may be uninformative.")
if X.isna().any().any():
    st.info("Blank feature cells are filled with medians learned during training.")

for column, metric in zip(st.columns(6), METRICS):
    value = result["metrics"][metric]
    column.metric(metric, "N/A" if value is None else f"{value:.4f}")

tab_eval, tab_compare, tab_rows, tab_method = st.tabs(["Model evaluation", "Compare all five", "Data & predictions", "Method & definitions"])
with tab_eval:
    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(np.array(result["confusion_matrix"]), display_labels=["Non-spam", "Spam"]).plot(ax=ax, cmap="Blues", colorbar=False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        tn, fp, fn, tp = np.array(result["confusion_matrix"]).ravel()
        st.caption(f"Correct non-spam: {tn} · False alarms: {fp} · Missed spam: {fn} · Correct spam: {tp}")
    with right:
        st.subheader("ROC curve")
        if y.nunique() == 2:
            fig, ax = plt.subplots(figsize=(5, 4))
            RocCurveDisplay.from_predictions(y, result["scores"], ax=ax, name=choice)
            ax.plot([0, 1], [0, 1], "--", color="#9ca3af")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Both target classes are needed to draw a meaningful ROC curve.")
    st.subheader("Classification report")
    st.dataframe(pd.DataFrame(result["classification_report"]).T, use_container_width=True)

with tab_compare:
    st.subheader("Same rows, same metrics")
    evaluations = {name: evaluate(model, X, y) for name, model in models.items()}
    comparison = pd.DataFrame([{"Model": name, **values["metrics"]} for name, values in evaluations.items()])
    st.dataframe(comparison.style.format({m: "{:.4f}" for m in METRICS}, na_rep="N/A"), hide_index=True, use_container_width=True)
    st.bar_chart(comparison.set_index("Model")[["Accuracy", "F1", "AUC"]])
    best = comparison.sort_values(["F1", "AUC"], ascending=False).iloc[0]
    st.info(f"Highest F1 on these rows: {best['Model']} ({best['F1']:.4f}). This is a descriptive comparison, not a guarantee for future email.")
    st.download_button("Download comparison CSV", comparison.to_csv(index=False), "model_comparison.csv", "text/csv")

with tab_rows:
    st.subheader("Input preview")
    st.dataframe(frame.head(25), use_container_width=True)
    predictions = pd.DataFrame({"actual": y.to_numpy(), "predicted": result["predictions"],
                                result["score_name"]: result["scores"]})
    predictions["correct"] = predictions.actual == predictions.predicted
    st.subheader("Prediction preview")
    st.dataframe(predictions.head(100), use_container_width=True)
    st.caption("SVM scores are signed decision margins; they are not probabilities.")
    st.download_button("Download predictions CSV", predictions.to_csv(index=False), "predictions.csv", "text/csv")

with tab_method:
    st.markdown(f"""### Reproducible experiment
- Dataset: [UCI Spambase](https://archive.ics.uci.edu/dataset/94/spambase), licensed CC BY 4.0.
- Split: **{manifest['train_rows']:,} training / {manifest['test_rows']:,} test rows**, seed 42.
- Identical feature vectors stay in one partition: **{manifest['shared_feature_groups']} shared groups**.
- Training-only median imputation; standard scaling for Logistic Regression, KNN and SVM.
- Fixed model settings; five-fold, group-aware cross-validation uses training rows only.
- Precision, recall and F1 refer to **spam (1)**; ROC AUC uses scores, not predicted labels.

### Read the metrics
| Metric | Meaning |
|---|---|
| Accuracy | Fraction of all emails classified correctly. |
| AUC | How well the scores rank spam above non-spam across thresholds. |
| Precision | Of predicted spam, the fraction that is actually spam. |
| Recall | Of actual spam, the fraction detected. |
| F1 | Harmonic mean of spam precision and recall. |
| MCC | Correlation-like summary of all four confusion-matrix counts; -1 to 1. |

This historical dataset includes person-specific clues. Its scores do not establish performance on modern mail.
The app accepts extracted numeric features, not raw email text.
""")
    with st.expander("Required CSV columns"):
        st.code(", ".join(FEATURES + ["spam"]))
    st.json({k: manifest[k] for k in ["raw_sha256", "python", "scikit_learn", "created_utc"]})

st.divider()
st.caption("SpamScope · Saved-model evaluation · No API key required")
