# Verification and completion status

## Published application

- Public GitHub repository: https://github.com/Devapradeepreddy/bits-ml-spambase
- Live Streamlit app: https://deva-bits-ml-spambase.streamlit.app/
- Deployment: branch main, entrypoint app.py, Python 3.12.
- GitHub automated checks passed: https://github.com/Devapradeepreddy/bits-ml-spambase/actions/runs/41062955427
- All 49 initial published files were checked against their local Git blob hashes, including all five saved models. No mismatches.

## Completed checks

- Original UCI data checked: 4,601 rows, 57 input features and binary target.
- Five models trained and saved with fitted preprocessing pipelines.
- 3,680 training rows and 921 held-out rows; zero shared exact-feature groups.
- Five-fold group-aware cross-validation used training rows only.
- All six held-out metrics measured for every model and saved with predictions.
- 12 automated project checks passed.
- Streamlit execution checks passed for all five models and six metrics, with a guard preventing training.
- All five model selections were verified in the live browser against the saved results.
- Sample CSV download and browser CSV upload tested successfully.
- Uploaded CSV evaluated on 921 rows; comparison CSV export reproduced all 30 saved metric values within 1e-9.
- Confusion matrix, classification report, ROC curve and comparison view rendered online.
- PyArrow is pinned to 24.0.0, matching the working cloud runtime after Streamlit's compatibility adjustment.
- Final PDF generation requires actual links, student details and a screenshot.

## Measured holdout winner

Support Vector Machine: accuracy 0.918567, ROC AUC 0.972321, spam precision
0.913793, recall 0.876033, F1 0.894515 and MCC 0.828739. Confusion counts:
528 correct non-spam, 30 false alarms, 45 missed spam and 318 correct spam.
These scores describe this fixed historical-data split.

## Remaining student action

Execute the assignment in BITS Virtual Lab and capture one genuine screenshot.
Add its path to private_submission/config.json, then run:

```bash
python -m pip install -r requirements-dev.txt
python scripts/build_submission.py --config private_submission/config.json
```

The completed file is private_submission/BITS_ID.pdf using the configured ID.
A prepared report without the screenshot remains a draft and must be completed
before submitting. No BITS Virtual Lab execution or lab proof is claimed by
this project's development or cloud-deployment verification.
