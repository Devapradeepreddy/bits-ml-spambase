# Verification and completion status

## Completed in the development environment

- Official UCI data fetched and bundled with attribution and source documentation.
- Original size checked: 4,601 rows, 57 input features, binary target.
- Five models trained and saved with their fitted preprocessing pipelines.
- Duplicate feature groups kept together: zero shared groups across train/test.
- 3,680 training rows and 921 held-out test rows.
- Five-fold group-aware cross-validation run on training rows only.
- All six held-out metrics measured for each model and saved with predictions.
- 12 automated project checks passed.
- Streamlit execution checks passed for all five model selections, all six metric
  displays and the upload waiting state. A guard raised an error on any attempted
  Pipeline.fit call; the app made none.
- Command-line evaluation reproduced all five model results from saved artifacts.
- Draft PDF generated and all seven rendered pages visually inspected.
- Final PDF generation rejects missing identity, publication links or screenshot.

## Measured holdout winner

Support Vector Machine: accuracy 0.918567, ROC AUC 0.972321, spam precision
0.913793, recall 0.876033, F1 0.894515, MCC 0.828739. Confusion counts:
528 correct non-spam, 30 false alarms, 45 missed spam and 318 correct spam.
These are results on this fixed split, not a guarantee on future email.

## Still requires the relevant account or lab

- GitHub repository: https://github.com/Devapradeepreddy/bits-ml-spambase (public). Source publication is verified separately from hosting.
- Streamlit Community Cloud deployment and public-link verification: pending.
- Full browser interaction with CSV upload: verify after deployment. The build
  environment's browser download was unavailable; CSV parsing/validation and
  Streamlit execution were checked separately.
- BITS Virtual Lab execution and genuine screenshot: student action.
- Final personal PDF: generate once actual links, identity and screenshot exist.

The code package is complete. It is not evidence that public hosting or BITS
Virtual Lab execution has already happened. No invented URLs or lab proof are
included.

## Re-run the checks

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python tests/smoke_app.py
python evaluate.py --csv test_data.csv --model all
```
