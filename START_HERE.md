# Start here - complete ML Assignment 2 project

The five trained models and sample test data are already included. You can run
the app immediately after installing the dependencies. No Azure/OpenAI key,
paid service, or dataset download is needed.

## 1. Open the folder

Extract the ZIP, open `bits-ml-spambase` in VS Code, and open its terminal.
The terminal must be in the folder that contains `app.py`.

Use **Python 3.12**. It matches the tested runtime and the recommended deployment
setting. Keep the pinned scikit-learn version: serialized models are not
guaranteed compatible with a different scikit-learn version.

## 2. Install and run (Ubuntu / BITS lab terminal)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If your lab exposes Python 3.12 as `python3`, replace `python3.12` with `python3`
after checking `python3 --version`. If virtual environments are unavailable on
the lab, use its supported environment setup; do not overwrite system Python.

## 3. Install and run (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit. It is normally `http://localhost:8501`.
No model retraining is needed to start the app.

## 4. What to demonstrate

1. Open the app. The included holdout CSV is selected by default.
2. Select each of the five classifiers from the sidebar.
3. Inspect all six metrics, the confusion matrix, ROC curve and classification report.
4. Open **Compare all five** for a table calculated on the same active data.
5. Download the sample CSV. Change the data source to **Upload test CSV** and upload it.
6. Download the comparison and prediction CSVs if required.

The app expects 57 named numeric columns plus `spam`, the known target label.
It does not classify raw email text. AUC needs both target classes. Other
uploaded datasets require the same Spambase feature definitions and units.

## 5. Reproduce training and verify

Run these commands in BITS Virtual Lab for your own execution proof:

```bash
python train.py
python -m unittest discover -s tests -p 'test_*.py' -v
python tests/smoke_app.py
python evaluate.py --csv test_data.csv --model all
python scripts/update_readme.py
python -m streamlit run app.py
```

Training uses the bundled original data and reproduces the split, pipelines,
metrics, plots and prediction files. It also runs five-fold cross-validation on
the training partition. Runtime depends on the lab resources; the script prints
each model as it progresses. Do not compare your wall-clock speed with another
machine's speed. `python train.py --skip-cv` is a quicker development option;
use the full command for the documented experiment.

Capture one genuine screenshot showing your BITS Virtual Lab session and your
own execution/output. Follow the lab interface's screenshot method. Keep the
image locally; it does not need to be in your public repository. A local app
preview from this package is not a substitute for that screenshot.

## 6. Create the public GitHub repository

Recommended repository name: `bits-ml-spambase`.

The repository must contain the **contents** of this project folder at its root.
Check that `app.py`, `requirements.txt`, `README.md`, `test_data.csv`, `src/`,
`data/`, `results/` and all five `model/*.joblib` files are visible online.
The files are small enough for an ordinary repository; Git LFS is unnecessary.
Do not upload `.venv`, personal screenshots, credentials, or your final private PDF.

If GitHub CLI is installed and authenticated:

```bash
gh auth login
git init -b main
git add .
git commit -m "Add Spambase classifier comparison and Streamlit evaluation app"
gh repo create bits-ml-spambase --public --source=. --remote=origin --push
```

If the folder is already a Git repository, skip `git init`. If you already created
the remote repository, use its actual URL with `git remote add origin URL` and
`git push -u origin main` instead of `gh repo create`. Do not run a second
repository-create command for an existing repository.

Alternatively, create an empty public repository on github.com and upload the
project contents, including `model/` and `.streamlit/config.toml`.

## 7. Deploy on Streamlit Community Cloud

1. Open https://share.streamlit.io/ and sign in using the GitHub account containing this repository.
2. Choose **Create app** and select your public repository.
3. Branch: `main`. Main file path: `app.py`.
4. In **Advanced settings**, select **Python 3.12**.
5. Deploy. No secrets or API keys are required.
6. Wait until the real app opens, then test all five models and CSV upload.
7. Copy the actual `https://...streamlit.app` URL. Do not use a guessed URL or a localhost URL.

The deployment is incomplete until the public URL has been opened and checked.
If a model-loading error appears, confirm `scikit-learn==1.8.0` is installed and
all model files were committed. If dependencies fail to install, verify Python
3.12 and inspect the deployment logs before changing version pins.

Official deployment documentation:
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## 8. Update links, then generate your PDF

Install PDF dependencies only when you need to build the report:

```bash
python -m pip install -r requirements-dev.txt
```

Edit `submission_config.json` with your real name, BITS ID, GitHub repository
URL, live Streamlit URL and screenshot path. Do not commit personal details.
You can instead put the completed config in `private_submission/config.json`
(this directory is ignored by Git) and pass `--config private_submission/config.json`.

```bash
python scripts/update_readme.py --config private_submission/config.json
python scripts/build_submission.py --config private_submission/config.json
```

The final command validates the required fields and creates
`private_submission/YOUR_BITS_ID.pdf`. The report order is GitHub link, live app
link, your real lab screenshot, and complete README content. A draft can be
generated without the screenshot/URLs using `--draft`; its missing fields are
clearly marked and it must not be submitted.

Keep public README links current and commit the README changes. The README
includes dataset attribution, all six metrics, observations for every model,
the overall winner and reproducibility details.

## 9. Final check

- Public GitHub repository link opens and contains all required files.
- Public Streamlit link opens the app and CSV upload works.
- Your own BITS Virtual Lab execution is complete and the genuine screenshot is included.
- README metrics match the latest training run and are included in the PDF.
- PDF has your correct ID and working clickable links.
- Submit the single PDF through the required course portal by the announced deadline.

Read `EXPLANATION.md` for the purpose of every file and simple explanations.
