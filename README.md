# Deploy the evaluated Titanic and Heart Disease pipelines

Both prediction options use the exact saved evaluation artifacts. There is no
accuracy gate and no extra fit on combined development and test records.

| Task | CV accuracy | Test accuracy | Records used to train the evaluated model |
|---|---:|---:|---:|
| Titanic | 76.25% | 65.00% (13/20) | 80 |
| Heart Disease | 79.32% | 86.89% (53/61) | 242 |

The model fingerprints in `reports/*_metrics.json` identify the original saved
artifacts. The app checks those fingerprints and then loads preprocessing and
classifier together. The app never trains a model.

## Run locally

Use Python 3.12. Extract this folder, open a terminal inside it, then run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Select Heart Disease or Titanic in the sidebar. Both work in ordinary app mode;
no preview environment variable or accuracy override is needed.

## Publish on Streamlit Community Cloud

1. Upload the extracted contents of this folder to your intended GitHub repository.
2. Keep `app.py`, `inference.py`, and `requirements.txt` at repository root, with
   the `artifacts/`, `reports/`, and `.streamlit/` directories alongside them.
3. Open Streamlit Community Cloud, sign in, and select Create app.
4. Choose that repository, branch, and `app.py`; select Python 3.12.
5. Deploy and verify both prediction forms against the saved model outputs.
6. Place the verified live link at the top of each notebook.

The initial selection can be set using `?task=heart` or `?task=titanic` on the
actual published app URL. No public URL exists until publication is completed.
Account sign-in and any required terms acceptance must be completed by the user.

## Files needed by the app

- `app.py`: prediction interface.
- `inference.py`: raw-input validation and model loading.
- `requirements.txt`: matching Python package versions.
- `.streamlit/config.toml`: app appearance.
- `artifacts/heart_evaluation.joblib` and `artifacts/titanic_evaluation.joblib`:
  the actual evaluated pipelines, with their original bytes preserved.
- `reports/heart_metrics.json` and `reports/titanic_metrics.json`: real scores,
  schema metadata, and model fingerprints.

The two standalone notebooks are included in `notebooks/`. Their Phase 9 records
the evaluated artifact for deployment; it performs no full-data fit. Training
and evaluation phases still reproduce the earlier experiment when Run All is used.

## Accuracy improvement direction

Titanic: the supplied 100-record four-input sample has conflicting outcomes for
matching profiles. Keep this assignment reproduction; a separate larger-data
experiment with Age, Fare, Parch, and useful family-size features is the stronger
next step. There is no guaranteed gain from additional tuning of the current sample.

Heart: retain exactly four raw inputs, compare a small predefined set of quartets
(for example replacing thalach with ca), then tune around the current SVC settings
or compare Logistic Regression / Gradient Boosting. Select only measurements
available at prediction time. Use nested CV for the expanded selection experiment,
and preserve the current exposed holdout as historical evidence.

This is an educational demonstration; the Heart Disease output is a prediction
of a historical dataset label, not a clinical diagnosis.
