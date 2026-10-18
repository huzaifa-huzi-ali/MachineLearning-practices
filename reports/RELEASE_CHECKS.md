# Evaluated-pipeline release checks

The accuracy gate and full-data refit were removed at the user's request.
The app loads the exact evaluation artifacts; no new all-data model is required.

- Both notebooks executed fully after the change: 13 Titanic code cells and
  12 Heart Disease code cells. Two plots are embedded in each notebook.
- The training, CV, and held-out test results reproduce the previous experiment.
- Each newly generated notebook model matches the original saved model byte for byte.
- `python -m pytest -q tests --junitxml=reports/deployment_tests.xml`:
  14 tests passed. Both Streamlit forms predict in normal mode without a preview
  flag. Actual test predictions are reproduced and input validation remains active.
- `python http_smoke.py`: local health and app HTML verified; the owned server
  is terminated by the check. An initial runner invocation used the wrong source
  path before the test started; correcting that path resolved it.
- Model fingerprints are checked before loading the artifacts in the app.
- No full-data fit and no accuracy condition remains in notebook source.

This verifies local source and inference. Public publication remains incomplete
because Streamlit requires account sign-in and terms acceptance. No live URL has
been invented, and no external account terms were accepted on the user's behalf.

Original model fingerprints:

- Heart: `b4c509696de62b4b6bbb1aa687640881fb9256e2ed9073a69b314235450b84be`
- Titanic: `8530b7f56185a63e88a0527dd032b2025c7c584b70581cbe90a6dfe7c0cefb69`
