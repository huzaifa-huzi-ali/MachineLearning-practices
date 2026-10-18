"""Check the actual saved evaluated models and ordinary app prediction flows."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference import FEATURES, load_bundle, predict_one, validate_input

@pytest.mark.parametrize('task', ['heart', 'titanic'])
def test_served_pipeline_reproduces_every_recorded_test_prediction(task):
    model, metrics = load_bundle(task)
    rows = pd.read_csv(ROOT / f'reports/{task}_test_predictions.csv')
    predictions = model.predict(rows[FEATURES[task]])
    np.testing.assert_array_equal(predictions, rows.predicted.to_numpy())
    assert (predictions == rows.actual).sum() == metrics['correct']
    assert metrics['full_data_refit'] is False
    assert metrics['deployment_policy'] == 'serve_evaluated_pipeline'
    assert not (ROOT / f'artifacts/{task}_deployment.joblib').exists()

@pytest.mark.parametrize('task', ['heart', 'titanic'])
def test_no_accuracy_gate_and_original_model_bytes_are_preserved(task):
    model, metrics = load_bundle(task)
    assert metrics['cv_accuracy'] < .8  # These actual models must load normally.
    assert hashlib.sha256((ROOT / f'artifacts/{task}_evaluation.joblib').read_bytes()).hexdigest() == metrics['artifact_sha256']

@pytest.mark.parametrize('values', [
    {}, {'cp':4,'thalach':150,'oldpeak':1},
    {'cp':0,'thalach':150,'oldpeak':1,'thal':3},
    {'cp':4,'thalach':150,'oldpeak':1,'thal':2},
    {'cp':4,'thalach':float('nan'),'oldpeak':1,'thal':3},
    {'cp':4,'thalach':0,'oldpeak':1,'thal':3},
    {'cp':4,'thalach':150,'oldpeak':-1,'thal':3},
    {'cp':4,'thalach':150,'oldpeak':1,'thal':3,'num':1}])
def test_invalid_heart_inputs_stay_rejected(values):
    with pytest.raises(ValueError):
        validate_input('heart', values)

def test_both_app_forms_predict_without_preview_mode(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.delenv('ML_LOCAL_EVALUATION_PREVIEW', raising=False)
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
    for name in ['Heart Disease', 'Titanic Survival']:
        app.sidebar.radio[0].set_value(name).run()
        assert not app.exception
        assert len(app.button) == 1
        app.button[0].click().run()
        assert not app.exception
        assert len(app.success) == 1
        assert 'Predicted class:' in app.success[0].value

def test_notebooks_contain_no_full_data_fit_or_accuracy_condition():
    for path in (ROOT / 'notebooks').glob('*.ipynb'):
        nb=json.loads(path.read_text())
        source='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
        assert 'if accepted:' not in source
        assert '.fit(X, y)' not in source
        assert 'eligible_for_final_refit' not in source
        assert 'THRESHOLD' not in source
