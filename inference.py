"""Validate raw form inputs and load fitted sklearn pipelines; never train here."""
from pathlib import Path
import json
import math
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
FEATURES = {
    'titanic': ['PClass', 'Gender', 'Sibling', 'Embarked'],
    'heart': ['cp', 'thalach', 'oldpeak', 'thal'],
}
TITANIC_CATEGORIES = {
    'PClass': ['First', 'Second', 'Third'],
    'Gender': ['Female', 'Male'],
    'Sibling': ['Zero', 'One', 'Two', 'Three'],
    'Embarked': ['Cherbourg', 'Queenstown', 'Southampton'],
}

def validate_input(task, values):
    if task not in FEATURES:
        raise ValueError('Choose Titanic or Heart Disease.')
    if not isinstance(values, dict) or set(values) != set(FEATURES[task]):
        raise ValueError('Provide exactly these inputs: ' + ', '.join(FEATURES[task]))
    result = dict(values)
    if task == 'titanic':
        for key, allowed in TITANIC_CATEGORIES.items():
            if values[key] not in allowed:
                raise ValueError(f'Invalid {key}; choose one of {allowed}.')
    else:
        for key in FEATURES['heart']:
            try:
                if isinstance(values[key], bool):
                    raise ValueError()
                result[key] = float(values[key])
            except (TypeError, ValueError):
                raise ValueError(f'{key} must be a finite number.') from None
            if not math.isfinite(result[key]):
                raise ValueError(f'{key} must be a finite number.')
        if result['cp'] not in [1, 2, 3, 4]:
            raise ValueError('cp must use the UCI codes 1, 2, 3, or 4.')
        if result['thal'] not in [3, 6, 7]:
            raise ValueError('thal must use the UCI codes 3, 6, or 7.')
        if not 0 < result['thalach'] <= 300:
            raise ValueError('Maximum exercise heart rate must be greater than 0 and at most 300 bpm.')
        if not 0 <= result['oldpeak'] <= 20:
            raise ValueError('oldpeak must be between 0 and 20 for this demonstration.')
    return pd.DataFrame([result], columns=FEATURES[task])

def load_bundle(task):
    if task not in FEATURES:
        raise ValueError('Unknown task.')
    metrics = json.loads((ROOT / f'reports/{task}_metrics.json').read_text())
    artifact = ROOT / f'artifacts/{task}_evaluation.joblib'
    import hashlib
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != metrics['artifact_sha256']:
        raise ValueError('Model fingerprint differs from the recorded evaluated artifact.')
    model = joblib.load(artifact)
    if list(model.feature_names_in_) != FEATURES[task]:
        raise ValueError('Saved model input schema does not match this app.')
    return model, metrics

def predict_one(task, values):
    raw = validate_input(task, values)
    model, metrics = load_bundle(task)
    return int(model.predict(raw)[0]), metrics
