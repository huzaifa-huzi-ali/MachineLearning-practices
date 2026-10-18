"""AIC354 prediction demo. Run: python -m streamlit run app.py"""
import json
from pathlib import Path
import streamlit as st
from inference import validate_input, load_bundle

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='ML Systems | AIC354', page_icon='◈', layout='centered')
st.markdown('''<style>
 .block-container {max-width:900px;padding-top:2.4rem}
 h1 {letter-spacing:-.045em}
 [data-testid="stMetric"] {background:#f1f5f9;border-radius:12px;padding:14px}
 .eyebrow {font-size:.78rem;letter-spacing:.14em;color:#0891b2;font-weight:700}
 </style>''', unsafe_allow_html=True)
requested = str(st.query_params.get('task', 'heart'))
initial = 0 if requested == 'heart' else 1
choice = st.sidebar.radio('Prediction system', ['Heart Disease', 'Titanic Survival'], index=initial)
task = 'heart' if choice == 'Heart Disease' else 'titanic'
st.sidebar.markdown('**Muhammad Huzaifa Ali**  \nSP24-BAI-034  \nMachine Learning Fundamentals')
st.sidebar.caption('COMSATS University Islamabad, Lahore Campus')
st.markdown('<div class="eyebrow">AIC354 / END-TO-END MACHINE LEARNING</div>', unsafe_allow_html=True)
st.title(choice + ' Predictor')
st.write('Enter the four recorded attributes to apply the saved prediction pipeline.')

metrics_path = ROOT / f'reports/{task}_metrics.json'
if not metrics_path.exists():
    st.error('Model artifacts are missing. Run the corresponding notebook first.')
    st.stop()
metrics = json.loads(metrics_path.read_text())
a, b, c = st.columns(3)
a.metric('Validation accuracy', f"{metrics['cv_accuracy']:.1%}")
b.metric('Held-out test accuracy', f"{metrics['test_accuracy']:.1%}")
c.metric('Test records', f"{metrics['correct']} / {metrics['test_records']} correct")
st.caption('Predictions use the same development-trained pipeline that produced these test results.')

try:
    @st.cache_resource
    def get_model(task_name, artifact_stamp):
        return load_bundle(task_name)
    artifact_path = ROOT / f'artifacts/{task}_evaluation.joblib'
    model, _ = get_model(task, artifact_path.stat().st_mtime_ns)

except (ValueError, FileNotFoundError) as exc:
    st.warning(str(exc))
    st.stop()

if task == 'heart':
    st.caption('Educational demonstration of historical UCI disease-presence labels. Not a clinical diagnosis.')
    cp_labels = {1: 'Typical angina', 2: 'Atypical angina', 3: 'Non-anginal pain', 4: 'Asymptomatic'}
    thal_labels = {3: 'Normal', 6: 'Fixed defect', 7: 'Reversible defect'}
    with st.form('heart_prediction'):
        left, right = st.columns(2)
        cp = left.selectbox('Chest pain type', list(cp_labels), format_func=cp_labels.get)
        thal = right.selectbox('Thallium test result', list(thal_labels), format_func=thal_labels.get)
        thalach = left.number_input('Maximum exercise heart rate (bpm)', min_value=1, max_value=300, value=150)
        oldpeak = right.number_input('ST depression (oldpeak)', min_value=0.0, max_value=20.0, value=1.0, step=0.1)
        submitted = st.form_submit_button('Predict disease presence', type='primary')
    values = {'cp': cp, 'thalach': thalach, 'oldpeak': oldpeak, 'thal': thal}
else:
    with st.form('titanic_prediction'):
        left, right = st.columns(2)
        pclass = left.selectbox('Passenger class', ['First', 'Second', 'Third'])
        gender = right.selectbox('Gender', ['Female', 'Male'])
        sibling = left.selectbox('Siblings / spouses aboard', ['Zero', 'One', 'Two', 'Three'])
        embarked = right.selectbox('Embarkation port', ['Cherbourg', 'Queenstown', 'Southampton'])
        submitted = st.form_submit_button('Predict survival', type='primary')
    values = {'PClass': pclass, 'Gender': gender, 'Sibling': sibling, 'Embarked': embarked}

if submitted:
    try:
        raw = validate_input(task, values)
        prediction = int(model.predict(raw)[0])
        if task == 'heart':
            message = 'Predicted class: disease present' if prediction else 'Predicted class: disease absent'
        else:
            message = 'Predicted class: survived' if prediction else 'Predicted class: not survived'
        st.success(message)
        st.caption('This is a predicted dataset class. It does not establish an individual outcome.')
    except ValueError as exc:
        st.error(str(exc))

st.divider()
with st.expander('Evaluation details'):
    st.write('Selected model:', metrics['model'])
    st.write('Disease / survival class recall:', f"{metrics['recall']:.1%}")
    st.write('F1 score:', round(metrics['f1'], 3))
    st.write('95% test-accuracy interval:', [round(v, 3) for v in metrics['test_accuracy_wilson_95']])
    st.caption('Validation used five stratified folds. Final testing used a fixed reserved 20% split.')
