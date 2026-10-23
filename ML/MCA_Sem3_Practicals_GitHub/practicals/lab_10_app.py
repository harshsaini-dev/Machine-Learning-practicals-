"""Optional Streamlit UI for both Lab 10 capstone projects.

Install `pip install streamlit`, train `python lab_10.py`, then run:
`streamlit run lab_10_app.py`.
"""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'outputs' / 'lab_10'

st.set_page_config(page_title='MCA201P Capstone ML Demos',page_icon='🧪',layout='centered')
st.title('Capstone Machine Learning — Lab 10')
st.warning('SIMULATED EDUCATIONAL DEMOS ONLY. No real lending decisions or medical '
           'diagnoses. Do not submit personal, financial or patient records.')


def load_model(name):
    location = MODEL_DIR / name
    if not location.exists():
        st.error('Model not found. First run: python lab_10.py')
        st.stop()
    return joblib.load(location)  # Only load trusted, self-generated models.


page = st.radio('Select project', ['Loan Approval Prediction', 'Symptom-Pattern Classification'])
if page == 'Loan Approval Prediction':
    st.subheader('Loan decision simulation')
    with st.form('loan'):
        income = st.number_input('Monthly income (INR)', min_value=0.0,value=58000.0,step=1000.0)
        amount = st.number_input('Loan amount (INR)',min_value=0.0,value=400000.0,step=10000.0)
        term = st.select_slider('Term (months)',options=[24,36,48,60])
        history = st.selectbox('Recorded credit history', [1,0],format_func=lambda n: 'Positive' if n else 'No positive history')
        employment = st.selectbox('Employment', ['Salaried','Self-employed','Unemployed'])
        education = st.selectbox('Education', ['Graduate','Not Graduate'])
        married = st.selectbox('Marital status', ['Married','Unmarried'])
        submit = st.form_submit_button('Predict (simulation)')
    if submit:
        model = load_model('loan_approval_pipeline.joblib')
        row = pd.DataFrame([{'income_monthly_inr': income,'loan_amount_inr':amount,
             'loan_term_months':term, 'credit_history':history,'employment':employment,
             'education':education, 'marital_status':married}])
        outcome = int(model.predict(row)[0])
        st.info(f'Simulated class: {"Approved" if outcome else "Not Approved"}')
        st.caption('Synthetic data and model labels are not real eligibility criteria. '
                   'No applicant should be judged using this application.')
else:
    st.subheader('Symptom pattern — NOT a diagnosis')
    with st.form('symptoms'):
        features = ['fever','cough','sore_throat','sneezing','itchy_eyes',
                    'headache','nausea','body_aches','fatigue']
        values = {name:int(st.checkbox(name.replace('_',' ').title())) for name in features}
        values['duration_days'] = st.slider('Duration in days',1,30,4)
        submit = st.form_submit_button('Classify simulated pattern')
    if submit:
        model = load_model('symptom_pattern_pipeline.joblib')
        predicted = str(model.predict(pd.DataFrame([values]))[0])
        st.info('Closest simulated pattern: ' + predicted)
        st.error('This is not a clinical assessment. Symptoms overlap across diseases; '
                 'seek qualified medical evaluation for health concerns.')
