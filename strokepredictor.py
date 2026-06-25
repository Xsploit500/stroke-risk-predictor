## Importing necessary packages
import joblib
import streamlit as st
import pandas as pd
import numpy as np



## Important Information
st.title("Stroke Information & Education")

st.markdown("""
Before making any predictions, I believe it is important to understand what a stroke is, its causes, risk factors, symptoms, complications, and prevention measures.
""")


with st.expander("1. Overview"):
    st.subheader("What is a Stroke?")
    st.write("""
    A stroke occurs when the blood supply to part of the brain is interrupted or reduced, preventing brain tissue from getting oxygen and nutrients. Brain cells begin to die within minutes.
    """)
    st.write("**Types of Stroke:**")
    st.markdown("""
    - **Ischemic Stroke:** Caused by blocked or narrowed arteries, often due to fatty deposits or blood clots. Most common type.  
    - **Hemorrhagic Stroke:** Caused by a blood vessel in the brain leaking or bursting, increasing pressure on brain cells.  
    - **Transient Ischemic Attack (TIA):** Temporary disruption of blood flow to the brain; does not cause lasting damage but increases risk of future stroke.
    """)
    st.image("images/what_is_stroke.png", caption="What is Stroke", use_container_width=True)


with st.expander("2. Symptoms"):
    st.write("""
    Common symptoms include:
    - Trouble speaking or understanding speech
    - Numbness, weakness, or paralysis, often on one side of the body
    - Vision problems in one or both eyes
    - Sudden severe headache, possibly with vomiting, dizziness, or change in consciousness
    - Trouble walking, loss of balance, or coordination
    """)
    st.image("images/stroke_symptoms.jpg", caption="Stroke Symptoms (FAST)", use_container_width=True)
    st.subheader("FAST Method")
    st.markdown("""
    - **F (Face):** Check for drooping on one side  
    - **A (Arms):** Can the person raise both arms?  
    - **S (Speech):** Is speech slurred or unusual?  
    - **T (Time):** Call 911 immediately if any of these signs are present
    """)


with st.expander("3. Causes"):
    st.write("""
    - **Ischemic Stroke:** Blocked or narrowed blood vessels due to fatty deposits, blood clots, or debris from the heart.  
    - **Hemorrhagic Stroke:** High blood pressure, anticoagulant overuse, aneurysms, head trauma, cerebral amyloid angiopathy, AVM rupture.  
    - **TIA (Mini-Stroke):** Temporary blockage; warning sign for potential future strokes.
    """)
    st.image("images/cause_of_stroke.png", caption="Stroke Causes", use_container_width=True)

    
with st.expander("4. Risk Factors"):
    st.subheader("Lifestyle Factors")
    st.write("- Obesity, Physical inactivity, Heavy alcohol use, Illicit drug use (cocaine, methamphetamine)")
    st.subheader("Medical Factors")
    st.write("- High blood pressure, Diabetes, High cholesterol, Heart disease or arrhythmias, Smoking, Obstructive sleep apnea, Personal/family history of stroke or TIA, COVID-19 infection")
    st.subheader("Other Factors")
    st.write("- Age ≥55, Race/ethnicity: African American/Hispanic, Sex: Men higher risk, Hormone use (birth control/estrogen therapy)")
    st.image("images/stroke_risk_factors.png", caption="Stroke Risk Factors", use_container_width=True)

    
with st.expander("5. Complications"):
    st.write("""
    - Paralysis or loss of muscle movement
    - Speech and swallowing difficulties
    - Memory loss or cognitive issues
    - Emotional changes or depression
    - Pain, numbness, or tingling
    - Changes in behavior or self-care ability
    """)
    st.image("images/complications_of_stroke.jpg", caption="Stroke Complications", use_container_width=True)

    
with st.expander("6. Prevention"):
    st.subheader("Lifestyle Measures")
    st.write("""
    - Control blood pressure
    - Reduce cholesterol and saturated fat
    - Quit smoking
    - Maintain healthy weight
    - Manage diabetes
    - Eat a diet rich in fruits and vegetables (Mediterranean diet recommended)
    - Exercise regularly (150 min/week moderate or 75 min/week vigorous)
    - Moderate alcohol consumption
    - Treat obstructive sleep apnea
    - Avoid illicit drugs
    """)
    st.subheader("Preventive Medicines")
    st.write("""
    For high-risk patients:
    - **Anti-platelet drugs:** aspirin, clopidogrel  
    - **Anticoagulants:** heparin, warfarin, dabigatran, rivaroxaban, apixaban, edoxaban
    """)
    st.image("images/preventing_stroke.jpg", caption="Stroke Prevention Measures", use_container_width=True)



## Loading our trained and developed pipeline for new predictions
pipeline = joblib.load('stroke_prediction_pipeline.pkl')



## Creating the User Interface (UI)
st.title('Stroke Prediction App')
st.write('Predict the likelihood of stroke based on user health information.')

st.header('Enter Patient Information')

# Numeric Inputs
age = st.number_input('Age', min_value = 0, max_value = 120, value = 60)
avg_glucose_level = st.number_input('Average Glucose Level', min_value = 50.0, max_value = 300.0, value = 100.0)
bmi = st.number_input('BMI', min_value = 10.0, max_value = 50.0, value = 25.0)

# Categorical Inputs
hypertension = st.selectbox('Hypertension', ['Yes', 'No'])
heart_disease = st.selectbox('Heart Disease', ['Yes', 'No'])
gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
ever_married = st.selectbox('Ever Married?', ['Yes', 'No'])
work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
Residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

# Binary encodings
hypertension_val = 1 if hypertension == 'Yes' else 0
heart_disease_val = 1 if heart_disease == 'Yes' else 0

# Engineered Features
age_hypertension = age * hypertension_val
smoking_heart = (1 if smoking_status == 'smokes' else 0) * heart_disease_val

# Define BMI Categories
def categorize(bmi):
    if bmi < 18.5:
        return 'underweight'
    elif bmi >= 18.5 and bmi < 25:
        return 'normal'
    elif bmi >= 25 and bmi < 30:
        return 'overweight'
    else:
        return 'obese'

bmi_category = categorize(bmi)



## Combining Features into a dataframe
input_data = pd.DataFrame({
    'age': [age],
    'hypertension': [hypertension_val],
    'heart_disease': [heart_disease_val],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'age_hypertension': [age_hypertension],
    'smoking_heart': [smoking_heart],
    'gender': [gender],
    'ever_married': [ever_married],
    'work_type': [work_type],
    'Residence_type': [Residence_type],
    'smoking_status': [smoking_status],
    'bmi_category': [bmi_category]
})



## Prediction Button
if st.button('Predict Stroke Likelihood'):
    prediction_proba = pipeline.predict_proba(input_data)[0][1]
    prediction = pipeline.predict(input_data)[0]

    st.subheader('Prediction Results')
    st.write(f'**Stroke Probability:** {prediction_proba:.2%}')
    if prediction == 1:
        st.warning('\u26A0 The model predicts a high likelihood of stroke.')
    else:
        st.success('\u2705 The model predicts a low likelihood of stroke.')
