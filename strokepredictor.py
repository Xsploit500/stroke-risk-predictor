import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb


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


## Train the pipeline on startup
@st.cache_resource
def train_pipeline():
    data = pd.read_csv('stroke.csv')

    # Drop id column and rows where gender is 'Other'
    data = data[data['gender'] != 'Other'].drop(columns=['id'])

    target = data['stroke']
    features = data.drop(columns=['stroke'])

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    # Impute missing bmi values
    group_cols = ['gender', 'age']
    x_train['bmi'] = x_train.groupby(group_cols)['bmi'].transform(
        lambda x: x.fillna(x.median())
    )
    x_train['bmi'] = x_train['bmi'].fillna(features['bmi'].median())

    x_test['bmi'] = x_test.apply(
        lambda row: row['bmi'] if pd.notna(row['bmi'])
        else x_train[(x_train['gender'] == row['gender']) & (x_train['age'] == row['age'])]['bmi'].median(),
        axis=1
    )
    x_test['bmi'] = x_test['bmi'].fillna(x_train['bmi'].median())

    # Feature engineering
    for df in [x_train, x_test]:
        df['age_hypertension'] = df['age'] * df['hypertension']
        df['smoking_heart'] = (df['smoking_status'] == 'smokes').astype(int) * df['heart_disease']
        df['bmi_category'] = df['bmi'].apply(lambda b: 'underweight' if b < 18.5 else ('normal' if b < 25 else ('overweight' if b < 30 else 'obese')))

    numeric_features = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease', 'age_hypertension', 'smoking_heart']
    categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'bmi_category']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    x_train_encoded = preprocessor.fit_transform(x_train)

    smote = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train_encoded, y_train)

    lr = LogisticRegression(C=5.996584841970366, solver='lbfgs', class_weight='balanced', max_iter=1000, random_state=42)
    xgb_model = xgb.XGBClassifier(
        learning_rate=0.025593820005455386, max_depth=7, subsample=0.7468055921327309,
        scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
        eval_metric='logloss', random_state=42
    )
    lgb_model = lgb.LGBMClassifier(
        num_leaves=94, learning_rate=0.12973169683940733, subsample=0.8377746675897602,
        is_unbalance=True, random_state=42, verbose=-1
    )

    roc_auc_scores = np.array([0.8078, 0.7862, 0.7803])
    weights = roc_auc_scores / roc_auc_scores.sum()

    voting_clf = VotingClassifier(
        estimators=[('Logistic Regression', lr), ('XGBoost', xgb_model), ('LightGBM', lgb_model)],
        voting='soft',
        weights=weights
    )
    voting_clf.fit(x_train_resampled, y_train_resampled)

    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', voting_clf)
    ])

    full_pipeline.fit(x_train, y_train)

    return full_pipeline


pipeline = train_pipeline()


## Creating the User Interface (UI)
st.title('Stroke Prediction App')
st.write('Predict the likelihood of stroke based on user health information.')

st.header('Enter Patient Information')

age = st.number_input('Age', min_value=0, max_value=120, value=60)
avg_glucose_level = st.number_input('Average Glucose Level', min_value=50.0, max_value=300.0, value=100.0)
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)

hypertension = st.selectbox('Hypertension', ['Yes', 'No'])
heart_disease = st.selectbox('Heart Disease', ['Yes', 'No'])
gender = st.selectbox('Gender', ['Male', 'Female'])
ever_married = st.selectbox('Ever Married?', ['Yes', 'No'])
work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
Residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

hypertension_val = 1 if hypertension == 'Yes' else 0
heart_disease_val = 1 if heart_disease == 'Yes' else 0

age_hypertension = age * hypertension_val
smoking_heart = (1 if smoking_status == 'smokes' else 0) * heart_disease_val

def categorize(bmi):
    if bmi < 18.5:
        return 'underweight'
    elif bmi < 25:
        return 'normal'
    elif bmi < 30:
        return 'overweight'
    else:
        return 'obese'

bmi_category = categorize(bmi)

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

if st.button('Predict Stroke Likelihood'):
    prediction_proba = pipeline.predict_proba(input_data)[0][1]
    prediction = pipeline.predict(input_data)[0]

    st.subheader('Prediction Results')
    st.write(f'**Stroke Probability:** {prediction_proba:.2%}')
    if prediction == 1:
        st.warning('\u26A0 The model predicts a high likelihood of stroke.')
    else:
        st.success('\u2705 The model predicts a low likelihood of stroke.')
