import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from utils.feature_engineering import (
    engineer_features,
    transform_target_encoding,
)
from utils.predict_proba import predict_probe

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

@st.cache_data()
def load_input_options():
    with open(r"C:\Users\harwi\Downloads\student_health_risk_prediction_system\data\input_options.json") as f:
        input_options = json.load(f)
    return input_options

@st.cache_data()
def load_sleep_probe():
    sleep_probe = joblib.load(r"C:\Users\harwi\Downloads\student_health_risk_prediction_system\models\sleep_probe_1.pkl")
    return sleep_probe

@st.cache_data()
def load_stress_probe():
    stress_probe = joblib.load(r"C:\Users\harwi\Downloads\student_health_risk_prediction_system\models\stress_probe_1.pkl")
    return stress_probe

@st.cache_data()
def load_weights():
    best_weights = np.load(r'C:\Users\harwi\Downloads\student_health_risk_prediction_system\models\best_weights_catboost.npy')
    return best_weights

@st.cache_resource()
def load_artifacts():
    artifacts = joblib.load(r'C:\Users\harwi\Downloads\student_health_risk_prediction_system\models\final_catboost.pkl')
    return artifacts

input_options = load_input_options()
artifacts = load_artifacts()
sleep_probe = load_sleep_probe()
stress_probe =load_stress_probe()
weights = load_weights()

model = artifacts['model']
label_encoder = artifacts['label_encoder']

st.title("Student Health Risk Assessment")
st.caption(
    "Enter your health and lifestyle information to estimate your health risk."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Lifestyle")

    gender = st.selectbox('Gender', input_options['gender'])
    diet_type = st.selectbox('Diet Type', input_options['diet_type'])
    smoking_alcohol = st.selectbox('Smoking Alcohol', input_options['smoking_alcohol'])
    physical_activity_level = st.selectbox('Physical Ativity Level', input_options['physical_activity_level'])

with col2:
    st.subheader("Health Metrics")

    heart_rate = float(st.number_input('Heart Rate', min_value=50.0, max_value=108.0))
    bmi = float(st.number_input('BMI', min_value=16.0, max_value=35.0))
    calorie_expenditure = st.number_input('Calorie Expenditure', min_value=1200, max_value=3600)
    step_count = st.number_input('Step count', min_value=1000, max_value=15000)


col1, col2 = st.columns(2)

with col1:
    st.subheader("Sleep")
    sleep_duration =float(st.number_input("Sleep Duration (in hours)",min_value=3.0,max_value= 10.0))
    sleep_quality = st.selectbox('Sleep Quality', input_options['sleep_quality'])

with col2:
    st.subheader("Wellness")
    stress_level = st.selectbox('Stress Level', input_options['stress_level'])
    exercise_duration = float(st.number_input('Exercise Duration (in minutes)', min_value=0.0, max_value=100.0))
    water_intake = float(st.number_input('Water Intake(in liters)', min_value=0.5, max_value=4.5))

predict = st.button(
    "Assess Health Risk",
    type="primary",
    use_container_width=True
)

if predict:
    data = [[sleep_duration,heart_rate,bmi,calorie_expenditure,step_count,exercise_duration,
            water_intake,diet_type,stress_level,sleep_quality,physical_activity_level,smoking_alcohol,gender]]
    columns = ['sleep_duration','heart_rate','bmi','calorie_expenditure','step_count','exercise_duration',
            'water_intake','diet_type','stress_level','sleep_quality','physical_activity_level','smoking_alcohol','gender',]
    one_df = pd.DataFrame(data = data, columns= columns)
    one_df_eng = engineer_features(one_df)

    probs_stress = predict_probe(one_df_eng,stress_probe)
    le = stress_probe['label_encoder']
    for i, cls in enumerate(le.classes_):
        one_df_eng[f"p_stress_{cls}"] = probs_stress[:, i]

    probs_sleep = predict_probe(one_df_eng,sleep_probe)
    le = sleep_probe['label_encoder']
    for i, cls in enumerate(le.classes_):
        one_df_eng[f"p_sleep_{cls}"] = probs_sleep[:, i]
    
    one_df_eng_target  = transform_target_encoding(one_df_eng,artifacts['target_encoding_info'])
    

    probs = model.predict_proba(one_df_eng_target)
    weighted_probs = probs * weights
    pred = weighted_probs.argmax(axis=1)
    pred_trans = label_encoder.inverse_transform(pred)

    prediction = pred_trans[0]
    st.session_state.prediction_result = prediction
    st.session_state.raw_probs = probs[0]

if st.session_state.prediction_result is not None:

    prediction = st.session_state.prediction_result
    probs = st.session_state.raw_probs

    if prediction == "fit":
        st.success("Predicted Health Status: FIT")

    elif prediction == "at-risk":
        st.warning("Predicted Health Status: AT-RISK")

    else:
        st.error("Predicted Health Status: UNHEALTHY")

    st.info(
        "The final prediction is determined using an optimized decision rule. "
        "The probabilities shown below are the model's raw probabilities before "
        "applying the decision weights."
    )

    for i, encoded_class in enumerate(model.classes_):
        class_name = label_encoder.inverse_transform(
            [encoded_class]
        )[0]

        probability = probs[i]

        st.write(f"**{class_name.title()}** — {probability:.1%}")
        st.progress(float(probability))

    st.info(
        "This assessment is generated by a machine-learning model for "
        "informational purposes and should not be considered a medical diagnosis."
    )