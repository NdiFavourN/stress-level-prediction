import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# --- Load The Model and Scaler ---
@st.cache_resource 
def load_model_and_scaler():
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

model, scaler = load_model_and_scaler()

# --- Page Configuration ---
st.set_page_config(
    page_title="Stress Level Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"  
)

st.markdown(
    """
    <style>
        .main-title {
            font-size: 4rem !important;
            font-weight: bold;
            color: white;
            text-align: center;
            animation: fadeInDown 1s ease-in-out;
        }
        @keyframes fadeInDown {
            0% { opacity: 0; transform: translateY(-20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        .prediction-result {
            font-size: 1.5rem;
            color: #1565c0;
            font-weight: bold;
            text-align: center;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
    </style>
    """,
    unsafe_allow_html=True
)


# --- App Title and Description ---
st.markdown("<div class='main-title'>Stress Level Detector 🧠</div>", unsafe_allow_html=True)
st.write(
    """
    This app predicts the **Stress Level** of individuals based on their input data.
    Enter the details of the individual, and the model will estimate their stress level.
    This is based on a machine learning model trained on a dataset of psychological factors.
    """
)
st.markdown("---")
# --- Sidebar ---
st.sidebar.header("Input Features")

def user_input_features():
    AXL = st.sidebar.slider('Anxiety Level (AXL)', 0.0, 30.0, 1.0)
    SE = st.sidebar.slider('Self Esteem (SE)', 0.0, 30.0, 1.0)
    MHH = st.sidebar.slider('Mental health history (MHH)', 0.0, 1.0, 0.01)
    DPS = st.sidebar.slider('Depression (DPS)', 0.0, 30.0, 1.0)
    HA = st.sidebar.slider('Headache (NOX)', 0.0, 5.0, 0.1)
    BP = st.sidebar.slider('Blood pressure (BP)', 0.0, 3.0, 0.01)
    SQ = st.sidebar.slider('Sleep quality (SQ)', 0.0, 5.0, 1.0)
    BRP = st.sidebar.slider('Breathing problem (BRP)', 0.0, 5.0, 1.0)
    NL = st.sidebar.slider('Noise Level (NL)', 0.0, 5.0, 1.0)
    LC = st.sidebar.slider('Living Conditions (LC)', 0.0, 5.0, 1.0)
    ST= st.sidebar.slider('Safety (ST)', 0.0, 5.0, 1.0)
    BN = st.sidebar.slider('Basic needs (BN)', 0.0, 5.0, 1.0)
    ACP= st.sidebar.slider('Academic peformance (ACP)', 1.0, 5.0, 1.0)
    SL = st.sidebar.slider('Study Load (SL)', 0.0, 5.0, 1.0)
    TSR = st.sidebar.slider('Teacher student relationship (TSR)', 0.0, 5.0, 1.0)
    FCC = st.sidebar.slider('Future career concerns (FCC)', 0.0, 5.0, 1.0)
    SS = st.sidebar.slider('Social support (SS)',( 0, 3, 0.1))
    PP = st.sidebar.slider('Peer Pressure (PP)', 0.0, 5.0, 1.0)
    EXA = st.sidebar.slider('Extracurricular activities (EXA)', 0.0, 5.0, 1.0)
    BUL = st.sidebar.slider('Bullying (BUL)', 0.0, 5.0, 1.0)

    data = {
        'anxiety_level': AXL,
        'self_esteem': SE,
        'mental_health_history': MHH,
        'depression': DPS,
        'headache': HA,
        'blood_pressure': BP,
        'sleep_quality': SQ,
        'breathing_problem': BRP,
        'noise_level': NL,
        'living_conditions': LC,
        'safety': ST,
        'basic_needs': BN,
        'academic_performance': ACP,
        'study_load': SL,
        'teacher_student_relationship': TSR,
        'future_career_concerns': FCC,
        'social_support': SS,
        'peer_pressure': PP,
        'extracurricular_activities': EXA,
        'bullying': BUL
    }
    return pd.DataFrame(data, index=[0])

input_data = user_input_features()

# --- Main Panel ---
st.header("Your Input")
st.dataframe(input_data)

if st.sidebar.button("Predict Stress Level"):
    with st.spinner("Calculating stress Level..."):
        time.sleep(2.5)  # Simulate processing delay
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)
        predicted_stress_level = prediction[0]

    st.markdown(f"<div class='prediction-result'>Predicted Stress level: {predicted_stress_level:,.2f}</div>", unsafe_allow_html=True)
    st.balloons()

st.markdown("---")
st.write("Disclaimer: The dataset used for this app has known ethical issues. This app is for educational purposes only.")