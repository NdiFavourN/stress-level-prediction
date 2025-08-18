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
    MHH = st.sidebar.slider('Mental health history (MHH)', 0.0, 1.0, 0.1)
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
    SS = st.sidebar.slider('Social support (SS)', 0.0, 3.0, 0.1)
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


def generate_advice(input_data):
    advice = {}
    for key, value in input_data.items():
#----------- Extract the single value from each column/Series----------------------
        i=1
       # value = value_series.iloc[i]
        if key == "anxiety_level":
            if value[i] >= 15 and value < 20:
               AXL= "your anxiety levels are rising i would advise you prioritize a healthy lifestyle and seek professional guidance."
               advice['anxiety_level'] = AXL
            elif value >20:
                AXL="your anxiety levels are severe i advise you to seek professional help immediately and create a reliable support system"
                advice['anxiety_level'] = AXL
            else :
                AXL="your anxiety levels are normal i would advise you to maintain a balanced lifestyle and stay aware of your mental health."
                advice['anxiety_level'] = AXL
        elif key == "self_esteem":
            if value >= 15 and value < 20:
                SE="your self-esteem is slightly average. Consider engaging in activities that promote self-confidence, self-care and building on your strengths."
                advice['self_esteem'] = SE
            elif value > 20:
                SE="your self-esteem is normal. Keep up the good work and continue to nurture your self-worth and maintain a growth mindset."
                advice['self_esteem'] = SE
            else:
                SE="your self-esteem is low. I strongly recommend seeking support from a mental health professional and challenge negative thoughts."
                advice['self_esteem'] = SE
        elif key == "mental_health_history":
            if value >= 0.5 and value < 0.7:
                MHH="your mental health history is mild  I advice that you maintain consistent self care and identify triggers."
                advice['mental_health_history'] = MHH
            elif value >0.7:
                MHH="you have a significant mental health history. I strongly recommend seeking support from a mental health professional and focus on symptom management"
                advice['mental_health_history'] = MHH
            else:
                MHH="your mental health history is minimal. Continue to prioritize self-care, monitor your mental well-being and promote proactive wellness."
                advice['mental_health_history'] = MHH
        elif key == "depression":
            if value >= 15 and value < 20:
                DPS="your depression levels are rising i would advise you prioritize a healthy lifestyle, establish a daily routine and focus on small acheivable goals."
                advice['depression'] = DPS
            elif value > 20:
                DPS="your depression levels are severe i advise you to seek professional help immediately and ensure a support system is activated."
                advice['depression'] = DPS
            else:
                DPS="your depression levels are normal i would advise you to maintain a balanced lifestyle and stay aware of your mental health and engage in joyful activities."
                advice['depression'] = DPS
        elif key == "headache":
            if value >= 2.5 and value < 3.5:
                HA="your headache levels are rising i would advise you to get a Medical Consultation and practice stress management."
                advice['headache'] = HA
            elif value > 3.5:
                HA="your headache levels are severe i advise you to seek immediate medical attention and avoid self medication"
                advice['headache'] = HA
            else:
                HA="your headache levels are normal i would advise you to drink alot of water and rest. Also try to identify triggers"
                advice['headache'] = HA
        elif key == "blood_pressure":
            if value >= 1.5 and value < 2.5:
                BP="your blood pressure levels are rising i would advise you to monitor your diet, exercise regularly and consult a healthcare professional."
                advice['blood_pressure'] = BP
            elif value > 2.5:
                BP="your blood pressure levels are severe i advise you to seek immediate medical attention and avoid high-sodium foods."
                advice['blood_pressure'] = BP
            else:
                BP="your blood pressure levels are normal i would advise you to maintain a balanced diet and stay physically active."
                advice['blood_pressure'] = BP
        elif key == "sleep_quality":
            if value >= 2.5 and value < 3.5: 
                SQ="Great! you are getting a decent amount of rest,i advise that you etablish a relaxing bedtime ritual and be consistent."
                advice['sleep_quality'] = SQ
            elif value > 3.5:
                SQ="This is Fantastic! in that case, stick to what is working and be mindful of change"
                advice['sleep_quality'] = SQ
            else:
                SQ="You might be facing significant difficulties with sleep. Consider identifying the 'why' and addressing stress and anxiety"
                advice['sleep_quality'] = SQ
        elif key == "breathing_problem":
            if value > 1:       
                BRP="This might be a symptom of a wide range of conditions. I advise that you should contact a medical professional immediately"
                advice['breathing_problem'] = BRP
        elif key == "noise_level":
            if value >= 2.5 and value < 3.5: 
                NL="Okay, i'd advice you practice environmental awareness and also create quiet zones"
                advice['noise_level'] = NL
            elif value > 3.5:
                NL="It is quite high, i recommend focusing on what is controllable at the moment"
                advice['noise_level'] = NL
            else:
                NL="This is an excellent state to be in! i suggest you perfect the peace and use the quiet  time for growth"
                advice['noise_level'] = NL
        elif key == "living_conditions":
            if value >= 2.5 and value < 3.5: 
                LC="Your living space is generally safe and functional, i advice you practice Micro-tidying and establish a simple routine"
                advice['living_conditions'] = LC
            elif value > 3.5:
                LC="Excellent! Maintain this conditions and do well to appreciate them."
                advice['living_conditions'] = LC
            else:
                LC="Hmmmmm i advise that you focus on your core needs first and begin with 'tiny wins'"
                advice['living_conditions'] = LC
        elif key == "safety":
            if value >= 2.5 and value < 3.5: 
                ST="Your safety levels are generally acceptable, i advise you to remain vigilant and proactive."
                advice['safety'] = ST
            elif value > 3.5:
                ST="Your safety levels are optimal, great job maintaining a secure environment."
                advice['safety'] = ST
            else:
                ST="Your safety levels are concerning, i advise you to take immediate action to mitigate risks."
                advice['safety'] = ST
        elif key == "basic_needs":
            if value >= 2.5 and value < 3.5: 
                BN="Your basic needs are generally met, i advise you to continue prioritizing self-care."
                advice['basic_needs'] = BN
            elif value > 3.5:
                BN="Your basic needs are well taken care of, great job maintaining this balance."
                advice['basic_needs'] = BN
            else:
                BN="You might be struggling to meet your basic needs. Consider reaching out for support."
                advice['basic_needs'] = BN
        elif key == "academic_performance":
            if value >= 2.5 and value < 3.5: 
                ACP="You are doing well! i suggest you improve study habits and not just study time and also focus on 'Deep Learning'"
                advice['academic_performance'] = ACP
            elif value >3.5:
                ACP="Nice work there! keep up the good work and cultivate a growth mindset"
                advice['academic_performance'] = ACP
            else:
                ACP="you might be facing some challenges academically. i suggest you address the 'what' and 'why' of the situation, breakdown the problem and connect yourself to resources."
                advice['academic_performance'] = ACP
        elif key == "study_load":
            if value >= 2.5 and value < 3.5: 
                SL="You might feel a bit overwhelmed, i advice that you improve your time management and practice micro-breaks"
                advice['study_load'] = SL
            elif value >3.5:
                SL="the load might be heavy on you, I suggest you breakdown the problem and communicate with an authority figure. "
                advice['study_load'] = SL
            else:
                SL="Good, I suggest you leverage your free time to build new skills and plan for the future."
                advice['study_load'] = SL
        elif key == "teacher_student_relationship":
            if value >= 2.5 and value < 3.5: 
                TSR="Your relationship with your teacher is generally positive, i suggest you maintain open communication and seek feedback."
                advice['teacher_student_relationship'] = TSR
            elif value > 3.5:
                TSR="Excellent! Your relationship with your teacher is strong, keep up the good work."
                advice['teacher_student_relationship'] = TSR
            else:
                TSR="It seems there might be some challenges in your relationship with your teacher. I suggest addressing any concerns directly and seeking support if needed."
                advice['teacher_student_relationship'] = TSR
        elif key == "future_career_concerns":
            if value >= 2.5 and value < 3.5: 
                FCC="Your future career concerns are valid, I suggest exploring different career options and seeking guidance."
                advice['future_career_concerns'] = FCC
            elif value > 3.5:
                FCC="Great! You have a clear vision for your future career, keep up the good work."
                advice['future_career_concerns'] = FCC
            else:
                FCC="It seems you might be feeling uncertain about your future career. I suggest seeking mentorship and exploring your interests."
                advice['future_career_concerns'] = FCC
        elif key == "social_support":
            if value >= 1.5 and value < 2.5: 
                SS="Your social support is generally adequate, I suggest reaching out to friends or family for additional support."
                advice['social_support'] = SS
            elif value > 2.5:
                SS="Great! You have a strong support network, keep nurturing these relationships."
                advice['social_support'] = SS
            else:
                SS="It seems you might be lacking social support. I suggest seeking connections and building a support system."
                advice['social_support'] = SS
        elif key == "peer_pressure":
            if value >= 2.5 and value < 3.5: 
                PP="Your peer pressure levels are generally manageable, I suggest staying true to your values and seeking support if needed."
                advice['peer_pressure'] = PP
            elif value > 3.5:
                PP="Great! You have a strong sense of self and are able to resist peer pressure effectively."
                advice['peer_pressure'] = PP
            else:
                PP="It seems you might be struggling with peer pressure. I suggest finding supportive friends and setting clear boundaries."
                advice['peer_pressure'] = PP
        elif key == "extracurricular_activities":
            if value >= 2.5 and value < 3.5: 
                EXA="Your involvement in extracurricular activities is generally positive, I suggest continuing to explore your interests and seek balance."
                advice['extracurricular_activities'] = EXA
            elif value > 3.5:
                EXA="Excellent! You are highly engaged in extracurricular activities, keep up the great work."
                advice['extracurricular_activities'] = EXA
            else:
                EXA="It seems you might not be very involved in extracurricular activities. I suggest exploring new opportunities and finding activities that interest you."
                advice['extracurricular_activities'] = EXA
        elif key == "bullying":
            if value >= 2.5 and value < 3.5: 
                BUL="Your bullying experiences are generally manageable, I suggest seeking support if needed."
                advice['bullying'] = BUL
            elif value > 3.5:
                BUL="It seems you might be struggling with bullying. I suggest finding supportive friends and setting clear boundaries."
                advice['bullying'] = BUL
            else:
                BUL="Great! You have a strong sense of self and are able to resist bullying effectively."
                advice['bullying'] = BUL
        else:
            print("no such key.")
        i=i+1
        return advice

#-------------function calling------------------
input_data = user_input_features()
advice = generate_advice(input_data)

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
#for key, value in advice.items():
st.markdown(advice)
        #st.markdown(f" For ***{key}***:  {value}")
st.markdown("---")        
st.write("Disclaimer: The dataset used for this app has known ethical issues. This app is for educational purposes only.")