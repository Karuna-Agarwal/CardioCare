import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model
model = pickle.load(open('svm_model.pkl','rb'))

# Page config
st.set_page_config(page_title="CardioCare AI", layout="wide")

# ---------- CUSTOM STYLE ----------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.stButton>button {
    background-color: #e63946;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.result-box {
    padding: 20px;
    border-radius: 10px;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("""
<h1 style='text-align: center;'>❤️ CardioCare AI</h1>
<p style='text-align: center; font-size:18px;'>
Smart Heart Disease Risk Prediction System
</p>
""", unsafe_allow_html=True)

st.write("---")

# ---------- INPUT ----------
st.header("📝 Patient Information")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age ℹ️", 20, 100, 25,
        help="Age of the person in years")

    gender = st.selectbox("Gender ℹ️", ('M','F'),
        help="Biological gender")

    ChestPainType = st.selectbox("Chest Pain Type ℹ️", ('ATA', 'NAP', 'ASY', 'TA'),
        help="Type of chest pain experienced")

    RestingBP = st.number_input("Resting BP ℹ️", 0, 250, 120,
        help="Normal is around 120")

with col2:
    Cholesterol = st.number_input("Cholesterol ℹ️", 0, 650, 200,
        help="Normal is below 200")

    MaxHR = st.number_input("Max Heart Rate ℹ️", 60, 250, 150,
        help="Max heart rate during activity")

    Oldpeak = st.number_input("Oldpeak ℹ️", -3, 10, 1,
        help="Higher value indicates higher risk")

    FastingBS = st.selectbox("Fasting Blood Sugar ℹ️", (0,1),
        help="1 = High sugar, 0 = Normal")

st.write("---")

col3, col4 = st.columns(2)

with col3:
    RestingECG = st.selectbox("Resting ECG ℹ️", ('Normal', 'ST', 'LVH'),
        help="Heart electrical activity")

with col4:
    ExerciseAngina = st.selectbox("Exercise Angina ℹ️", ('Y','N'),
        help="Chest pain during exercise")

    ST_Slope = st.selectbox("ST Slope ℹ️", ('Up', 'Flat', 'Down'),
        help="Slope of ECG signal")

st.write("---")

# ---------- ENCODING ----------
sex = 1 if gender=='M' else 0
exerciseAngina = 1 if ExerciseAngina=='Y' else 0

RestingECG_LVH = 1 if RestingECG=='LVH' else 0
RestingECG_Normal = 1 if RestingECG=='Normal' else 0
RestingECG_ST = 1 if RestingECG=='ST' else 0

ChestPainType_ASY = 1 if ChestPainType=='ASY' else 0
ChestPainType_ATA = 1 if ChestPainType=='ATA' else 0
ChestPainType_NAP = 1 if ChestPainType=='NAP' else 0
ChestPainType_TA = 1 if ChestPainType=='TA' else 0

st_Slope_dict = {'Up':0,'Down':1,'Flat':2}
st_Slope = st_Slope_dict[ST_Slope]

# ---------- DATA ----------
input_features = pd.DataFrame({
    'Age':[Age], 'RestingBP':[RestingBP], 'Cholesterol':[Cholesterol],
    'FastingBS':[FastingBS], 'MaxHR':[MaxHR], 'Oldpeak':[Oldpeak],
    'sex':[sex], 'exerciseAngina':[exerciseAngina], 
    'RestingECG_LVH':[RestingECG_LVH],
    'RestingECG_Normal':[RestingECG_Normal], 
    'RestingECG_ST':[RestingECG_ST],
    'ChestPainType_ASY':[ChestPainType_ASY],
    'ChestPainType_ATA':[ChestPainType_ATA], 
    'ChestPainType_NAP':[ChestPainType_NAP],
    'ChestPainType_TA':[ChestPainType_TA],
    'st_Slope':[st_Slope]
})

# ---------- PREDICT ----------
if st.button("🔍 Predict Risk"):

    prediction = model.predict(input_features)

    st.subheader("🩺 Result")

    try:
        prob = model.predict_proba(input_features)[0][1]
        st.write(f"### Risk Score: {round(prob*100,2)}%")
        st.progress(float(prob))
    except:
        st.info("Probability not available")

    if prediction[0] == 1:
        st.markdown(
            "<div class='result-box' style='background-color:#ff4d4d; color:white;'>⚠️ High Risk of Heart Disease</div>",
            unsafe_allow_html=True
        )
        st.write("👉 Consult a doctor and improve lifestyle habits.")

    else:
        st.markdown(
            "<div class='result-box' style='background-color:#2ecc71; color:white;'>✅ Low Risk of Heart Disease</div>",
            unsafe_allow_html=True
        )
        st.write("👉 Maintain your current healthy lifestyle.")

# ---------- ABOUT ----------
with st.expander("ℹ️ About Model"):
    st.write("""
    This project uses a Machine Learning model (SVM) to predict heart disease risk.
    
    Inputs include:
    - Age, Blood Pressure, Cholesterol
    - ECG readings and exercise-related data
    
    Output:
    - Risk prediction (High / Low)
    - Risk score (percentage)
    """)

st.write("---")
st.caption("Final Year Mini Project | Streamlit Deployment")