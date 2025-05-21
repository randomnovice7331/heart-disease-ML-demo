import os
import joblib
import pandas as pd
import streamlit as st

# 1) Učitaj pipeline
save_dir      = os.getcwd()  # pretpostavka: radiš iz korijena repozitorija
pipeline_path = os.path.join(save_dir, "full_pipeline.joblib")
pipeline      = joblib.load(pipeline_path)

st.title("Demo: Heart disease prediction")

# 2) Unos – kao primjer, mala forma za unos
age      = st.number_input(
    "Age of the patient (age)",
    min_value=1, max_value=140,
    value=50
)

sex      = st.selectbox(
    "Gender of the patient (sex) Male = 0; Female = 1",
    options=[0, 1],
    index=0
)

cp       = st.selectbox(
    "Type of chest pain. Typical Angina = 0; Atypical angina = 1; Non-anginal pain = 2; Asymptomatic = 3",
    options=[0, 1, 2, 3],
    index=1
)

trestbps = st.number_input(
    "Resting systolic blood pressure [mm Hg]",
    min_value=60, max_value=230,
    value=135
)

chol     = st.number_input(
    "Serum cholesterol [mg/dl]",
    min_value=90, max_value=650,
    value=245
)

fbs      = st.selectbox(
    "Fasting blood sugar is > 120 mg/dl? (true=1, false=0)",
    options=[0, 1],
    index=1
)

restecg  = st.selectbox(
    "Resting electrocardiographic results [Normal=0, ST elevation or depression of > 0.05 mV = 1, Left ventricular hypertrophy = 2]",
    options=[0, 1, 2],
    index=2
)

thalach  = st.number_input(
    "Maximum heart rate achieved",
    min_value=50, max_value=250,
    value=150
)

exang    = st.selectbox(
    "Exercise-induced angina [yes=1, no=0]",
    options=[0, 1],
    index=1
)

oldpeak  = st.number_input(
    "ST depression induced by exercise relative to rest",
    min_value=0.0, max_value=9.0,
    value=1.1
)

slope    = st.selectbox(
    "The slope of the peak exercise ST segment [up-slope=0, flat=1, down-slope=2]",
    options=[0, 1, 2],
    index=2
)

ca       = st.selectbox(
    "Number of major blood vessels visible by fluoroscopy",
    options=[0, 1, 2, 3, 4],
    index=2
)

thal     = st.selectbox(
    "Thalassemia [no thalassemia = 1, fixed defect = 2, reversible defect = 3]",
    options=[1, 2, 3],
    index=2
)                 
                        
# … ponovi za sve ostale feature  
# Napomena: možeš i učitati cijeli CSV, ali za početak radimo ručni unos

# 3) Kreiraj DataFrame od unosa sa svim prediktorima
row = pd.DataFrame({
    "age":       [age],
    "sex":       [sex],
    "cp":        [cp],
    "trestbps":  [trestbps],
    "chol":      [chol],
    "fbs":       [fbs],
    "restecg":   [restecg],
    "thalach":   [thalach],
    "exang":     [exang],
    "oldpeak":   [oldpeak],
    "slope":     [slope],
    "ca":        [ca],
    "thal":      [thal]
})


# 4) Pokreni predikciju
label = pipeline.predict(row)[0]
proba = pipeline.predict_proba(row)[0,1]

# probas = pipeline.predict_proba(row)[0]   # npr. array([0.062, 0.938])
# print(f"P(0) = {probas[0]:.3f}, P(1) = {probas[1]:.3f}")

st.write(f"**Predviđena klasa:** {label}")
st.write(f"**Vjerojatnost bolesti:** {proba:.3f}")
