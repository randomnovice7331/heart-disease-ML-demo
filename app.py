import os
import joblib
import pandas as pd
import streamlit as st

# ————————————————
# 0) Definicija custom transformera (isti kod kao u treniranju)
from copulas.multivariate import GaussianMultivariate
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

class GaussianCopulaImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = GaussianMultivariate()
    def fit(self, X, y=None):
        self.model.fit(X.dropna())
        self.is_fitted_ = True
        return self
    def transform(self, X):
        df = X.copy()
        mask = df.isna()
        samples = pd.DataFrame(self.model.sample(len(df)),
                               columns=df.columns, index=df.index)
        for col in df.columns:
            df.loc[mask[col], col] = samples.loc[mask[col], col]
        return df

class OldpeakFeatureCreator(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        vals = X['oldpeak'].values
        def kde_max(v):
            kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(v[:,None])
            grid = np.linspace(v.min(), v.max(), 1000)[:,None]
            dens = np.exp(kde.score_samples(grid))
            peaks = (np.diff(np.sign(np.diff(dens)))<0).nonzero()[0] + 1
            return grid[peaks[np.argmax(dens[peaks])]][0] if peaks.size else v.mean()
        self.h_max = kde_max(vals[y==0])
        self.s_max = kde_max(vals[y==1])
        self.is_fitted_ = True
        return self
    def transform(self, X):
        df = X.copy()
        df['oldpeak_healthy_max_dist'] = np.abs(df['oldpeak'] - self.h_max)
        df['oldpeak_sick_max_dist']    = np.abs(df['oldpeak'] - self.s_max)
        return df.drop(columns='oldpeak')

class ColumnScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.scaler = StandardScaler()
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns])
        self.is_fitted_ = True
        return self
    def transform(self, X):
        Xc = X.copy()
        Xc[self.columns] = self.scaler.transform(Xc[self.columns])
        return Xc

# 1) Učitaj pipeline
save_dir      = os.getcwd()  # pretpostavka: radiš iz korijena repozitorija
pipeline_path = os.path.join(save_dir, "full_pipeline.joblib")
pipeline      = joblib.load(pipeline_path)

st.title("Demo: Heart disease prediction")

# 2) Unos – kao primjer, mala forma za unos
# - Age i Sex u istom redu
col1, col2 = st.columns(2)
with col1:
    age = st.number_input(
        "Age of the patient (age)",
        min_value=1, max_value=140,
        value=50
    )
with col2:
    sex = st.selectbox(
        "Gender of the patient (sex) Male = 0; Female = 1",
        options=[0, 1],
        index=0
    )

# - Type of chest pain sam u svom redu
cp = st.selectbox(
    "Type of chest pain. Typical Angina = 0; Atypical angina = 1; Non-anginal pain = 2; Asymptomatic = 3",
    options=[0, 1, 2, 3],
    index=1
)

# - Resting systolic blood pressure i Serum cholesterol u istom redu
col3, col4 = st.columns(2)
with col3:
    trestbps = st.number_input(
        "Resting systolic blood pressure [mm Hg]",
        min_value=60, max_value=230,
        value=135
    )
with col4:
    chol = st.number_input(
        "Serum cholesterol [mg/dl]",
        min_value=90, max_value=650,
        value=245
    )

# - Fasting blood sugar sam u svom redu
fbs = st.selectbox(
    "Fasting blood sugar is > 120 mg/dl? (true=1, false=0)",
    options=[0, 1],
    index=1
)

# - Resting ECG sam u svom redu
restecg = st.selectbox(
    "Resting ECG results [Normal=0, ST elevation or depression of > 0.05 mV = 1, Left ventricular hypertrophy = 2]",
    options=[0, 1, 2],
    index=2
)

# - Max heart rate achieved i Exercise-induced angina u istom redu
col5, col6 = st.columns(2)
with col5:
    thalach = st.number_input(
        "Maximum heart rate achieved",
        min_value=50, max_value=250,
        value=150
    )
with col6:
    exang = st.selectbox(
        "Exercise-induced angina [yes=1, no=0]",
        options=[0, 1],
        index=1
    )

# Ostali unosi ostaju kako jesu
oldpeak = st.number_input(
    "ST depression induced by exercise relative to rest",
    min_value=0.0, max_value=9.0,
    value=1.1
)

slope = st.selectbox(
    "The slope of the peak exercise ST segment [up-slope=0, flat=1, down-slope=2]",
    options=[0, 1, 2],
    index=2
)

ca = st.selectbox(
    "Number of major blood vessels visible by fluoroscopy",
    options=[0, 1, 2, 3, 4],
    index=2
)

thal = st.selectbox(
    "Thalassemia [no thalassemia = 1, fixed defect = 2, reversible defect = 3]",
    options=[1, 2, 3],
    index=2
)

# 3) Kreiraj DataFrame od unosa sa svim prediktorima
row = pd.DataFrame({
    "age":      [age],
    "sex":      [sex],
    "cp":       [cp],
    "trestbps": [trestbps],
    "chol":     [chol],
    "fbs":      [fbs],
    "restecg":  [restecg],
    "thalach":  [thalach],
    "exang":    [exang],
    "oldpeak":  [oldpeak],
    "slope":    [slope],
    "ca":       [ca],
    "thal":     [thal]
})

# 4) Pokreni predikciju
label = pipeline.predict(row)[0]
proba = pipeline.predict_proba(row)[0,1]

st.write(f"**Predicted class [healthy=0, sick=1]:**  {label}")
st.write(f"**Probability of disease:** {proba:.3f}")
