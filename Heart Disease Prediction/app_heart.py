import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# ─── Page Config ───
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

# ─── Custom CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #e74c3c, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    color: #a0aec0;
    text-align: center;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

.glass-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.1);
    padding: 2rem;
    margin-bottom: 1.5rem;
}

.metric-card {
    background: linear-gradient(135deg, rgba(231,76,60,0.15), rgba(255,107,107,0.15));
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid rgba(231,76,60,0.2);
}

.result-disease {
    background: linear-gradient(135deg, #e74c3c, #c0392b);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 12px;
    text-align: center;
    font-size: 1.3rem;
    font-weight: 600;
    animation: fadeIn 0.5s ease-in;
}

.result-healthy {
    background: linear-gradient(135deg, #27ae60, #2ecc71);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 12px;
    text-align: center;
    font-size: 1.3rem;
    font-weight: 600;
    animation: fadeIn 0.5s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

div[data-testid="stNumberInput"] label, div[data-testid="stSelectbox"] label {
    color: #e2e8f0 !important;
    font-weight: 500;
}

.stButton > button {
    background: linear-gradient(90deg, #e74c3c, #ff6b6b);
    color: white;
    border: none;
    padding: 0.75rem 3rem;
    font-size: 1.1rem;
    font-weight: 600;
    border-radius: 12px;
    width: 100%;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(231,76,60,0.3);
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_and_train():
    csv_path = os.path.join(os.path.dirname(__file__), "heart_disease_data.csv")
    df = pd.read_csv(csv_path)
    X = df.drop(columns="target", axis=1)
    Y = df["target"]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    train_acc = accuracy_score(Y_train, model.predict(X_train))
    test_acc = accuracy_score(Y_test, model.predict(X_test))
    return model, train_acc, test_acc


model, train_acc, test_acc = load_and_train()

# ─── Header ───
st.markdown('<h1 class="hero-title">❤️ Heart Disease Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">AI-powered cardiac risk assessment using Logistic Regression</p>', unsafe_allow_html=True)

# ─── Accuracy Cards ───
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color:#a0aec0;font-size:0.85rem;">Training Accuracy</div>
        <div style="color:#e74c3c;font-size:2rem;font-weight:700;">{train_acc*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color:#a0aec0;font-size:0.85rem;">Test Accuracy</div>
        <div style="color:#2ecc71;font-size:2rem;font-weight:700;">{test_acc*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─── Input Form ───
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 📋 Enter Patient Details")

c1, c2, c3, c4 = st.columns(4)
with c1:
    age = st.number_input("Age", min_value=1, max_value=120, value=55)
with c2:
    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
with c3:
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                       format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][x])
with c4:
    trestbps = st.number_input("Resting BP (mmHg)", min_value=50, max_value=250, value=130)

c5, c6, c7, c8 = st.columns(4)
with c5:
    chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=240)
with c6:
    fbs = st.selectbox("Fasting Blood Sugar > 120", options=[0, 1],
                        format_func=lambda x: "Yes" if x == 1 else "No")
with c7:
    restecg = st.selectbox("Resting ECG", options=[0, 1, 2],
                            format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])
with c8:
    thalach = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)

c9, c10, c11, c12 = st.columns(4)
with c9:
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1],
                          format_func=lambda x: "Yes" if x == 1 else "No")
with c10:
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
with c11:
    slope = st.selectbox("Slope of Peak ST", options=[0, 1, 2],
                          format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
with c12:
    ca = st.selectbox("Number of Major Vessels (ca)", options=[0, 1, 2, 3, 4])

thal_val = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3],
                         format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x])

st.markdown('</div>', unsafe_allow_html=True)

# ─── Predict ───
if st.button("🔍  Predict Heart Disease Risk"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal_val]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.markdown('<div class="result-disease">⚠️ Risk Detected — The patient may have <b>Heart Disease</b></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-healthy">✅ Low Risk — The patient likely has a <b>Healthy Heart</b></div>', unsafe_allow_html=True)
