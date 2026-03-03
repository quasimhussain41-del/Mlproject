import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import os

# ─── Page Config ───
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")

# ─── Custom CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d2ff, #3a7bd5);
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
    background: linear-gradient(135deg, rgba(0,210,255,0.15), rgba(58,123,213,0.15));
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid rgba(0,210,255,0.2);
}

.result-positive {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 12px;
    text-align: center;
    font-size: 1.3rem;
    font-weight: 600;
    animation: fadeIn 0.5s ease-in;
}

.result-negative {
    background: linear-gradient(135deg, #11998e, #38ef7d);
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

div[data-testid="stNumberInput"] label {
    color: #e2e8f0 !important;
    font-weight: 500;
}

.stButton > button {
    background: linear-gradient(90deg, #00d2ff, #3a7bd5);
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
    box-shadow: 0 8px 25px rgba(0,210,255,0.3);
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_and_train():
    csv_path = os.path.join(os.path.dirname(__file__), "diabetes.csv")
    df = pd.read_csv(csv_path)
    X = df.drop(columns="Outcome", axis=1)
    Y = df["Outcome"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=0.2, stratify=Y, random_state=2
    )
    model = svm.SVC(kernel="linear")
    model.fit(X_train, Y_train)
    train_acc = accuracy_score(Y_train, model.predict(X_train))
    test_acc = accuracy_score(Y_test, model.predict(X_test))
    return model, scaler, train_acc, test_acc


model, scaler, train_acc, test_acc = load_and_train()

# ─── Header ───
st.markdown('<h1 class="hero-title">🩺 Diabetes Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">AI-powered diabetes risk assessment using Support Vector Machine</p>', unsafe_allow_html=True)

# ─── Accuracy Cards ───
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color:#a0aec0;font-size:0.85rem;">Training Accuracy</div>
        <div style="color:#00d2ff;font-size:2rem;font-weight:700;">{train_acc*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color:#a0aec0;font-size:0.85rem;">Test Accuracy</div>
        <div style="color:#38ef7d;font-size:2rem;font-weight:700;">{test_acc*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─── Input Form ───
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 📋 Enter Patient Details")

c1, c2, c3, c4 = st.columns(4)
with c1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
with c2:
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120)
with c3:
    bp = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=70)
with c4:
    skin = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

c5, c6, c7, c8 = st.columns(4)
with c5:
    insulin = st.number_input("Insulin (µU/mL)", min_value=0, max_value=900, value=80)
with c6:
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=32.0, step=0.1)
with c7:
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.47, step=0.01)
with c8:
    age = st.number_input("Age", min_value=1, max_value=120, value=33)

st.markdown('</div>', unsafe_allow_html=True)

# ─── Predict ───
if st.button("🔍  Predict Diabetes Risk"):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.markdown('<div class="result-positive">⚠️ High Risk — The patient is likely <b>Diabetic</b></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-negative">✅ Low Risk — The patient is likely <b>Not Diabetic</b></div>', unsafe_allow_html=True)
