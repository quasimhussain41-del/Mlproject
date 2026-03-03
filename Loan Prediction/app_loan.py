import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import os

# ─── Page Config ───
st.set_page_config(page_title="Loan Prediction", page_icon="🏦", layout="wide")

# ─── Custom CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0c0c1d 0%, #1b2838 50%, #0d1b2a 100%);
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #f7971e, #ffd200);
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
    background: linear-gradient(135deg, rgba(247,151,30,0.15), rgba(255,210,0,0.15));
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid rgba(247,151,30,0.2);
}

.result-approved {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 12px;
    text-align: center;
    font-size: 1.3rem;
    font-weight: 600;
    animation: fadeIn 0.5s ease-in;
}

.result-rejected {
    background: linear-gradient(135deg, #e74c3c, #c0392b);
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
    background: linear-gradient(90deg, #f7971e, #ffd200);
    color: #1a1a2e;
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
    box-shadow: 0 8px 25px rgba(247,151,30,0.3);
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_and_train():
    csv_path = os.path.join(os.path.dirname(__file__), "loan.csv")
    df = pd.read_csv(csv_path)
    df = df.dropna()
    df = df.replace({"Loan_Status": {"N": 0, "Y": 1}})
    df = df.replace(to_replace="3+", value=4)
    df.replace({"Married": {"No": 0, "Yes": 1},
                "Gender": {"Male": 1, "Female": 0},
                "Self_Employed": {"No": 0, "Yes": 1},
                "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
                "Education": {"Graduate": 1, "Not Graduate": 0}}, inplace=True)
    df["Dependents"] = df["Dependents"].astype(int)
    X = df.drop(columns=["Loan_ID", "Loan_Status"], axis=1)
    Y = df["Loan_Status"]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2
    )
    model = svm.SVC(kernel="linear")
    model.fit(X_train, Y_train)
    train_acc = accuracy_score(Y_train, model.predict(X_train))
    test_acc = accuracy_score(Y_test, model.predict(X_test))
    return model, train_acc, test_acc


model, train_acc, test_acc = load_and_train()

# ─── Header ───
st.markdown('<h1 class="hero-title">🏦 Loan Status Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">AI-powered loan eligibility assessment using Support Vector Machine</p>', unsafe_allow_html=True)

# ─── Accuracy ───
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color:#a0aec0;font-size:0.85rem;">Training Accuracy</div>
        <div style="color:#f7971e;font-size:2rem;font-weight:700;">{train_acc*100:.1f}%</div>
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

# ─── Input ───
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 📋 Enter Loan Application Details")

c1, c2, c3 = st.columns(3)
with c1:
    gender = st.selectbox("Gender", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
with c2:
    married = st.selectbox("Married", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
with c3:
    dependents = st.selectbox("Dependents", options=[0, 1, 2, 4])

c4, c5, c6 = st.columns(3)
with c4:
    education = st.selectbox("Education", options=[1, 0], format_func=lambda x: "Graduate" if x == 1 else "Not Graduate")
with c5:
    self_employed = st.selectbox("Self Employed", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with c6:
    applicant_income = st.number_input("Applicant Income", min_value=0, max_value=100000, value=5000)

c7, c8, c9 = st.columns(3)
with c7:
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, max_value=50000.0, value=1500.0)
with c8:
    loan_amount = st.number_input("Loan Amount (thousands)", min_value=0.0, max_value=1000.0, value=128.0)
with c9:
    loan_term = st.number_input("Loan Term (days)", min_value=12.0, max_value=500.0, value=360.0)

c10, c11 = st.columns(2)
with c10:
    credit_history = st.selectbox("Credit History", options=[1.0, 0.0], format_func=lambda x: "Good (1)" if x == 1.0 else "Bad (0)")
with c11:
    property_area = st.selectbox("Property Area", options=[0, 1, 2], format_func=lambda x: ["Rural", "Semiurban", "Urban"][x])

st.markdown('</div>', unsafe_allow_html=True)

# ─── Predict ───
if st.button("🔍  Predict Loan Eligibility"):
    input_data = np.array([[gender, married, dependents, education, self_employed,
                            applicant_income, coapplicant_income, loan_amount,
                            loan_term, credit_history, property_area]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.markdown('<div class="result-approved">✅ Congratulations! Your loan is likely to be <b>Approved</b></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-rejected">❌ Sorry, your loan is likely to be <b>Rejected</b></div>', unsafe_allow_html=True)
