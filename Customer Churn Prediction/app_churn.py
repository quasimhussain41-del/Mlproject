import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

# ─── Page Config ───
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

# ─── Custom CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a192f 0%, #112240 50%, #1d3557 100%);
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00b4d8, #48cae4);
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
    background: linear-gradient(135deg, rgba(0,180,216,0.15), rgba(72,202,228,0.15));
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid rgba(0,180,216,0.2);
}

.result-churn {
    background: linear-gradient(135deg, #e74c3c, #c0392b);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 12px;
    text-align: center;
    font-size: 1.3rem;
    font-weight: 600;
    animation: fadeIn 0.5s ease-in;
}

.result-stay {
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
    background: linear-gradient(90deg, #00b4d8, #48cae4);
    color: #0a192f;
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
    box-shadow: 0 8px 25px rgba(0,180,216,0.3);
}

.section-header {
    color: #48cae4;
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_and_train():
    csv_path = os.path.join(os.path.dirname(__file__), "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = pd.read_csv(csv_path)
    df = df.drop("customerID", axis=1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Label encode all object columns
    le_dict = {}
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    X = df.drop("Churn", axis=1)
    Y = df["Churn"]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)
    train_acc = accuracy_score(Y_train, model.predict(X_train))
    test_acc = accuracy_score(Y_test, model.predict(X_test))

    return model, le_dict, X.columns.tolist(), train_acc, test_acc


model, le_dict, feature_cols, train_acc, test_acc = load_and_train()

# ─── Header ───
st.markdown('<h1 class="hero-title">📊 Customer Churn Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">AI-powered customer retention analysis using Random Forest</p>', unsafe_allow_html=True)

# ─── Accuracy ───
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color:#a0aec0;font-size:0.85rem;">Training Accuracy</div>
        <div style="color:#00b4d8;font-size:2rem;font-weight:700;">{train_acc*100:.1f}%</div>
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
st.markdown("### 📋 Enter Customer Details")

# ─── Demographics ───
st.markdown('<div class="section-header">👤 Demographics</div>', unsafe_allow_html=True)
d1, d2, d3, d4 = st.columns(4)
with d1:
    gender = st.selectbox("Gender", options=["Female", "Male"])
with d2:
    senior = st.selectbox("Senior Citizen", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with d3:
    partner = st.selectbox("Partner", options=["Yes", "No"])
with d4:
    dependents = st.selectbox("Dependents", options=["Yes", "No"])

# ─── Services ───
st.markdown('<div class="section-header">📱 Services</div>', unsafe_allow_html=True)
s1, s2, s3 = st.columns(3)
with s1:
    phone_service = st.selectbox("Phone Service", options=["Yes", "No"])
with s2:
    multiple_lines = st.selectbox("Multiple Lines", options=["No", "Yes", "No phone service"])
with s3:
    internet_service = st.selectbox("Internet Service", options=["DSL", "Fiber optic", "No"])

s4, s5, s6, s7 = st.columns(4)
with s4:
    online_security = st.selectbox("Online Security", options=["No", "Yes", "No internet service"])
with s5:
    online_backup = st.selectbox("Online Backup", options=["No", "Yes", "No internet service"])
with s6:
    device_protection = st.selectbox("Device Protection", options=["No", "Yes", "No internet service"])
with s7:
    tech_support = st.selectbox("Tech Support", options=["No", "Yes", "No internet service"])

s8, s9 = st.columns(2)
with s8:
    streaming_tv = st.selectbox("Streaming TV", options=["No", "Yes", "No internet service"])
with s9:
    streaming_movies = st.selectbox("Streaming Movies", options=["No", "Yes", "No internet service"])

# ─── Account ───
st.markdown('<div class="section-header">💳 Account Info</div>', unsafe_allow_html=True)
a1, a2, a3 = st.columns(3)
with a1:
    contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"])
with a2:
    paperless = st.selectbox("Paperless Billing", options=["Yes", "No"])
with a3:
    payment = st.selectbox("Payment Method", options=[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

a4, a5, a6 = st.columns(3)
with a4:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
with a5:
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
with a6:
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0, step=1.0)

st.markdown('</div>', unsafe_allow_html=True)

# ─── Predict ───
if st.button("🔍  Predict Customer Churn"):
    # Build input matching training feature order
    input_dict = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    # Encode categoricals using same LabelEncoders
    for col, val in input_dict.items():
        if col in le_dict:
            try:
                input_dict[col] = le_dict[col].transform([val])[0]
            except ValueError:
                input_dict[col] = 0  # fallback for unseen labels

    input_df = pd.DataFrame([input_dict])
    # Ensure column order matches training
    input_df = input_df[feature_cols]

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.markdown('<div class="result-churn">⚠️ High Risk — This customer is likely to <b>Churn</b></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-stay">✅ Low Risk — This customer is likely to <b>Stay</b></div>', unsafe_allow_html=True)
