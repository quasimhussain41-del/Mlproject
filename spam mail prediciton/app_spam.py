import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# ─── Page Config ───
st.set_page_config(page_title="Spam Mail Prediction", page_icon="📧", layout="wide")

# ─── Custom CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #1e1440 50%, #2d1b69 100%);
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a855f7, #6366f1);
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
    background: linear-gradient(135deg, rgba(168,85,247,0.15), rgba(99,102,241,0.15));
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid rgba(168,85,247,0.2);
}

.result-spam {
    background: linear-gradient(135deg, #e74c3c, #c0392b);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 12px;
    text-align: center;
    font-size: 1.3rem;
    font-weight: 600;
    animation: fadeIn 0.5s ease-in;
}

.result-ham {
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

div[data-testid="stTextArea"] label {
    color: #e2e8f0 !important;
    font-weight: 500;
}

.stButton > button {
    background: linear-gradient(90deg, #a855f7, #6366f1);
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
    box-shadow: 0 8px 25px rgba(168,85,247,0.3);
}

textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #e2e8f0 !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
    min-height: 150px !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_and_train():
    csv_path = os.path.join(os.path.dirname(__file__), "mail_data.csv")
    df = pd.read_csv(csv_path)
    data = df.where(pd.notnull(df), "")
    data.loc[data["Category"] == "spam", "Category"] = 0
    data.loc[data["Category"] == "ham", "Category"] = 1
    X = data["Message"]
    Y = data["Category"].astype(int)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=3
    )
    vectorizer = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
    X_train_feat = vectorizer.fit_transform(X_train)
    X_test_feat = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_feat, Y_train)
    train_acc = accuracy_score(Y_train, model.predict(X_train_feat))
    test_acc = accuracy_score(Y_test, model.predict(X_test_feat))
    return model, vectorizer, train_acc, test_acc


model, vectorizer, train_acc, test_acc = load_and_train()

# ─── Header ───
st.markdown('<h1 class="hero-title">📧 Spam Mail Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">AI-powered email classification using Logistic Regression & TF-IDF</p>', unsafe_allow_html=True)

# ─── Accuracy ───
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color:#a0aec0;font-size:0.85rem;">Training Accuracy</div>
        <div style="color:#a855f7;font-size:2rem;font-weight:700;">{train_acc*100:.1f}%</div>
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
st.markdown("### ✉️ Paste Email Content Below")
email_text = st.text_area("Email Message", height=200,
                           placeholder="Paste the email message here to check if it's spam or not...")
st.markdown('</div>', unsafe_allow_html=True)

# ─── Predict ───
if st.button("🔍  Analyze Email"):
    if email_text.strip():
        input_feat = vectorizer.transform([email_text])
        prediction = model.predict(input_feat)

        if prediction[0] == 0:
            st.markdown('<div class="result-spam">🚫 This email is classified as <b>SPAM</b></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-ham">✅ This email is classified as <b>Legitimate (Ham)</b></div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter an email message to analyze.")
