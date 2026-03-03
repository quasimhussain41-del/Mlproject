import streamlit as st
import subprocess
import sys
import os

# ─── Page Config ───
st.set_page_config(page_title="ML Models Hub", page_icon="🧠", layout="wide")

# ─── Custom CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 40%, #16213e 70%, #0f3460 100%);
}

.hub-title {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d2ff, #a855f7, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}

.hub-subtitle {
    color: #8892b0;
    text-align: center;
    font-size: 1.15rem;
    margin-bottom: 3rem;
    font-weight: 300;
}

.model-card {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    padding: 2rem;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
    min-height: 280px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    position: relative;
    overflow: hidden;
}

.model-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    border-radius: 20px 20px 0 0;
    opacity: 0;
    transition: opacity 0.4s ease;
}

.model-card:hover {
    transform: translateY(-8px);
    border-color: rgba(255,255,255,0.15);
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
}

.model-card:hover::before {
    opacity: 1;
}

.card-icon {
    font-size: 3.5rem;
    margin-bottom: 1rem;
}

.card-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.5rem;
}

.card-desc {
    color: #8892b0;
    font-size: 0.9rem;
    line-height: 1.5;
    margin-bottom: 1rem;
}

.card-tag {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}

/* Card-specific gradients */
.card-diabetes::before { background: linear-gradient(90deg, #00d2ff, #3a7bd5); }
.card-diabetes:hover { border-color: rgba(0,210,255,0.3); }
.card-diabetes .card-tag { background: rgba(0,210,255,0.15); color: #00d2ff; }

.card-heart::before { background: linear-gradient(90deg, #e74c3c, #ff6b6b); }
.card-heart:hover { border-color: rgba(231,76,60,0.3); }
.card-heart .card-tag { background: rgba(231,76,60,0.15); color: #ff6b6b; }

.card-loan::before { background: linear-gradient(90deg, #f7971e, #ffd200); }
.card-loan:hover { border-color: rgba(247,151,30,0.3); }
.card-loan .card-tag { background: rgba(247,151,30,0.15); color: #ffd200; }

.card-spam::before { background: linear-gradient(90deg, #a855f7, #6366f1); }
.card-spam:hover { border-color: rgba(168,85,247,0.3); }
.card-spam .card-tag { background: rgba(168,85,247,0.15); color: #a855f7; }

.card-movie::before { background: linear-gradient(90deg, #e50914, #ff6b6b); }
.card-movie:hover { border-color: rgba(229,9,20,0.3); }
.card-movie .card-tag { background: rgba(229,9,20,0.15); color: #e50914; }

.card-churn::before { background: linear-gradient(90deg, #00b4d8, #48cae4); }
.card-churn:hover { border-color: rgba(0,180,216,0.3); }
.card-churn .card-tag { background: rgba(0,180,216,0.15); color: #48cae4; }

.stButton > button {
    border-radius: 12px;
    font-weight: 600;
    padding: 0.6rem 2rem;
    transition: all 0.3s ease;
    width: 100%;
    margin-top: 0.5rem;
}

.footer {
    text-align: center;
    color: #4a5568;
    font-size: 0.85rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.05);
}
</style>
""", unsafe_allow_html=True)

# ─── Header ───
st.markdown('<h1 class="hub-title">🧠 ML Models Hub</h1>', unsafe_allow_html=True)
st.markdown('<p class="hub-subtitle">Select a machine learning model to explore and make predictions</p>', unsafe_allow_html=True)

# ─── Define Models ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

models = [
    {
        "icon": "🩺",
        "title": "Diabetes Prediction",
        "desc": "Predict diabetes risk using patient health metrics with SVM",
        "tag": "SVM",
        "class": "card-diabetes",
        "app": os.path.join(BASE_DIR, "Diabetes prediction", "app_diabetes.py"),
    },
    {
        "icon": "❤️",
        "title": "Heart Disease",
        "desc": "Cardiac risk assessment using clinical data with Logistic Regression",
        "tag": "Logistic Regression",
        "class": "card-heart",
        "app": os.path.join(BASE_DIR, "Heart Disease Prediction", "app_heart.py"),
    },
    {
        "icon": "🏦",
        "title": "Loan Prediction",
        "desc": "Loan eligibility assessment based on applicant profile using SVM",
        "tag": "SVM",
        "class": "card-loan",
        "app": os.path.join(BASE_DIR, "Loan Prediction", "app_loan.py"),
    },
    {
        "icon": "📧",
        "title": "Spam Mail Detector",
        "desc": "Email classification using TF-IDF & Logistic Regression",
        "tag": "TF-IDF + LR",
        "class": "card-spam",
        "app": os.path.join(BASE_DIR, "spam mail prediciton", "app_spam.py"),
    },
    {
        "icon": "🎬",
        "title": "Movie Recommender",
        "desc": "Content-based movie recommendations using cosine similarity",
        "tag": "Cosine Similarity",
        "class": "card-movie",
        "app": os.path.join(BASE_DIR, "Movie Recommendation", "app_movie.py"),
    },
    {
        "icon": "📊",
        "title": "Customer Churn",
        "desc": "Customer retention prediction using Random Forest classifier",
        "tag": "Random Forest",
        "class": "card-churn",
        "app": os.path.join(BASE_DIR, "Customer Churn Prediction", "app_churn.py"),
    },
]

# ─── Model Selection ───
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# ─── Display Cards ───
row1 = st.columns(3)
row2 = st.columns(3)
all_cols = row1 + row2

for i, (col, m) in enumerate(zip(all_cols, models)):
    with col:
        st.markdown(f"""
        <div class="model-card {m['class']}">
            <div class="card-icon">{m['icon']}</div>
            <div class="card-title">{m['title']}</div>
            <div class="card-desc">{m['desc']}</div>
            <span class="card-tag">{m['tag']}</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"Launch {m['title']}", key=f"btn_{i}"):
            st.session_state.selected_model = m

# ─── Launch Selected App ───
if st.session_state.selected_model:
    selected = st.session_state.selected_model
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center;padding:1.5rem;">
        <div style="color:#8892b0;font-size:1rem;margin-bottom:0.5rem;">Launching...</div>
        <div style="color:#e2e8f0;font-size:1.5rem;font-weight:700;">{selected['icon']} {selected['title']}</div>
        <div style="color:#4a5568;font-size:0.9rem;margin-top:0.5rem;">
            Run in terminal: <code style="color:#00d2ff;">streamlit run "{selected['app']}"</code>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.info(f"💡 To run this model, open a terminal and execute:\n\n`streamlit run \"{selected['app']}\"`")

# ─── Footer ───
st.markdown("""
<div class="footer">
    Built with Streamlit • 6 ML Models • Powered by scikit-learn
</div>
""", unsafe_allow_html=True)
