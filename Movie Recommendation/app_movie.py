import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import os

# ─── Page Config ───
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# ─── Custom CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0d0d0d 0%, #1a1a2e 50%, #16213e 100%);
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #e50914, #ff6b6b);
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

.movie-card {
    background: linear-gradient(135deg, rgba(229,9,20,0.1), rgba(255,107,107,0.05));
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-bottom: 0.75rem;
    border-left: 4px solid #e50914;
    transition: all 0.3s ease;
}

.movie-card:hover {
    background: linear-gradient(135deg, rgba(229,9,20,0.2), rgba(255,107,107,0.1));
    transform: translateX(5px);
}

.movie-rank {
    color: #e50914;
    font-weight: 700;
    font-size: 1.3rem;
    margin-right: 1rem;
}

.movie-name {
    color: #e2e8f0;
    font-size: 1.1rem;
    font-weight: 500;
}

.info-card {
    background: linear-gradient(135deg, rgba(229,9,20,0.15), rgba(255,107,107,0.15));
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid rgba(229,9,20,0.2);
}

div[data-testid="stTextInput"] label {
    color: #e2e8f0 !important;
    font-weight: 500;
}

.stButton > button {
    background: linear-gradient(90deg, #e50914, #ff6b6b);
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
    box-shadow: 0 8px 25px rgba(229,9,20,0.3);
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.results-container {
    animation: fadeIn 0.5s ease-in;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_and_build():
    csv_path = os.path.join(os.path.dirname(__file__), "movies.csv")
    df = pd.read_csv(csv_path)
    features = ["genres", "keywords", "tagline", "cast", "director"]
    for f in features:
        df[f] = df[f].fillna("")
    df["combined"] = df["genres"] + " " + df["keywords"] + " " + df["tagline"] + " " + df["cast"] + " " + df["director"]
    vectorizer = TfidfVectorizer()
    feat_vectors = vectorizer.fit_transform(df["combined"])
    similarity = cosine_similarity(feat_vectors)
    return df, similarity


df, similarity = load_and_build()

# ─── Header ───
st.markdown('<h1 class="hero-title">🎬 Movie Recommender</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Content-based recommendation engine using TF-IDF & Cosine Similarity</p>', unsafe_allow_html=True)

# ─── Stats ───
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="info-card">
        <div style="color:#a0aec0;font-size:0.85rem;">Total Movies</div>
        <div style="color:#e50914;font-size:2rem;font-weight:700;">{len(df):,}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    unique_genres = set()
    for g in df["genres"].dropna():
        for genre in g.split():
            unique_genres.add(genre)
    st.markdown(f"""
    <div class="info-card">
        <div style="color:#a0aec0;font-size:0.85rem;">Unique Genres</div>
        <div style="color:#ff6b6b;font-size:2rem;font-weight:700;">{len(unique_genres)}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="info-card">
        <div style="color:#a0aec0;font-size:0.85rem;">Algorithm</div>
        <div style="color:#e50914;font-size:1.2rem;font-weight:700;">Cosine Similarity</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─── Input ───
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 🎥 Enter a Movie Title")
movie_name = st.text_input("Movie Name", placeholder="e.g. Iron Man, Avatar, The Dark Knight...")
st.markdown('</div>', unsafe_allow_html=True)

# ─── Recommend ───
if st.button("🔍  Get Recommendations"):
    if movie_name.strip():
        all_titles = df["title"].tolist()
        close_matches = difflib.get_close_matches(movie_name, all_titles, n=1, cutoff=0.4)

        if close_matches:
            match = close_matches[0]
            idx = df[df.title == match].index[0]
            similarity_scores = list(enumerate(similarity[idx]))
            sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:11]

            st.markdown(f"""
            <div style="text-align:center;margin:1rem 0;">
                <span style="color:#a0aec0;">Showing results for: </span>
                <span style="color:#e50914;font-weight:700;font-size:1.2rem;">{match}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            for i, (movie_idx, score) in enumerate(sorted_movies, 1):
                title = df.iloc[movie_idx]["title"]
                st.markdown(f"""
                <div class="movie-card">
                    <span class="movie-rank">#{i}</span>
                    <span class="movie-name">{title}</span>
                    <span style="color:#666;float:right;font-size:0.85rem;">Score: {score:.3f}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ No close match found. Try a different movie title.")
    else:
        st.warning("⚠️ Please enter a movie name.")
