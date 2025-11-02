
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎧",
    layout="centered",
)


@st.cache_data
def load_data(sample_size=10000):
    df = pd.read_csv("spotify_millsongdata.csv")
    df = df.dropna(subset=['text']).reset_index(drop=True)

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
    return df

df = load_data(sample_size=10000)


@st.cache_resource
def build_model():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['text'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf, cosine_sim

tfidf, cosine_sim = build_model()


def recommend(song_title, top_n=5):
    idx = df[df['song'].str.lower() == song_title.lower()].index
    if len(idx) == 0:
        return []
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    song_indices = [i[0] for i in sim_scores]
    return df.iloc[song_indices][['song', 'artist']]


st.title("🎶 Music Recommendation System")
st.write("Welcome to your personalized music recommender! \
          This app uses song lyrics to find similar tracks 🎧")

song_input = st.text_input("Enter a song name (from the sampled dataset):")

top_n = st.slider("Number of recommendations", 1, 10, 5)

if st.button("Recommend"):
    if not song_input.strip():
        st.warning("⚠️ Please enter a song name.")
    else:
        recs = recommend(song_input, top_n)
        if len(recs) == 0:
            st.error(f"❌ '{song_input}' not found in the sampled dataset.")
        else:
            st.success(f"🎧 Top {top_n} recommendations for '{song_input}':")
            for _, row in recs.iterrows():
                st.write(f"- **{row['song']}** — *{row['artist']}*")
