import streamlit as st
import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# 🎬 Load Data
# --------------------------
@st.cache_data
def load_data():
    movies_data = pd.read_csv('movies.csv')
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')

    combined_features = (
        movies_data['genres'] + ' ' +
        movies_data['keywords'] + ' ' +
        movies_data['tagline'] + ' ' +
        movies_data['cast'] + ' ' +
        movies_data['director']
    )

    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)

    return movies_data, similarity

movies_data, similarity = load_data()

# --------------------------
# 🎥 Recommend Function
# --------------------------
def recommend_movies(movie_name):
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not find_close_match:
        st.warning("Sorry, no close match found. Try another movie name!")
        return []

    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match].index[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommended = []
    for movie in sorted_similar_movies[1:31]:  # top 30 recommendations
        index = movie[0]
        title_from_index = movies_data.iloc[index].title
        recommended.append(title_from_index)
    return recommended, close_match

# --------------------------
# 🎨 Streamlit UI
# --------------------------
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")

st.title("🎥 Movie Recommendation System")
st.write("Get personalized movie suggestions using **content-based filtering**!")

movie_name = st.text_input("Enter your favorite movie:")

if st.button("Recommend"):
    if movie_name.strip() == "":
        st.warning("Please enter a movie name!")
    else:
        recommended_movies, match = recommend_movies(movie_name)
        if recommended_movies:
            st.success(f"Because you liked **{match}**, you might enjoy:")
            for i, title in enumerate(recommended_movies, start=1):
                st.write(f"{i}. {title}")

# --------------------------
# 🧠 EDA Section (Optional)
# --------------------------
with st.expander("📊 Explore Dataset"):
    st.write("### Dataset Overview")
    st.dataframe(movies_data.head())

    st.write("### Missing Values")
    st.write(movies_data.isnull().sum())

    st.write("### Top Genres")
    st.bar_chart(movies_data['genres'].value_counts().head(10))
