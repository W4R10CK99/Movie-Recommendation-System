import streamlit as st

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
#cosine similarity function is a efficient way to calculate similarity of 2 data
from sklearn.metrics.pairwise import cosine_similarity
#diiflib is used to indentify given input with closest data
import difflib

st.markdown(
    "<style>"
    ".stApp h1 {"
    "font-family: 'Arial', sans-serif;"
    "font-weight: bold;"
    "color: black;"  # Text color for the title
    "background-color: #FFD700;"  # IMDb-like yellow background color
    "border-radius: 10px;"  # Rounded corners
    "padding: 10px 20px;"  # Add some padding for spacing
    "}"
    "</style>",
    unsafe_allow_html=True,
)

# Apply custom CSS styles to the subtitle

# Your existing code for movie recommendation
st.title("Movie Recommender")
st.write("")
st.write("")
st.write("")
st.write("")

movies_data = pd.read_csv('movies.csv')
relevant_features = ['genres','keywords','cast','director','tagline']

for feature in relevant_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

similarity = cosine_similarity(feature_vectors)

csv_file = "titles.csv"  # Change this to your CSV file path
titles_df = pd.read_csv(csv_file)

# Extract the list of titles from the DataFrame
titles = titles_df['Titles'].tolist()

# Create a dropdown widget to select a movie title
movie_name = st.selectbox("Movie You watched :", titles)


if movie_name != "NONE":
    list_of_all_titles = movies_data['title'].tolist()

    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    close_match = find_close_match[0]

    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

    similarity_score = list(enumerate(similarity[index_of_the_movie]))

    sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)

    st.markdown("<style>@keyframes dust { 0% { transform: translate(0, -10px); opacity: 0; } 100% { transform: translate(0, 0); opacity: 1; } } .dust-in { animation: dust 2.0s ease-in; }</style>", unsafe_allow_html=True)
    st.markdown("<h2>Movies suggested for you:</h2>", unsafe_allow_html=True)

    i = 1
    imdb_search_base_url = "https://www.imdb.com/find?q="  # Define the IMDb search base URL here
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        imdb_search_query = title_from_index.replace(" ", "+")  # Convert movie title to a search query
        if (i < 7):
            if(i!=1):
                st.markdown(f"<div class='dust-in'><h3>{i-1}. <a href='{imdb_search_base_url}{imdb_search_query}' target='_blank'>{title_from_index}</a></h3></div>", unsafe_allow_html=True)
            i += 1
else:
    st.write('Waiting')