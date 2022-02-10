import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from script.recommender import contend_based_recommendations, best_score_based_recommendations, contend_based_recommendations_extra
from config import score_based_cfg, content_based_cfg, content_extra_based_cfg
from utils import create_recommender_system, show_recommended_movie_info

st.set_page_config(page_title="Reccomender system")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

PATH = "tmdb"
credit = pd.read_csv(os.path.join(PATH, 'tmdb_5000_credits.csv'))
movie = pd.read_csv(os.path.join(PATH, 'tmdb_5000_movies.csv'))
credit.columns = ['id', 'tittle', 'cast', 'crew']
movie = movie.merge(credit, on='id')
tfidf = TfidfVectorizer(stop_words='english')
movie['overview'] = movie['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(movie['overview'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movie.index, index=movie['title']).drop_duplicates()


st.markdown('# Movie Recommender system')
main_layout, search_layout = st.columns([10, 1])
options = main_layout.multiselect('Which movies do you like?', movie["title"].unique())
show_recommended_movies_btn = search_layout.button("search")

col_for_score_based = create_recommender_system(score_based_cfg)
col_for_content_based = create_recommender_system(content_based_cfg)
col_for_content_based_extra = create_recommender_system(content_extra_based_cfg)

if show_recommended_movies_btn:
    score_based_recommended_movies = best_score_based_recommendations(movie)
    show_recommended_movie_info(score_based_recommended_movies, col_for_score_based)

    contend_based_recommended_movies = contend_based_recommendations(movie, indices, options[0], cosine_sim)
    show_recommended_movie_info(contend_based_recommended_movies, col_for_content_based)

    contend_extra_based_recommended_movies = contend_based_recommendations_extra(movie, options[0])
    show_recommended_movie_info(contend_extra_based_recommended_movies, col_for_content_based_extra)
