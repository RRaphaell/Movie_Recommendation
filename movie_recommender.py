import pandas as pd
import streamlit as st
from script.recommender import contend_based_recommendations, best_score_based_recommendations, contend_based_recommendations_extra
from config import score_based_cfg, content_based_cfg, content_extra_based_cfg
from utils import create_recommender_system, show_recommended_movie_info

st.set_page_config(page_title="Recommender system")

# add styling
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

movie = pd.read_pickle("data/movies.pkl")

# add search panel and button widget
st.markdown('# Movie Recommender system')
main_layout, search_layout = st.columns([10, 1])
options = main_layout.multiselect('Which movies do you like?', movie["title"].unique())
show_recommended_movies_btn = search_layout.button("search")

# create horizontal layouts for movies
col_for_score_based = create_recommender_system(score_based_cfg)
col_for_content_based = create_recommender_system(content_based_cfg)
col_for_content_based_extra = create_recommender_system(content_extra_based_cfg)

# when search clicked
if show_recommended_movies_btn:
    score_based_recommended_movies = best_score_based_recommendations()
    show_recommended_movie_info(score_based_recommended_movies, col_for_score_based)

    contend_based_recommended_movies = contend_based_recommendations(movie, options[0])
    show_recommended_movie_info(contend_based_recommended_movies, col_for_content_based)

    contend_extra_based_recommended_movies = contend_based_recommendations_extra(movie, options[0])
    show_recommended_movie_info(contend_extra_based_recommended_movies, col_for_content_based_extra)
