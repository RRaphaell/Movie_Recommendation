import pickle
import streamlit as st
import streamlit.components.v1 as components
from script.recommender import contend_based_recommendations, weighted_average_based_recommendations, contend_based_recommendations_extra
from config import score_based_cfg, content_based_cfg, content_extra_based_cfg
from UI.widgets import initialize_movie_widget, show_recommended_movie_info
import constants as const
st.set_page_config(page_title="Recommender system", layout="wide")


# add styling
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# load main movie dataframe
with open('data/movie_df.pickle', 'rb') as handle:
    movie = pickle.load(handle)

st.markdown('# Movie Recommender system')
social_components = open("assets/social_components.html", 'r', encoding='utf-8')
components.html(social_components.read())

# add search panel and search button
main_layout, search_layout = st.columns([10, 1])
options = main_layout.multiselect('Which movies do you like?', movie["title"].unique())
show_recommended_movies_btn = search_layout.button("search")

# add widgets on sidebar
recommended_movie_num = st.sidebar.slider("Recommended movie number", min_value=5, max_value=10)
if recommended_movie_num:
    const.MOVIE_NUMBER = recommended_movie_num
show_score = st.sidebar.checkbox("Show score")

# create horizontal layouts for movies
col_for_score_based = initialize_movie_widget(score_based_cfg)
col_for_content_based = initialize_movie_widget(content_based_cfg)
col_for_content_based_extra = initialize_movie_widget(content_extra_based_cfg)

# show recommended movies based on weighted average (this is same for all movies)
score_based_recommended_movies = weighted_average_based_recommendations()
show_recommended_movie_info(score_based_recommended_movies, col_for_score_based, show_score)

# when search clicked
if show_recommended_movies_btn:
    contend_based_recommended_movies = contend_based_recommendations(movie, options)
    show_recommended_movie_info(contend_based_recommended_movies, col_for_content_based, show_score)

    contend_extra_based_recommended_movies = contend_based_recommendations_extra(movie, options)
    show_recommended_movie_info(contend_extra_based_recommended_movies, col_for_content_based_extra, show_score)
