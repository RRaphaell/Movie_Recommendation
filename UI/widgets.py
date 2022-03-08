import streamlit as st
import constants as const
from script.utils import movie_link, fetch_poster


def initialize_movie_widget(cfg):
    """here we create empty blanks for all recommended movies
    and add description and title from appropriate config file"""
    with st.expander(cfg["title"]):
        st.markdown(cfg["description"])

    movie_cols = st.columns(const.MOVIE_NUMBER)
    for c in movie_cols:
        with c:
            st.empty()

    return movie_cols


def show_recommended_movie_info(recommended_movies, movie_cols, show_score):
    """in this function we get all data what we want to show and put in on webpage"""
    movie_ids = recommended_movies["movieId"]
    movie_titles = recommended_movies["title"]
    movie_scores = recommended_movies["score"]
    posters = [fetch_poster(i) for i in movie_ids]
    links = [movie_link(i) for i in movie_ids]
    for c, t, s, p, l in zip(movie_cols, movie_titles, movie_scores, posters, links):
        with c:
            st.markdown(f"<a style='display: block; text-align: center;' href='{l}'>{t}</a>", unsafe_allow_html=True)
            st.image(p)
            if show_score:
                st.write(round(s, 3))
                