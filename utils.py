import streamlit as st
from script.recommender import fetch_poster, movie_link


def create_recommender_system(cfg, movie_num=5):
    """here we create empty blanks for all recommended movies
    and add description and title from appropriate config file"""
    with st.expander(cfg["title"]):
        st.markdown(cfg["description"])
    movie_cols = st.columns(movie_num)
    for c in movie_cols:
        with c:
            st.empty()

    return movie_cols


def show_recommended_movie_info(recommended_movies, movie_cols):
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
            st.write(round(s, 3))
