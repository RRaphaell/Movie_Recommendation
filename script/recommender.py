import scipy
import pickle
import requests
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import constants as const


def movie_link(movie_id):
    """generate movie link based on id"""
    return f"https://www.themoviedb.org/movie/{movie_id}"


def fetch_poster(movie_id):
    """generate poster link based on id"""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path


def best_score_based_recommendations():
    """we have already saved dataframe which is sorted based on scores.
    and just read and get top movies"""
    with open('data/movie_scores.pickle', 'rb') as handle:
        movies = pickle.load(handle)
    movies = movies.head(const.MOVIE_NUMBER)
    movies = movies[["id", "title", "score"]]
    movies.columns = ["movieId", "title", "score"]
    return movies


def get_recommendations(movie, titles, cosine_sim):
    """in this function we find similarity score for specific movie sorted
    and gets all metadata for it"""
    indices = pd.Series(movie.index, index=movie['title']).drop_duplicates()
    idx = {indices[t] for t in titles}
    sim_scores = dict()
    for movie_idx in idx:
        sim = cosine_sim[movie_idx]
        for i, s in enumerate(sim):
            sim_scores[i] = s if s > sim_scores.get(i, 0) else sim_scores.get(i, 0)

    for i in idx:
        del sim_scores[i]

    sim_scores = list(sorted(sim_scores.items(), key=lambda item: item[1], reverse=True))[:const.MOVIE_NUMBER]

    movie_indices = [i[0] for i in sim_scores]
    movie_similarity = [i[1] for i in sim_scores]
    return pd.DataFrame(zip(movie['id'].iloc[movie_indices], movie['title'].iloc[movie_indices], movie_similarity),
                        columns=["movieId", "title", "score"])


def contend_based_recommendations(movie, titles):
    """read matrix create similarity function and call main function"""
    tfidf_matrix = scipy.sparse.load_npz('data/tfidf_matrix.npz')
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return get_recommendations(movie, titles, cosine_sim)


def contend_based_recommendations_extra(movie, titles):
    """read matrix create similarity function and call main function"""
    count_matrix = scipy.sparse.load_npz("data/count_matrix.npz")
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    return get_recommendations(movie, titles, cosine_sim)

