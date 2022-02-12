import scipy
import requests
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


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


def best_score_based_recommendations(num_movies=5):
    """we have already saved dataframe which is sorted based on scores.
    and just read and get top movies"""
    movies = pd.read_pickle("data/movie_scores.pkl")
    movies = movies.head(num_movies)
    return pd.DataFrame(zip(movies['id'], movies['title'], movies["score"]),
                        columns=["movieId", "title", "score"])


def get_recommendations(movie, title, cosine_sim, num_movies=10):
    """in this function we find similarity score for specific movie sorted
    and gets all metadata for it"""

    indices = pd.Series(movie.index, index=movie['title']).drop_duplicates()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_movies+1]  # first is the same movie
    movie_indices = [i[0] for i in sim_scores]
    movie_similarity = [i[1] for i in sim_scores]
    return pd.DataFrame(zip(movie['id'].iloc[movie_indices], movie['title'].iloc[movie_indices], movie_similarity),
                        columns=["movieId", "title", "score"])


def contend_based_recommendations(movie, title):
    """read matrix create similarity function and call main function"""
    tfidf_matrix = scipy.sparse.load_npz('data/tfidf_matrix.npz')
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return get_recommendations(movie, title, cosine_sim)


def contend_based_recommendations_extra(movie, title):
    """read matrix create similarity function and call main function"""
    count_matrix = scipy.sparse.load_npz("data/count_matrix.npz")
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    return get_recommendations(movie, title, cosine_sim)

