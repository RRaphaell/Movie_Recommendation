import requests
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


def movie_link(movie_id):
    return f"https://www.themoviedb.org/movie/{movie_id}"


def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path


def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']

    return (v/(v+m) * R) + (m/(m+v) * C)


def best_score_based_recommendations(movie):
    C = movie['vote_average'].mean()
    m = movie['vote_count'].quantile(0.9)
    q_movies = movie.copy().loc[movie['vote_count'] >= m]

    q_movies['score'] = q_movies.apply(lambda x: weighted_rating(x, m, C), axis=1)
    q_movies = q_movies.sort_values('score', ascending=False)
    q_movies = q_movies.head(5)

    return pd.DataFrame(zip(q_movies['id'], q_movies['title'], q_movies["score"]), columns=["movieId", "title", "score"])


def contend_based_recommendations(movie, indices, title, cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    movie_similarity = [i[1] for i in sim_scores]

    return pd.DataFrame(zip(movie['id'].iloc[movie_indices], movie['title'].iloc[movie_indices], movie_similarity), columns=["movieId", "title", "score"])


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]

        if len(names) > 3:
            names = names[:3]
        return names

    return []


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


def contend_based_recommendations_extra(movie, title):
    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        movie[feature] = movie[feature].apply(literal_eval)

    movie['director'] = movie['crew'].apply(get_director)
    features = ['cast', 'keywords', 'genres']
    for feature in features:
        movie[feature] = movie[feature].apply(get_list)

    features = ['cast', 'keywords', 'director', 'genres']
    for feature in features:
        movie[feature] = movie[feature].apply(clean_data)

    movie['soup'] = movie.apply(create_soup, axis=1)

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(movie['soup'])

    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

    movie = movie.reset_index()
    indices = pd.Series(movie.index, index=movie['title'])

    return contend_based_recommendations(movie, indices, title, cosine_sim2)
