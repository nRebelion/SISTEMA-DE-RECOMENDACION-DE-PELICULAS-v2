from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def content_recommendation(df, movie_title, top_n=5):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["genres"])

    similarity = cosine_similarity(tfidf_matrix)
    indices = pd.Series(df.index, index=df["title"])

    idx = indices[movie_title]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    return df["title"].iloc[[i[0] for i in scores]].tolist()
