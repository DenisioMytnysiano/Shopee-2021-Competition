from sklearn.feature_extraction.text import TfidfVectorizer


def get_tf_idf_embeddings(titles_array: list):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(titles_array)
    return embeddings.toarray()
