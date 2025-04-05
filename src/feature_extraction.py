from requirments.py import *

def vectorize_text(text_series, vectorizer_type='tfidf', max_features=15000):
    vectorizer = TfidfVectorizer(max_features=max_features) if vectorizer_type == 'tfidf' else CountVectorizer(max_features=max_features)
    numeric_text = vectorizer.fit_transform(text_series)
    return Normalizer().fit_transform(numeric_text)

def train_word2vec(tokenized_text, vector_size=100, window=5, min_count=2):
    return Word2Vec(
        sentences=tokenized_text,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4
    )

def document_vector(words, model):
    valid_words = [word for word in words if word in model.wv]
    return np.zeros(model.vector_size) if not valid_words else np.mean(model.wv[valid_words], axis=0)