from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import Word2Vec
import numpy as np

def kmeans_clustering(data, n_clusters=3, random_state=42):
    """Perform KMeans clustering"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return kmeans.fit_predict(data)

def hierarchical_clustering(data, n_clusters=3, linkage_method='ward'):
    """Perform hierarchical clustering"""
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    return model.fit_predict(data)

def get_linkage_matrix(data, method='ward'):
    """Get linkage matrix for hierarchical clustering"""
    return linkage(data, method=method)

def train_word2vec(tokenized_text, vector_size=100, window=5, min_count=2):
    """Train Word2Vec model"""
    model = Word2Vec(sentences=tokenized_text, 
                    vector_size=vector_size, 
                    window=window, 
                    min_count=min_count, 
                    workers=4)
    return model

def document_vector(words, model):
    """Convert words to document vector using Word2Vec model"""
    valid_words = [word for word in words if word in model.wv]
    if not valid_words:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[valid_words], axis=0)