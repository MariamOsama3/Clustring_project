def hierarchical_clustering(X, n_samples=200, n_clusters=3):
    linkage_matrix = linkage(X[:n_samples].toarray(), method='ward')
    
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, truncate_mode='level', p=10)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.show()
    
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    return clusterer.fit_predict(X.toarray())

def lda_clustering(X, n_components=3):
    lda = LatentDirichletAllocation(n_components=n_components, random_state=42)
    lda_matrix = lda.fit_transform(X)
    return np.argmax(lda_matrix, axis=1), lda.perplexity(X)

def train_word2vec(tokenized_text, vector_size=100, window=5, min_count=2, workers=4):
    """Train a Word2Vec model on tokenized text"""
    model = Word2Vec(
        sentences=tokenized_text,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )
    return model

def get_document_vectors(tokenized_text, model):
    """Convert tokenized documents to document vectors by averaging word embeddings"""
    doc_vectors = []
    for doc in tokenized_text:
        valid_words = [word for word in doc if word in model.wv]
        if valid_words:
            doc_vectors.append(np.mean(model.wv[valid_words], axis=0))
        else:
            doc_vectors.append(np.zeros(model.vector_size))
    return np.array(doc_vectors)

def word2vec_clustering(text_series, n_clusters=3, vector_size=100, 
                       visualize=True, true_labels=None):
    """
    Complete Word2Vec clustering pipeline:
    1. Tokenizes text
    2. Trains Word2Vec model
    3. Creates document vectors
    4. Performs K-means clustering
    5. (Optional) Visualizes results
    6. Returns metrics
    """
    # Tokenize text
    tokenized = text_series.apply(lambda x: x.split())
    
    # Train Word2Vec model
    w2v_model = train_word2vec(tokenized, vector_size=vector_size)
    
    # Get document vectors
    doc_vectors = get_document_vectors(tokenized, w2v_model)
    
    # Cluster documents
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=100)
    cluster_labels = kmeans.fit_predict(doc_vectors)
    
    # Evaluate
    metrics = evaluate_clustering(doc_vectors, cluster_labels, true_labels)
    
    # Visualize if requested
    if visualize:
        # PCA Visualization
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(doc_vectors)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], 
                        hue=cluster_labels, palette='tab10', alpha=0.7)
        plt.title("Word2Vec Clustering (PCA)")
        plt.show()
        
        # t-SNE Visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced_data = tsne.fit_transform(doc_vectors)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], 
                        hue=cluster_labels, palette='tab10', alpha=0.7)
        plt.title("Word2Vec Clustering (t-SNE)")
        plt.show()
    
    return {
        'labels': cluster_labels,
        'model': w2v_model,
        'vectors': doc_vectors,
        'metrics': metrics
    }