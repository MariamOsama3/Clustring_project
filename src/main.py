from data_loading import load_20newsgroups_data, load_wiki_data
from preprocessing import preprocess_text, vectorize_text
from clustering import kmeans_clustering, hierarchical_clustering, get_linkage_matrix
from visualization import (plot_elbow_method, plot_silhouette_scores, 
                         plot_pca_clusters, plot_tsne_clusters, plot_dendrogram)
from evaluation import calculate_silhouette_score, purity_score
from config import DEFAULT_N_CLUSTERS, MAX_FEATURES
import pandas as pd

def main():
    # Load and preprocess 20 Newsgroups data
    print("Processing 20 Newsgroups dataset...")
    news_df = load_20newsgroups_data()
    news_df['cleaned_text'] = news_df['text'].apply(preprocess_text)
    news_vectors = vectorize_text(news_df['cleaned_text'], max_features=MAX_FEATURES)
    
    # KMeans clustering
    kmeans_labels = kmeans_clustering(news_vectors, n_clusters=DEFAULT_N_CLUSTERS)
    sil_score = calculate_silhouette_score(news_vectors, kmeans_labels)
    pur_score = purity_score(news_df['category'], kmeans_labels)
    print(f"KMeans - Silhouette: {sil_score:.4f}, Purity: {pur_score:.4f}")
    
    # Visualize
    plot_pca_clusters(news_vectors, kmeans_labels)
    plot_tsne_clusters(news_vectors, kmeans_labels)

    # Hierarchical clustering
    linkage_matrix = get_linkage_matrix(news_vectors[:200])  # Use subset for dendrogram
    plot_dendrogram(linkage_matrix)

    # Use Word2Vec
    print("\nApplying Word2Vec clustering...")
    # Tokenize text
    tokenized = news_df['cleaned_text'].apply(lambda x: x.split())
    
    # Train Word2Vec model
    w2v_model = train_word2vec(tokenized)  # Using imported function
    
    # Create document vectors
    doc_vectors = np.array([document_vector(doc, w2v_model) for doc in tokenized])
    
    # Cluster with KMeans
    k = 3  # Using optimal k from elbow method
    w2v_labels = kmeans_clustering(doc_vectors, n_clusters=k)  # Using imported function
    
    # Evaluate
    sil_score = silhouette_score(doc_vectors, w2v_labels)
    print(f"Word2Vec clustering Silhouette Score: {sil_score:.4f}")
    
    # Load and preprocess Wikipedia data
    print("\nProcessing Wikipedia dataset...")
    wiki_df = load_wiki_data()
    wiki_df['cleaned_text'] = wiki_df['text'].apply(preprocess_text)
    wiki_vectors = vectorize_text(wiki_df['cleaned_text'], max_features=MAX_FEATURES)
    
    # Hierarchical clustering
    linkage_matrix = get_linkage_matrix(wiki_vectors[:200])  # Use subset for dendrogram
    plot_dendrogram(linkage_matrix)
    
    hier_labels = hierarchical_clustering(wiki_vectors, n_clusters=15)
    sil_score = calculate_silhouette_score(wiki_vectors, hier_labels)
    print(f"Hierarchical - Silhouette: {sil_score:.4f}")

    # KMeans clustering
    kmeans_labels = kmeans_clustering(wiki_vectors, n_clusters=DEFAULT_N_CLUSTERS)
    sil_score = calculate_silhouette_score(wiki_vectors, kmeans_labels)
    print(f"KMeans - Silhouette: {sil_score:.4f})
    
    # Visualize
    plot_pca_clusters(wiki_vectors, kmeans_labels)
    plot_tsne_clusters(wiki_vectors, kmeans_labels)



if __name__ == "__main__":
    main()
