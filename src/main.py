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

if __name__ == "__main__":
    main()