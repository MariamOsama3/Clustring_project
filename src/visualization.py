import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_elbow_method(k_values, inertias):
    """Plot elbow method results"""
    plt.plot(k_values, inertias, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.show()

def plot_silhouette_scores(k_values, scores):
    """Plot silhouette scores"""
    plt.plot(k_values, scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Method')
    plt.show()

def plot_pca_clusters(data, clusters, n_components=2):
    """Visualize clusters using PCA"""
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    
    df_pca = pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(n_components)])
    df_pca['Cluster'] = clusters
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='tab10', data=df_pca, alpha=0.7)
    plt.title(f"PCA Visualization of {len(set(clusters))} Clusters")
    plt.legend(title="Cluster")
    plt.show()

def plot_tsne_clusters(data, clusters, perplexity=30):
    """Visualize clusters using t-SNE"""
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    reduced_data = tsne.fit_transform(data)
    
    df_tsne = pd.DataFrame(reduced_data, columns=['TSNE1', 'TSNE2'])
    df_tsne['Cluster'] = clusters
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='TSNE1', y='TSNE2', hue='Cluster', palette='tab10', data=df_tsne, alpha=0.7)
    plt.title(f"t-SNE Visualization of {len(set(clusters))} Clusters")
    plt.legend(title="Cluster")
    plt.show()

def plot_dendrogram(linkage_matrix, truncate_level=10):
    """Plot hierarchical clustering dendrogram"""
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, truncate_mode='level', p=truncate_level)
    plt.xlabel("Document Index")
    plt.ylabel("Distance")
    plt.title("Hierarchical Clustering Dendrogram")
    plt.show()
