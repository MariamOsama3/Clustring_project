from requirments.py import *

def visualize_clusters(X, labels, method='PCA'):
    if method == 'PCA':
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    
    reduced_data = reducer.fit_transform(X.toarray() if hasattr(X, 'toarray') else X)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=reduced_data[:, 0], 
        y=reduced_data[:, 1], 
        hue=labels, 
        palette='tab10', 
        alpha=0.7
    )
    plt.title(f"{method} Visualization")
    plt.legend(title="Cluster")
    plt.show()

