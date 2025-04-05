from requirments.py import *
def purity_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    clusters = np.unique(y_pred)
    correct_preds = 0
    
    for cluster in clusters:
        true_labels = y_true[y_pred == cluster]
        if len(true_labels) == 0:
            continue
        majority_label = mode(true_labels, keepdims=True).mode[0]
        correct_preds += np.sum(true_labels == majority_label)
    
    return correct_preds / len(y_true)

def evaluate_clustering(X, labels, true_labels=None):
    silhouette = silhouette_score(X, labels)
    results = {'silhouette': silhouette}
    
    if true_labels is not None:
        results['purity'] = purity_score(true_labels, labels)
    
    return results

def plot_elbow_and_silhouette(X, max_k=10):
    inertias, silhouettes = [], []
    k_range = range(2, max_k+1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=100)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(k_range, inertias, marker='o')
    ax1.set_title('Elbow Method')
    ax2.plot(k_range, silhouettes, marker='o')
    ax2.set_title('Silhouette Scores')
    plt.show()
    
    return k_range[np.argmax(silhouettes)]
