import numpy as np

def knn_score(X, Z, k=10):
    from sklearn.neighbors import NearestNeighbors

    X_neighbors = NearestNeighbors(n_neighbors=k).fit(X)
    X_indices = X_neighbors.kneighbors(return_distance=False)

    Z_neighbors = NearestNeighbors(n_neighbors=k).fit(Z)
    Z_indices = Z_neighbors.kneighbors(return_distance=False)

    intersections = 0.0
    for i in range(X.shape[0]):
        intersections += len(set(X_indices[i]) & set(Z_indices[i]))
    intersections = intersections / X.shape[0] / k
    return intersections


def knc_score(X, Z, classes, k=10):
    from sklearn.neighbors import NearestNeighbors

    unique_classes, unique_classes_inv = np.unique(classes, return_inverse=True)
    num_unique_classes = unique_classes.size
    X_class_means = np.zeros((num_unique_classes, X.shape[1]))
    Z_class_means = np.zeros((num_unique_classes, Z.shape[1]))
    for c in range(num_unique_classes):
        X_class_means[c,:] = np.mean(X[unique_classes_inv==c,:], axis=0)
        Z_class_means[c,:] = np.mean(Z[unique_classes_inv==c,:], axis=0)
    
    X_neighbors = NearestNeighbors(n_neighbors=k).fit(X_class_means)
    X_indices = X_neighbors.kneighbors(return_distance=False)

    Z_neighbors = NearestNeighbors(n_neighbors=k).fit(Z_class_means)
    Z_indices = Z_neighbors.kneighbors(return_distance=False)

    intersections = 0.0
    for i in range(num_unique_classes):
        intersections += len(set(X_indices[i]) & set(Z_indices[i]))
    intersections = intersections / num_unique_classes / k

    return intersections


def cpd_score(X, Z, subset_size=1000):
    from scipy.stats import spearmanr
    from scipy.spatial.distance import pdist

    subsets = np.random.choice(X.shape[0], size=subset_size, replace=False)
    X_distances = pdist(X[subsets,:])
    Z_distances = pdist(Z[subsets,:])

    cpd = spearmanr(X_distances[:,None], Z_distances[:,None]).correlation
    return cpd
