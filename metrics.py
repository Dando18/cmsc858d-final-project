import numpy as np

def knn_score(X, Z, k=10, metric='euclidean'):
    from sklearn.neighbors import NearestNeighbors

    X_neighbors = NearestNeighbors(n_neighbors=k, metric=metric).fit(X)
    X_indices = X_neighbors.kneighbors(return_distance=False)

    Z_neighbors = NearestNeighbors(n_neighbors=k, metric=metric).fit(Z)
    Z_indices = Z_neighbors.kneighbors(return_distance=False)

    intersections = 0.0
    for i in range(X.shape[0]):
        intersections += len(set(X_indices[i]) & set(Z_indices[i]))
    intersections = intersections / X.shape[0] / k
    return intersections


def knc_score(X, Z, classes, k=10, metric='euclidean'):
    from sklearn.neighbors import NearestNeighbors

    unique_classes, unique_classes_inv = np.unique(classes, return_inverse=True)
    num_unique_classes = unique_classes.size
    X_class_means = np.zeros((num_unique_classes, X.shape[1]))
    Z_class_means = np.zeros((num_unique_classes, Z.shape[1]))
    for c in range(num_unique_classes):
        X_class_means[c,:] = np.mean(X[unique_classes_inv==c,:], axis=0)
        Z_class_means[c,:] = np.mean(Z[unique_classes_inv==c,:], axis=0)
    
    X_neighbors = NearestNeighbors(n_neighbors=k, metric=metric).fit(X_class_means)
    X_indices = X_neighbors.kneighbors(return_distance=False)

    Z_neighbors = NearestNeighbors(n_neighbors=k, metric=metric).fit(Z_class_means)
    Z_indices = Z_neighbors.kneighbors(return_distance=False)

    intersections = 0.0
    for i in range(num_unique_classes):
        intersections += len(set(X_indices[i]) & set(Z_indices[i]))
    intersections = intersections / num_unique_classes / k

    return intersections


def cpd_score(X, Z, subset_size=1000, metric='euclidean'):
    from scipy.stats import spearmanr
    from scipy.spatial.distance import pdist

    if metric == 'l1':
        metric = 'cityblock'

    if subset_size > X.shape[0]:
        subset_size = X.shape[0]

    subsets = np.random.choice(X.shape[0], size=subset_size, replace=False)
    X_distances = pdist(X[subsets,:], metric=metric)
    Z_distances = pdist(Z[subsets,:], metric=metric)

    cpd = spearmanr(X_distances[:,None], Z_distances[:,None]).correlation
    return cpd


def ari_score(X, Z, classes):
    ''' Computes adjusted Rand index
        see page 5 of https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6469860/pdf/nihms-1021959.pdf
    '''
    from sklearn.metrics import adjusted_rand_score
    from sklearn.cluster import KMeans

    # run k-means on the dataset
    num_clusters = np.unique(classes).size
    model = KMeans(n_clusters=num_clusters)
    kmeans = model.fit(Z)

    ari = adjusted_rand_score(classes, kmeans.labels_)

    return ari


def cs_score(X, Z, classes):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, ShuffleSplit

    model = RandomForestClassifier()
    cv = ShuffleSplit(n_splits=3, test_size=0.4)
    scores = cross_val_score(model, Z, classes, cv=cv)

    return scores.mean()


def pds_score(X, Z, sample_size=1000, metric='euclidean'):
    import random
    from scipy.spatial.distance import pdist
    from scipy.stats import linregress
    from sklearn.metrics import r2_score

    if metric == 'l1':
        metric = 'cityblock'

    if sample_size > X.shape[0]:
        sample_size = X.shape[0]-1

    points = random.sample(list(zip(X, Z)), sample_size)
    X_points, Z_points = [], []
    for xp, zp in points:
        X_points.append(xp)
        Z_points.append(zp)
    
    X_distances = pdist(X_points, metric=metric)
    Z_distances = pdist(Z_points, metric=metric)
    
    try:
        _, _, r_value, _, _ = linregress(X_distances, Z_distances)
    except ValueError:
        return 0
    return r_value**2
