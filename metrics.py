

def knn_score(X, Z, k=10):
    from sklearn.neighbors import NearestNeighbors

    X_neighbors = NearestNeighbors(n_neighbors=k).fit(X)
    X_indices = X_neighbors.kneighbors(return_distance=False)

    Z_neighbors = NearestNeighbors(n_neighbors=k).fit(Z)
    Z_indices = Z_neighbors.kneighbors(return_distance=False)

    #intersections = sum(
    #    map(lambda i: len(set(X_indices[i]) & set(Z_indices[i])), range(X.shape[0]))
    #    )

    intersections = 0.0
    for i in range(X.shape[0]):
        intersections += len(set(X_indices[i]) & set(Z_indices[i]))
    intersections = intersections / X.shape[0] / k
    return intersections


# from repo
def embedding_quality(X, Z, classes, knn=10, knn_classes=10, subsetsize=1000):
    from sklearn.neighbors import NearestNeighbors
    from scipy.spatial.distance import pdist


    nbrs1 = NearestNeighbors(n_neighbors=knn).fit(X)
    ind1 = nbrs1.kneighbors(return_distance=False)

    nbrs2 = NearestNeighbors(n_neighbors=knn).fit(Z)
    ind2 = nbrs2.kneighbors(return_distance=False)

    intersections = 0.0
    for i in range(X.shape[0]):
        intersections += len(set(ind1[i]) & set(ind2[i]))
    mnn = intersections / X.shape[0] / knn
    
    cl, cl_inv = np.unique(classes, return_inverse=True)
    C = cl.size
    mu1 = np.zeros((C, X.shape[1]))
    mu2 = np.zeros((C, Z.shape[1]))
    for c in range(C):
        mu1[c,:] = np.mean(X[cl_inv==c,:], axis=0)
        mu2[c,:] = np.mean(Z[cl_inv==c,:], axis=0)
        
    nbrs1 = NearestNeighbors(n_neighbors=knn_classes).fit(mu1)
    ind1 = nbrs1.kneighbors(return_distance=False)
    nbrs2 = NearestNeighbors(n_neighbors=knn_classes).fit(mu2)
    ind2 = nbrs2.kneighbors(return_distance=False)
    
    intersections = 0.0
    for i in range(C):
        intersections += len(set(ind1[i]) & set(ind2[i]))
    mnn_global = intersections / C / knn_classes
    
    subset = np.random.choice(X.shape[0], size=subsetsize, replace=False)
    d1 = pdist(X[subset,:])
    d2 = pdist(Z[subset,:])
    rho = scipy.stats.spearmanr(d1[:,None],d2[:,None]).correlation
    
    return (mnn, mnn_global, rho)
