import numpy as np

def preprocess(X, normalize=True, subsets=None):
    magnitudes = np.sum(X, axis=1)
    if len(magnitudes.shape) == 1:
        magnitudes = np.expand_dims(magnitudes, axis=1)

    if subsets is not None:
        X = X[:, subsets]

    if normalize:
        X = np.log2(X / magnitudes * 1e+6 + 1)
        X = np.array(X)
        X -= X.mean(axis=0)

        U,s,V = np.linalg.svd(X, full_matrices=False)
        U[:, np.sum(V,axis=1)<0] *= -1
        X = np.dot(U, np.diag(s))
        X = X[:, np.argsort(s)[::-1]][:,:50]
    
    return X
