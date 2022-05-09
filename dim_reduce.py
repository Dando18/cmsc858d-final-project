import numpy as np

def get_dim_reduction(X, algorithm='pca', **kwargs):
    algorithm = algorithm.lower()
    func_map = {'pca': pca, 'tsne': tsne, 'umap': umap, 'densmap': densmap}

    if algorithm in func_map:
        return func_map[algorithm](X, **kwargs)
    else:
        raise NotImplementedError('{} not supported yet.'.format(algorithm))


def pca(X):
    ''' Compute the pca dim reduction
    '''
    from sklearn.decomposition import PCA

    model = PCA(n_components=2, whiten=False)
    reduced_data = model.fit_transform(X)

    return reduced_data


def tsne(X, perplexity=30, learning_rate='auto', init='random', metric='euclidean'):
    ''' Compute the t-SNE dimensionality reduction.
        openTSNE API: https://opentsne.readthedocs.io/en/latest/api/index.html
        metrics:  "cosine", "euclidean", "manhattan", "hamming", "dot", "l1", "l2", "taxicab"
    '''
    from openTSNE import TSNE

    if learning_rate == 'auto':
        learning_rate = X.shape[0]/12
    
    model = TSNE(perplexity=perplexity, learning_rate=learning_rate)
    reduced_data = model.fit(X)

    return reduced_data


def umap(X, n_neighbors=15, metric='euclidean'):
    ''' UMAP API: https://umap-learn.readthedocs.io/en/latest/api.html
    '''
    from umap import UMAP

    model = UMAP(n_neighbors=n_neighbors, metric=metric)
    reduced_data = model.fit_transform(X)

    return reduced_data

def densmap(X, n_neighbors=15, metric='euclidean'):
    ''' UMAP API: https://umap-learn.readthedocs.io/en/latest/api.html
    '''
    from umap import UMAP

    model = UMAP(densmap=True, n_neighbors=n_neighbors, metric=metric)
    reduced_data = model.fit_transform(X)

    return reduced_data
