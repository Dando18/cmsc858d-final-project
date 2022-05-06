import numpy as np

def get_dim_reduction(X, algorithm='pca', **kwargs):

    if algorithm == 'pca':
        return pca(X, **kwargs)
    elif algorithm == 'tsne':
        return tsne(X, **kwargs)
    elif algorithm == 'umap':
        return umap(X, **kwargs)
    else:
        raise NotImplementedError('{} not supported yet.'.format(algorithm))


def pca(ds):
    ''' Compute the pca dim reduction
    '''
    raise NotImplementedError('pca not supported yet.')


def tsne(X, perplexity=30, learning_rate='auto', init='random'):
    ''' Compute the t-SNE dimensionality reduction.
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
