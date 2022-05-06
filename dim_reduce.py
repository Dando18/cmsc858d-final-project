import numpy as np
from analysis_tools import preprocess

def get_dim_reduction(ds, algorithm='pca', **kwargs):

    if algorithm == 'pca':
        return pca(ds, **kwargs)
    elif algorithm == 'tsne':
        return tsne(ds, **kwargs)
    elif algorithm == 'umap':
        return umap(ds, **kwargs)
    else:
        raise NotImplementedError('{} not supported yet.'.format(algorithm))


def pca(ds):
    ''' Compute the pca dim reduction
    '''
    raise NotImplementedError('pca not supported yet.')


def tsne(ds, perplexity=30, learning_rate='auto', init=None):
    ''' Compute the t-SNE dimensionality reduction.
    '''
    from openTSNE import TSNE

    X = ds['counts']
    X = preprocess(X, normalize=True, subsets=ds['markerSubset'])

    if learning_rate == 'auto':
        learning_rate = X.shape[0]/12
    
    model = TSNE(perplexity=perplexity, learning_rate=learning_rate, initialization=init)
    reduced_data = model.fit(X)

    return reduced_data


def umap(ds):
    '''
    '''
    raise NotImplementedError('UMAP not supported yet.')