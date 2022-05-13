import numpy as np

def get_dim_reduction(X, algorithm='pca', **kwargs):
    algorithm = algorithm.lower()
    func_map = {'pca': pca, 'lda': lda, 'tsne': tsne, 'umap': umap, 'densne': densne, 
                'densmap': densmap, 'scvis': scvis, 'netsne': netsne, 'hsne': hsne}

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


def lda(X, classes=None):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    model = LinearDiscriminantAnalysis(n_components=2)
    reduced_data = model.fit_transform(X, classes)

    return reduced_data


def tsne(X, perplexity=30, learning_rate='auto', initialization='random', metric='euclidean'):
    ''' Compute the t-SNE dimensionality reduction.
        openTSNE API: https://opentsne.readthedocs.io/en/latest/api/index.html
        metrics:  "cosine", "euclidean", "manhattan", "hamming", "dot", "l1", "l2", "taxicab"
    '''
    from openTSNE import TSNE
    
    model = TSNE(perplexity=perplexity, learning_rate=learning_rate, 
        initialization=initialization, metric=metric)
    reduced_data = model.fit(X)

    return reduced_data


def umap(X, n_neighbors=15, metric='euclidean'):
    ''' UMAP API: https://umap-learn.readthedocs.io/en/latest/api.html
    '''
    from umap import UMAP

    model = UMAP(n_neighbors=n_neighbors, metric=metric)
    reduced_data = model.fit_transform(X)

    return reduced_data


def densne(X, dens_lambda=0.1, dens_frac=0.3):
    ''' densne API: https://github.com/hhcho/densvis/tree/master/densne
        Run following command in densvis/densne subdirectory first.
            g++ sptree.cpp densne.cpp densne_main.cpp -o den_sne -O2
    '''
    import sys
    sys.path.append('./densvis/densne')
    from densne import run_densne
    
    reduced_data = run_densne(X, verbose=True, final_dens=False, dens_lambda=dens_lambda, dens_frac=dens_frac)

    return reduced_data

def densmap(X, n_neighbors=15, metric='euclidean', dens_lambda=2.0, dens_frac=0.3):
    ''' UMAP API: https://umap-learn.readthedocs.io/en/latest/api.html
    '''
    from umap import UMAP

    model = UMAP(densmap=True, n_neighbors=n_neighbors, metric=metric, dens_lambda=dens_lambda, dens_frac=dens_frac)
    reduced_data = model.fit_transform(X)

    return reduced_data

def scvis(X):
    ''' scvis repo: https://github.com/shahcompbio/scvis
    '''
    from scvis.run import train
    from dataclasses import dataclass
    import tempfile
    import os.path
    from os import listdir
    import pandas as pd

    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    @dataclass
    class ScvisArgs:
        data_matrix_file: str
        out_dir: str
        config_file: str = None
        pretrained_model_file: str = None
        data_label_file: str = None
        normalize: bool = True
        verbose: bool = False
        verbose_interval: int = 100
        show_plot: bool = False

    reduced = None
    #with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = 'scvis-output'
    mat_fname = os.path.join(tmpdir, 'data.tsv')

    pd.DataFrame(X).to_csv(mat_fname, index=False, sep='\t')

    args = ScvisArgs(mat_fname, tmpdir, None, None, None, True, True)
    train(args)

    output_fname = [filename for filename in listdir(tmpdir) if filename.startswith("perplexity_") and filename.endswith('tsv') and (not filename.endswith('likelihood.tsv'))][0]
    output_fname = os.path.join(tmpdir, output_fname)
    reduced = pd.read_csv(output_fname, sep='\t', usecols=['z_coordinate_0', 'z_coordinate_1'])

    return reduced.to_numpy()


def netsne(X):
    '''
    '''
    pass



def hsne(X):
    '''
    '''
    pass
