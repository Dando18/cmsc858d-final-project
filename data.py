import os.path

import numpy as np
from scipy import sparse
import pandas as pd
import pickle
import logging

import rnaseqTools

def get_mouse_exon_dataset():
    PKL_FPATH = './data/mouse-exon.pkl'

    if os.path.isfile(PKL_FPATH):
        dataset = pickle.load(open(PKL_FPATH, 'rb'))
    else:
        fnames = ['./data/mouse_VISp_2018-06-14_exon-matrix.csv', 
            './data/mouse_ALM_2018-06-14_exon-matrix.csv']
        
        datasets = [rnaseqTools.sparseload(fname) for fname in fnames]
        
        counts = sparse.vstack([c[0] for c in datasets])
        cells = np.concatenate([c[2] for c in datasets])

        for d in datasets:
            gene = d[1]
            assert np.all(gene == datasets[0][1])
        genes = np.copy(datasets[0][1])

        genesTable = pd.read_csv('./data/mouse_VISp_2018-06-14_genes-rows.csv')
        ids = genesTable['gene_entrez_id'].tolist()
        symbols = genesTable['gene_symbol'].tolist()
        id2symbol = dict(zip(ids, symbols))
        genes = np.array([id2symbol[g] for g in genes])

        clusterInfo = pd.read_csv('data/tasic-sample_heatmap_plot_data.csv')
        goodCells = clusterInfo['sample_name'].values
        ids = clusterInfo['cluster_id'].values
        labels = clusterInfo['cluster_label'].values
        colors = clusterInfo['cluster_color'].values

        clusterNames = np.array([labels[ids==i+1][0] for i in range(np.max(ids))])
        clusterColors = np.array([colors[ids==i+1][0] for i in range(np.max(ids))])
        clusters = np.copy(ids)

        indices = np.array([np.where(cells==c)[0][0] for c in goodCells])
        counts = counts[indices, :]

        areas = (indices < datasets[0][2].size).astype(int)
        clusters -= 1

        markers = ['Snap25','Gad1','Slc17a7','Pvalb', 'Sst', 'Vip', 'Aqp4', 
           'Mog', 'Itgam', 'Pdgfra', 'Flt1', 'Bgn', 'Rorb', 'Foxp2']
        markerSubset = rnaseqTools.geneSelection(
            counts, n=3000, threshold=32, 
            markers=markers, genes=genes)

        dataset = {'counts': counts, 'genes': genes, 'clusters': clusters, 'areas': areas, 
                'clusterColors': clusterColors, 'clusterNames': clusterNames,
                'markerSubset': markerSubset}
        
        pickle.dump(dataset, open(PKL_FPATH, 'wb'))
    
    logging.info('Read data size with size: {}'.format(dataset['counts'].shape))
    logging.info('Areas: {} {}'.format(np.sum(dataset['areas']==0), np.sum(dataset['areas']==1)))
    logging.info('Unique clusters: {}'.format(np.unique(dataset['clusters']).size))

    return dataset


def get_ca1_neurons():
    PKL_FPATH = './data/mouse-ca1-neurons.pkl'

    if os.path.isfile(PKL_FPATH):
        dataset = pickle.load(open(PKL_FPATH, 'rb'))
    else:
        fname = './data/ca1-neurons/expressions.tsv'

        counts, genes, cells = rnaseqTools.sparseload(fname, sep='\t')

        genesTable = pd.read_csv('./data/mouse_VISp_2018-06-14_genes-rows.csv')
        ids = genesTable['gene_entrez_id'].tolist()
        symbols = genesTable['gene_symbol'].tolist()
        id2symbol = dict(zip(ids, symbols))
        genes = np.array([id2symbol[g] for g in genes])

        clusterInfo = pd.read_csv('data/tasic-sample_heatmap_plot_data.csv')
        goodCells = clusterInfo['sample_name'].values
        ids = clusterInfo['cluster_id'].values
        labels = clusterInfo['cluster_label'].values
        colors = clusterInfo['cluster_color'].values

        clusterNames = np.array([labels[ids==i+1][0] for i in range(np.max(ids))])
        clusterColors = np.array([colors[ids==i+1][0] for i in range(np.max(ids))])
        clusters = np.copy(ids)

        indices = np.array([np.where(cells==c)[0][0] for c in goodCells])
        counts = counts[indices, :]

        areas = (indices < datasets[0][2].size).astype(int)
        clusters -= 1

        markers = ['Snap25','Gad1','Slc17a7','Pvalb', 'Sst', 'Vip', 'Aqp4', 
           'Mog', 'Itgam', 'Pdgfra', 'Flt1', 'Bgn', 'Rorb', 'Foxp2']
        markerSubset = rnaseqTools.geneSelection(
            counts, n=3000, threshold=32, 
            markers=markers, genes=genes)

        dataset = {'counts': counts, 'genes': genes, 'clusters': clusters, 'areas': areas, 
                'clusterColors': clusterColors, 'clusterNames': clusterNames,
                'markerSubset': markerSubset}
        
        pickle.dump(dataset, open(PKL_FPATH, 'wb'))
    
    logging.info('Read data size with size: {}'.format(dataset['counts'].shape))
    logging.info('Areas: {} {}'.format(np.sum(dataset['areas']==0), np.sum(dataset['areas']==1)))
    logging.info('Unique clusters: {}'.format(np.unique(dataset['clusters']).size))

    return dataset


def get_pollen_dataset():
    PKL_FPATH = './data/pollen.pkl'

    if os.path.isfile(PKL_FPATH):
        dataset = pickle.load(open(PKL_FPATH, 'rb'))
    else:
        fname = './data/pollen/pollen.csv'
        counts = pd.read_csv(fname, header=None).transpose()

        ids = []
        with open('data/pollen/pollen_labels.txt', 'r') as fp:
            lines = fp.readlines()
            ids = np.array(list(map(int, lines)))

        colors = np.array(['#ff80ed', '#065535', '#133337', '#ffc0cb', '#008080', '#ffd700', '#00ffff',
                '#ff7373', '#ffa500', '#0000ff', '#003366', '#00ff00', '#800000', '#800080'])

        clusterColors = colors
        clusters = np.copy(ids)

        clusters -= 1

        dataset = {'counts': counts, 'genes': None, 'clusters': clusters, 'areas': None, 
                'clusterColors': clusterColors, 'clusterNames': None,
                'markerSubset': None}
        
        pickle.dump(dataset, open(PKL_FPATH, 'wb'))
    
    logging.info('Read data size with size: {}'.format(dataset['counts'].shape))
    logging.info('Unique clusters: {}'.format(np.unique(dataset['clusters']).size))

    return dataset


def get_dataset(name):
    if name == 'mouse-exon':
        return get_mouse_exon_dataset()
    elif name == 'pollen':
        return get_pollen_dataset()
