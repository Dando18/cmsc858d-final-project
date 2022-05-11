from data import get_dataset
from dim_reduce import get_dim_reduction
from analysis_tools import preprocess
from metrics import *
from plot import plot_reduced_data

from argparse import ArgumentParser
from itertools import product
import json
import logging
import os.path
import time

import pandas as pd


def run_single_test(X, dataset, args):
    logging.info('Doing dimensionality reduction...')
    start = time.time()
    reduced = get_dim_reduction(X, algorithm=args.algorithm)
    duration = time.time() - start
    logging.info('Finished dimensionality reduction in {} seconds.'.format(duration))

    logging.info('Calculating metrics...')
    start = time.time()
    knn = knn_score(X, reduced, k=10)
    knc = knc_score(X, reduced, dataset['clusters'], k=10) if 'clusters' in dataset else None
    cpd = cpd_score(X, reduced, subset_size=1000)
    ari = ari_score(X, reduced, dataset['clusters']) if 'clusters' in dataset else None
    pds = pds_score(X, reduced, sample_size=1000)
    cs = cs_score(X, reduced, dataset['clusters']) if 'clusters' in dataset else None
    duration = time.time() - start
    logging.info('Finished calculating metrics in {} seconds.'.format(duration))

    header = 'dataset,samples,dimensions,algorithm,duration,knn_score,knc_score,cpd_score,ari_score,pds_score,cs_score'
    cols = [args.dataset, X.shape[0], X.shape[1], args.algorithm, duration, knn, knc, cpd, ari, pds, cs]
    if (args.output is None) or (args.output == '-'):
        print(','.join(map(str, cols)))
    else:
        mode = 'a'
        if not os.path.isfile(arg.output):
            mode = 'w'

        with open(args.output, mode) as fp:
            if mode == 'w':
                fp.write(header)
            fp.write(','.join(map(str, cols)))

    # plot output
    logging.info('Plotting reduced data...')
    start = time.time()
    fname = '_'.join(map(str, [args.dataset, args.algorithm])) + '.png'
    plt_title = '{} on {}'.format(args.algorithm, args.dataset)
    plot_reduced_data(reduced, dataset, fname, plt_title)
    duration = time.time() - start
    logging.info('Finished plotting data in {} seconds.'.format(duration))


def run_experiments(X, dataset, args):
    params = json.load(open(args.params, 'r'))
    test_params = list((dict(zip(params.keys(), values)) for values in product(*params.values())))
    results = []

    logging.info('Starting experiment...')
    for idx, p in enumerate(test_params):
        if idx % 10 == 0 and idx != 0:
            logging.info('Progress {}/{} = {}%'.format(idx, len(test_params), idx/len(test_params)*100))
            tmp_df = pd.DataFrame(results)
            tmp_df.to_csv('results.csv')

        start = time.time()
        reduced = get_dim_reduction(X, algorithm=args.algorithm, **p)
        duration = time.time() - start

        result = p
        result['dataset'] = args.dataset
        result['algorithm'] = args.algorithm
        result['duration'] = duration
        result['knn'] = knn_score(X, reduced, k=10)
        result['knc'] = knc_score(X, reduced, dataset['clusters'], k=10) if 'clusters' in dataset else None
        result['cpd'] = cpd_score(X, reduced, subset_size=1000)
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv('results.csv')



def main():
    parser = ArgumentParser()
    parser.add_argument('--log', choices=['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', 
        type=str.upper, help='logging level')
    parser.add_argument('-d', '--dataset', type=str, choices=['synthetic', 'pollen', 'mouse-exon'])
    parser.add_argument('-a', '--algorithm', type=str, default='pca', choices=['pca', 'tsne', 'umap', 'hsne', 'densne', 'densmap', 'scvis', 'netsne'])
    parser.add_argument('-p', '--params', type=str, help='parameters json file')
    parser.add_argument('-o', '--output', help='output csv location')
    args = parser.parse_args()

    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(args.log))
    logging.basicConfig(format='%(asctime)s [%(levelname)s] -- %(message)s', 
        filename='log.txt', filemode='w', level=numeric_level)

    logging.info('Reading in dataset...')
    start = time.time()
    dataset = get_dataset(args.dataset)
    duration = time.time() - start
    logging.info('Finished reading dataset in {} seconds.'.format(duration))

    logging.info('Preprocessing...')
    start = time.time()
    X = preprocess(dataset['counts'], normalize=True, subsets=dataset['markerSubset'])
    duration = time.time() - start
    logging.info('Finished preprocessing in {} seconds.'.format(duration))

    if args.params:
        run_experiments(X, dataset, args)
    else:
        run_single_test(X, dataset, args)


if __name__ == '__main__':
    main()