from data import get_dataset
from dim_reduce import get_dim_reduction
from analysis_tools import preprocess
from metrics import knn_score, knc_score, cpd_score

from argparse import ArgumentParser
import logging
import os.path
import time


def main():
    parser = ArgumentParser()
    parser.add_argument('--log', choices=['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', 
        type=str.upper, help='logging level')
    parser.add_argument('-d', '--dataset', type=str, choices=['synthetic', 'mouse-exon'])
    parser.add_argument('-a', '--algorithm', type=str, default='pca', choices=['pca', 'tsne', 'umap', 'hsne', 'densne', 'denmap'])
    parser.add_argument('-o', '--output', help='output csv location')
    args = parser.parse_args()

    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(args.log))
    logging.basicConfig(format='%(asctime)s [%(levelname)s] -- %(message)s', 
        filename='log.txt', filemode='w', level=numeric_level)

    logging.info('Reading in dataset...')
    dataset = get_dataset(args.dataset)

    logging.info('Preprocessing...')
    X = preprocess(dataset['counts'], normalize=True, subsets=dataset['markerSubset'])

    logging.info('Doing dimensionality reduction...')
    start = time.time()
    reduced = get_dim_reduction(X, algorithm=args.algorithm)
    duration = time.time() - start
    logging.info('Finished dimensionality reduction in {} seconds.'.format(duration))

    logging.info('Calculating metrics...')
    knn = knn_score(X, reduced, k=10)
    knc = knc_score(X, reduced, dataset['clusters'], k=10)
    cpd = cpd_score(X, reduced, subset_size=1000)

    header = 'dataset,algorithm,duration,knn_score,knc_score,cpd_score'
    cols = [args.dataset, args.algorithm, duration, knn, knc, cpd]
    if (args.output is None) or (args.output == '-'):
        print(','.join(map(str, cols)))
    else:
        mode = 'a'
        if not os.path.isfile(arg.output):
            mode = 'w'

        with open(args.output, mode) as fp:
            if mode == 'w':
                fp.write(header)
            fp.write(','.join(cols))


if __name__ == '__main__':
    main()