from data import get_dataset
from dim_reduce import get_dim_reduction

from argparse import ArgumentParser
import logging


def main():
    parser = ArgumentParser()
    parser.add_argument('--log', choices=['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', 
        type=str.upper, help='logging level')
    parser.add_argument('-d', '--dataset', type=str, choices=['synthetic', 'mouse-exon'])
    parser.add_argument('-a', '--algorithm', type=str, default='pca', choices=['pca', 'tsne', 'umap', 'hsne', 'densne', 'denmap'])
    args = parser.parse_args()

    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(args.log))
    logging.basicConfig(format='%(asctime)s [%(levelname)s] -- %(message)s', 
        filename='log.txt', filemode='w', level=numeric_level)

    logging.info('Reading in dataset...')
    dataset = get_dataset(args.dataset)

    logging.info('Doing dimensionality reduction...')
    reduced = get_dim_reduction(dataset, algorithm=args.algorithm)



if __name__ == '__main__':
    main()