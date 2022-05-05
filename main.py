from data import get_dataset

from argparse import ArgumentParser
import logging


def main():
    parser = ArgumentParser()
    parser.add_argument('--log', choices=['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', 
        type=str.upper, help='logging level')
    parser.add_argument('-d', '--dataset', type=str, choices=['synthetic', 'mouse-exon'])
    args = parser.parse_args()

    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(args.log))
    logging.basicConfig(format='%(asctime)s [%(levelname)s] -- %(message)s', 
        filename='log.txt', filemode='w', level=numeric_level)

    dataset = get_dataset(args.dataset)



if __name__ == '__main__':
    main()