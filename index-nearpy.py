import sys
import pandas
import argparse
import numpy as np
from pprint import pprint

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.distances import EuclideanDistance
from nearpy.distances import CosineDistance

def main(argv):
    parser = argparse.ArgumentParser(prog='INDEX')
    parser.add_argument('source', help='path to the source metadata file')
    parser.add_argument(
        '--hash-size',
        help='Hash size.',
        type=int,
        default=10)
    parser.add_argument(
        '--num-tables',
        help='Number of tables.',
        type=int,
        default=5)
    parser.add_argument(
        '--query-index',
        help='Index to use for query.',
        type=int,
        default=0)

    args = parser.parse_args(argv[1:])

    # read in the data file
    data = pandas.read_csv(args.source, sep='\t')

    # Create a random binary hash with 10 bits
    rbp = RandomBinaryProjections('rbp', 10)

    # Create engine with pipeline configuration
    engine = Engine(len(data['features'][0].split(',')), lshashes=[rbp], distance=EuclideanDistance())

    # indexing
    for i in range(0, len(data)):
        engine.store_vector(np.asarray(data['features'][i].split(',')).astype('float64'), data['filename'][i])
    
    

    # query a vector q_vec
    response = engine.neighbours(np.asarray(data['features'][args.query_index].split(',')).astype('float64'))

    pprint(response)


if __name__ == '__main__':
    sys.exit(main(sys.argv))