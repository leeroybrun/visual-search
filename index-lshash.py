import sys
import pandas
import argparse
import numpy as np
from pprint import pprint
from lshash import LSHash

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

    # params
    k = args.hash_size # hash size
    L = args.num_tables  # number of tables
    d = len(data['features'][0].split(','))

    lsh = LSHash(hash_size=k, input_dim=d, num_hashtables=L)

    # indexing
    for i in range(0, len(data)):
        lsh.index(np.asarray(data['features'][i].split(',')).astype('float64'), extra_data=data['filename'][i])

    # query a vector q_vec
    response = lsh.query(np.asarray(data['features'][args.query_index].split(',')).astype('float64'))

    pprint(response)


if __name__ == '__main__':
    sys.exit(main(sys.argv))