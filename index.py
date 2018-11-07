from annoy import AnnoyIndex
import sys
import pandas
import argparse
import numpy as np

def main(argv):
    parser = argparse.ArgumentParser(prog='INDEX')
    parser.add_argument('source', help='path to the source metadata file')
    parser.add_argument(
        '--tree-count',
        help='Forest of (n) trees. More trees == higher query precision.',
        type=int,
        default=10)

    args = parser.parse_args(argv[1:])

    # read in the data file
    data = pandas.read_csv(args.source, sep='\t')

    ann_index = AnnoyIndex(len(data['features'][0].split(',')))
    for i in range(0, len(data)):
        ann_index.add_item(int(data['id'][i]), np.asarray(data['features'][i].split(',')).astype('float64'))

    print("[!] Constructing trees")
    ann_index.build(args.tree_count)
    print("[!] Saving the index to 'index.ann'")
    ann_index.save('index.ann')


if __name__ == '__main__':
    sys.exit(main(sys.argv))