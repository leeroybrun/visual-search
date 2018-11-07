#import numpy as np
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib._png import read_png
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
#from tsne import bh_sne
import argparse
import csv
import sys
import pandas
from pprint import pprint

def main(argv):
    parser = argparse.ArgumentParser(prog='VIZ')
    parser.add_argument('source', help='path to the source metadata file')
    args = parser.parse_args(argv[1:])

    # read in the data file
    data = pandas.read_csv(args.source, sep='\t')

    # load up data
    vis_x = data['x'].tolist()
    vis_y = data['y'].tolist()
    img_data = data['filename'].tolist()

    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for x, y, filepath in zip(vis_x, vis_y, img_data):
        im = plt.imread(filepath)

        bb = Bbox.from_bounds(x,y,1,1)  
        bb2 = TransformedBbox(bb,ax.transData)
        bbox_image = BboxImage(bb2,
                            norm = None,
                            origin=None,
                            clip_on=False)

        bbox_image.set_data(im)
        ax.add_artist(bbox_image)

    #plt.scatter(vis_x, vis_y, marker='s', c=vis_y)
    #plt.colorbar(ticks=range(10))
    #plt.clim(-0.5, 9.5)

    # Set the x and y limits
    ax.set_ylim(-50,50)
    ax.set_xlim(-50,50)


    plt.show()

def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

if __name__ == '__main__':
    sys.exit(main(sys.argv))