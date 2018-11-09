#!/usr/bin/env python3
#
# Copyright 2017 Zegami Ltd

"""Preprocess images using Keras pre-trained models."""

import argparse
import csv
import os
import glob

from keras import applications
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import numpy as np
import pandas

from pprint import pprint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def named_model(name):
    # include_top=False removes the fully connected layer at the end/top of the network
    # This allows us to get the feature vector as opposed to a classification
    if name == 'Xception':
        return applications.xception.Xception(weights='imagenet', include_top=False, pooling='avg')

    if name == 'VGG16':
        return applications.vgg16.VGG16(weights='imagenet', include_top=False, pooling='avg')

    if name == 'VGG19':
        return applications.vgg19.VGG19(weights='imagenet', include_top=False, pooling='avg')

    if name == 'InceptionV3':
        return applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')

    if name == 'MobileNet':
        return applications.mobilenet.MobileNet(weights='imagenet', include_top=False, pooling='avg')

    return applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')


parser = argparse.ArgumentParser(prog='Feature extractor')
parser.add_argument(
    'model',
    default='ResNet50',
    nargs="?",
    type=named_model,
    help='Name of the pre-trained model to use'
)

pargs = parser.parse_args()

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # return the processed image
    return image

def get_feature(image):
    try:
        # load image setting the image size to 224 x 224
        img = prepare_image(image, target=(224, 224))

        # extract the features
        features = pargs.model.predict(img)[0]
        # convert from Numpy to a list of values
        features_arr = np.char.mod('%f', features)

        return features_arr
    except Exception as ex:
        # skip all exceptions for now
        print(ex)
        pass
    return None


def start():
    try:
        features = []

        for fname in glob.glob('images/*.jpg'):
            print('{}'.format(fname))
            try:
                if os.path.isfile(fname):
                    print('is file: {}'.format(fname))
                    try:
                        # load image setting the image size to 224 x 224
                        img = image.load_img(fname, target_size=(224, 224))
                        features = get_feature(img)

                        features.append({
                            "filename": fname, 
                            "features": ','.join(features)
                        })
                    except Exception as ex:
                        # skip all exceptions for now
                        print(ex)
                        pass
            except Exception as ex:
                # skip all exceptions for now
                print(ex)
                pass

        with open(os.path.join('data', 'features.tsv'), 'w') as output:
            w = csv.DictWriter(output, fieldnames=['filename', 'features'], delimiter='\t', lineterminator='\n')
            w.writeheader()
            w.writerows(features)

    except EnvironmentError as e:
        print(e)


if __name__ == '__main__':
    start()