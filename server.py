#!/usr/bin/env python3
#
# Copyright 2017 Zegami Ltd

"""Preprocess images using Keras pre-trained models."""



from flask import Flask
from flask import request, jsonify, \
    send_from_directory

import base64

from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io

import sys
import pandas
import argparse
import numpy as np
from pprint import pprint

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.distances import EuclideanDistance
from nearpy.distances import CosineDistance
import numpy
import scipy

from nearpy.distances.distance import Distance

import tensorflow as tf
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
from scipy.spatial import distance

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

def load_model():
    global model
    model = named_model('ResNet50')
    global graph
    graph = tf.get_default_graph()

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

        with graph.as_default():
            # extract the features
            features = model.predict(img)[0]
            # convert from Numpy to a list of values
            features_arr = np.char.mod('%f', features)

            return features_arr
    except Exception as ex:
        # skip all exceptions for now
        print(ex)
        pass
    return None

class HammingDistance(Distance):
    """  Uses 1-cos(angle(x,y)) as distance measure. """

    def distance(self, x, y):
        """
        Computes distance measure between vectors x and y. Returns float.
        """

        return distance.hamming(x, y)

def load_search_engine():
    global engine

    # read in the data file
    data = pandas.read_csv(os.path.join('data', 'features.tsv'), sep='\t')
    data_objects = pandas.read_csv(os.path.join('data', 'object_features.tsv'), sep='\t')

    # Create a random binary hash with 10 bits
    rbp = RandomBinaryProjections('rbp', 10)

    # Create engine with pipeline configuration
    engine = Engine(len(data['features'][0].split(',')), lshashes=[rbp], distance=EuclideanDistance())

    # indexing
    for i in range(0, len(data)):
        engine.store_vector(np.asarray(data['features'][i].split(',')).astype('float64'), data['filename'][i].replace('images\\\\', '').replace('images\\', '').replace('images/', ''))
    
    for i in range(0, len(data_objects)):
        engine.store_vector(np.asarray(data_objects['features'][i].split(',')).astype('float64'), data_objects['filename'][i].replace('images\\\\', '').replace('images\\', '').replace('images/', ''))
    
    return engine


def query_index(v):
    global engine
    # query a vector q_vec
    return engine.neighbours(v)


IMGS_PATH = './images/'
app = Flask(__name__, static_url_path='')

load_model()
load_search_engine()

class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route("/hello", methods=['GET'])
def hello():
    return "Hello, world!"


@app.route("/search", methods=['GET', 'POST'])
def search():
    """get tags corresponding to a image"""
    if not 'img' in request.files:
        raise InvalidUsage('parameter "img" is missing', status_code=410)
    try:
        image = flask.request.files["img"].read()
        image = Image.open(io.BytesIO(image))
    except:
        raise InvalidUsage('Invalid "img" param, must be a blob string',
                           status_code=410)

    features = np.asarray(get_feature(image)).astype('float64')

    print("features")
    pprint(features)

    results = query_index(features)

    rs = []
    
    for res in results:
        src = res[1]
        dist = res[2]

        im_src = '/img/{}'.format(src)
        rs.append({
          'img_src': im_src,
          'distance': dist
        })

    out = {}
    out['hits'] = rs
    return jsonify(out)


@app.route('/static/<path:path>')
def send_static_files(path):
    "static files"
    return send_from_directory('static_data', path)


@app.route('/img/<path:path>')
def send_image(path):
    "static files"
    return send_from_directory(IMGS_PATH, path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
