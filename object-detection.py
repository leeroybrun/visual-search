#!/usr/bin/env python
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[4]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from pprint import pprint

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append(os.path.join("models", "research"))
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


# ## Env setup

# In[5]:


# This is needed to display the images.
#get_ipython().run_line_magic('matplotlib', 'inline')


# ## Object detection imports
# Here are the imports from the object detection module.

# In[6]:


from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

import base64

from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import io

import sys
import pandas
import argparse
import numpy as np
from pprint import pprint

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

load_model()

# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[7]:


# What model to download.
MODEL_NAME = 'faster_rcnn_resnet101_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
MODELS_PATH =  os.path.join('data', 'models')
MODEL_FILE_PATH_TGZ = os.path.join(MODELS_PATH, MODEL_FILE)
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODELS_PATH +'/'+ MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('models', 'research', 'object_detection', 'data', 'mscoco_label_map.pbtxt')


# ## Download Model

# In[8]:
if not os.path.isfile(PATH_TO_FROZEN_GRAPH):
  print('Downloading model {}'.format(MODEL_NAME))
  opener = urllib.request.URLopener()
  opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE_PATH_TGZ)
  tar_file = tarfile.open(MODEL_FILE_PATH_TGZ)
  for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
      tar_file.extract(file, MODELS_PATH)


# ## Load a (frozen) Tensorflow model into memory.

# In[9]:


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[10]:


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# ## Helper code

# In[11]:


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[ ]:


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_RES_DIR = os.path.join('images', 'obj_result')
PATH_TO_TEST_IMAGES_CROP_RES_DIR = os.path.join('images', 'obj_result', 'cropped')

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


with open(os.path.join('data', 'object_features.tsv'), 'w') as csv_output:
  w = csv.DictWriter(csv_output, fieldnames=['filename', 'object_class', 'score', 'ymin', 'xmin', 'ymax', 'xmax', 'features'], delimiter='\t', lineterminator='\n')
  w.writeheader()

  for image_path in glob.glob('images/*.jpg'):
    if os.path.isfile(image_path):
      image_path_splitted = image_path.split(os.sep)
      image_path_splitted = image_path_splitted[len(image_path_splitted)-1].split('.')
      image_path_filename = image_path_splitted[0]

      image_res_path = os.path.join(PATH_TO_TEST_IMAGES_RES_DIR, '{}.jpg'.format(image_path_filename))

      print('running on {0} ({1})'.format(image_path, image_path_filename))
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      print("image loaded and expanded")
      # Actual detection.
      output_dict = run_inference_for_single_image(image_np, detection_graph)
      print("runned inference")
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=8)
      print("showed viz")
      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(image_np)
      plt.savefig(image_res_path)
      plt.close()

      min_score_thresh = 0.5
      boxes = output_dict['detection_boxes']
      scores = output_dict['detection_scores']
      classes = output_dict['detection_classes']
      instance_masks = output_dict.get('detection_masks')
      im_width, im_height = image.size

      labels_whitelist = ['NA', 'desk','dining table','mirror','bed', 'bench','couch','chair','bowl','cup','plate', 'pillow', 'blanket', 'mat', 'rug', 'furniture', 'furniture-other', 'light', 'cupboard', 'shelf', 'table', 'desk', 'carpet']

      for j in range(boxes.shape[0]):
        if scores is None or scores[j] > min_score_thresh:
          box = tuple(boxes[j].tolist())
          
          ymin, xmin, ymax, xmax = box

          ymin_px = ymin * im_height
          ymax_px = ymax * im_height
          xmin_px = xmin * im_width
          xmax_px = xmax * im_width

          display_str = ''
          if classes[j] in category_index.keys():
            class_name = str(category_index[classes[j]]['name'])
          else:
            class_name = 'NA'
          display_str = str(class_name)

          if class_name in labels_whitelist:
            display_str = '{}_{}'.format(display_str, int(100*scores[j]))

            print('cropping box {0} ({1}, {2}, {3}, {4})'.format(display_str, xmin_px, ymin_px, xmax_px, ymax_px))

            cropped_img = image.crop((xmin_px, ymin_px, xmax_px, ymax_px))
            with tf.gfile.Open(os.path.join(PATH_TO_TEST_IMAGES_CROP_RES_DIR, '{0}_{1}_{2}.jpg'.format(image_path_filename, display_str, j)), 'w') as fid:
              cropped_img.save(fid, 'JPEG')
            
            features = get_feature(cropped_img)
            w.writerow({
              "filename": image_path, 
              "object_class": class_name,
              "score": scores[j],
              "ymin": ymin,
              "xmin": xmin,
              "ymax": ymax,
              "xmax": xmax,
              "features": ','.join(features)
            })

            cropped_img.close()

      image.close()