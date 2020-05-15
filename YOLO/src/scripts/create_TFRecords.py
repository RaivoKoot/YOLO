"""
Author: Raivo Koot
Date: 20 April 2020

Script that converts data to TFRecord files.

The data that is expected is:
1) In the data directory a txt file containing one file name per line.
2) All jpg images with those file names in data/images_and_annotations.
3) For each image, a txt annotation file with the same name inside
   of data/images_and_annotations/yolo.

   Each annotation file contains
   one line of 5 values per object. The first value is the class ID,
   the second and third value the x and y center of the object in
   relation to the whole image, the last two values the width and
   height of the object in relation to the whole image. The last
   four values are all between 0 and 1.

   These yolo annotation files are
   generated from PASCAL VOC annotation files using the
   voc_to_yolo.py script in the data directory.

To run this script, specify IMAGE_NAMES and OUTPUT_FILE_NAME
below. IMAGE_NAMES is the name of the txt file that contains
your image names. OUTPUT_FILE_NAME is the name the output
TFRecord files should have.
"""

from pathlib import Path
from contextlib import ExitStack
import tensorflow as tf
from PIL import Image
import numpy as np

from YOLO.src.classes.LabelTensor import YOLOLabelTensor
from YOLO.src.classes.ObjectAnnotations import YOLOObjectAnnotation
from YOLO.src.helper_functions.annotation_type_conversions import \
                                            global_boundingbox_togridcell

import YOLO.GlobalValues as GlobalValues
GlobalValues.initialize()

# Imports
BytesList = tf.train.BytesList
FloatList = tf.train.FloatList
Int64List = tf.train.Int64List
Feature = tf.train.Feature
Features = tf.train.Features
Example = tf.train.Example

IMAGE_NAMES = GlobalValues.IMAGE_NAMES_FILE
OUTPUT_FILE_NAME = GlobalValues.TFRECORD_OUTPUT_FILENAME

ROOTDATA_PATH = Path('.') / 'YOLO' / 'data'
DATA_PATH = ROOTDATA_PATH / GlobalValues.DATA_FOLDER_NAME
ANNOTATION_PATH = DATA_PATH / 'yolo'

OUTPUT_PATH = ROOTDATA_PATH / 'TFRecords'

GRID_LENGTH = GlobalValues.S
BOXES = GlobalValues.B
CLASSES = GlobalValues.CLASSES

def create_annotation_tensor(filename):
    '''
    Given a filename of a yolo annotation txt file,
    loads the annotation file and returns a B*5 Tensor
    filled with the annotations in VOC format.
    '''
    filename = filename +'.txt'
    file = ANNOTATION_PATH / filename

    objects = open(file, 'r').readlines()

    object_annotations = []

    for object in objects:
        values = object.split()
        class_id = int(values[0])
        x_center = float(values[1])
        y_center = float(values[2])
        width = float(values[3])
        height = float(values[4])

        annotation = [x_center, y_center, width, height, class_id]

        object_annotations.append(annotation)

    annotations = tf.constant(object_annotations)

    return annotations

def create_labeltensor(filename):
    '''
    Given a filename of a yolo annotation txt file,
    loads the annotation file and creates/returns an S*S*(B*BOX_OUTPUTS + C)
    Tensorflow Tensor representing the annotations.
    '''
    filename = filename +'.txt'
    file = ANNOTATION_PATH / filename
    label_tensor = YOLOLabelTensor(GRID_LENGTH, BOXES, CLASSES)

    objects = open(file, 'r').readlines()

    object_annotations = []

    for object in objects:
        values = object.split()
        class_id = int(values[0])
        x_center = float(values[1])
        y_center = float(values[2])
        width = float(values[3])
        height = float(values[4])

        x_center, y_center, grid_row, grid_column = \
            global_boundingbox_togridcell(x_center, y_center, GRID_LENGTH)

        annotation = YOLOObjectAnnotation(grid_row, grid_column,
                                          x_center, y_center,
                                          class_id,
                                          width, height)
        annotation.find_matching_anchor_boxes()

        object_annotations.append(annotation)

    if len(object_annotations) == 0:
        return None

    label_tensor.add_objects(object_annotations)
    return label_tensor.to_tensor()

def load_instance(filename):
    '''
    Loads an image. Loads and preprocesses its annotations into a tensor.
    Returns both.

    params:
    filename: Filename of an image and it's annotation file.
    '''
    file = DATA_PATH / (filename+'.jpg')
    image = tf.constant(np.asarray(tf.keras.preprocessing.image.load_img(file)))
    #label_tensor = create_labeltensor(filename)
    annotations = create_annotation_tensor(filename)

    return image, annotations #label_tensor

def create_example(image, label):
    '''
    Creates a TFRecord compatible Protoc Example
    from two tensors
    '''
    image_data = tf.io.encode_jpeg(image)
    label_data = tf.io.serialize_tensor(label)

    return Example(
        features=Features(
            feature={
                "image": Feature(bytes_list=BytesList(value=[image_data.numpy()])),
                "label": Feature(bytes_list=BytesList(value=[label_data.numpy()])),
            }))

def write_tfrecords(output_path, output_filename, filenames, n_shards=10):
    '''
    Edited by Raivo Koot
    Source: [https://github.com/ageron/handson-ml2/blob/master/13_loading_and_preprocessing_data.ipynb]
    Author: Aurelion Geron

    Takes a list of image file names,
    converts their object annotations to a YOLO label tensor,
    and saves each image and it's label in one of ten TFRecord files.
    '''
    paths = [output_path / "{}-{:03d}-of-{:03d}.tfrecord".format(output_filename, index, n_shards)
             for index in range(n_shards)]

    with ExitStack() as stack:
        writers = [stack.enter_context(tf.io.TFRecordWriter(str(path)))
                   for path in paths]


        for index, filename in enumerate(filenames):

            image, label = load_instance(filename)
            if label is None:
                continue

            example = create_example(image, label)

            shard = index % n_shards
            writers[shard].write(example.SerializeToString())

            if index % 100 == 0:
                print(str(index) + '/' + str(len(filenames)))

    return paths

if __name__ == "__main__":
    image_names = open(ROOTDATA_PATH / IMAGE_NAMES, 'r').readlines()
    image_names = [image_name.strip() for image_name in image_names]

    write_tfrecords(OUTPUT_PATH, OUTPUT_FILE_NAME, image_names, n_shards=10)
