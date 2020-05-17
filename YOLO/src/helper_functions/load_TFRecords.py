"""
Author: Raivo Koot
Date: 20 April 2020

Functions to serve as the input pipeline between the TFRecord files
and the Neural Network input. Reads and interleaves the TFRecord
shard files into a Tensorflow Dataset and defines preprocessing
operations on the dataset.
"""
import tensorflow as tf
from pathlib import Path
import cv2
import numpy as np
from YOLO.src.helper_functions.labeltensor_creator import \
                                            labeltensor_from_bboxes

from albumentations import OneOf, HorizontalFlip, CLAHE, IAASharpen, IAAEmboss, \
    RandomBrightnessContrast, BboxParams, Compose, ShiftScaleRotate, Resize, \
    HueSaturationValue, ChannelShuffle, RGBShift

import YOLO.GlobalValues as GlobalValues
GlobalValues.initialize()

BytesList = tf.train.BytesList
FloatList = tf.train.FloatList
Int64List = tf.train.Int64List
Feature = tf.train.Feature
Features = tf.train.Features
Example = tf.train.Example

# TFRecord files must be inside this directory
TFRECORD_PATH = Path('.') / 'YOLO' / 'data' / 'TFRecords'

def load_tfrecords(filepaths, n_readers=10):
    '''
    Loads, interleaves and parses data from the TFRecord filenames
    into a single tf.data.Dataset.
    '''
    dataset_files = tf.data.Dataset.list_files(filepaths, seed=42)

    dataset = dataset_files.interleave(
                        lambda filepath: tf.data.TFRecordDataset(filepath),
                        cycle_length=n_readers,
                        num_parallel_calls=n_readers)

    dataset = dataset.map(parse_tfrecord, num_parallel_calls=n_readers)
    dataset = dataset.shuffle(100)

    return dataset

def parse_tfrecord(tfrecord):
    '''
    Parses a TFRecord entry of an image and a label tensor, which
    are defined in YOLO.src.scripts.create_TFRecords.py
    '''
    feature_descriptions = {
        "image": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "label": tf.io.FixedLenFeature([], tf.string, default_value="")
    }
    example = tf.io.parse_single_example(tfrecord, feature_descriptions)

    image = tf.io.decode_jpeg(example['image'], channels=3)
    label = tf.io.parse_tensor(example['label'], out_type=tf.float32)

    return image, label


def tf_augment(image, bboxes):

    def augment(image, bboxes):
        '''
        Stochastically applies horizontal flipping, cropping rotating, shifting,
        and more to an image and its bounding boxes.
        Returns the image as a tensor and the new bboxes as a B*5 Tensor.

        params:
        image: An image tensor.
        bboxes: A B*5 tensor with YOLO style annotations and class ids.
        '''
        boxes = bboxes[:,:-1].numpy()
        boxes[:,[2,3]] = boxes[:,[2,3]] - 0.00999 # Prevent Albumentations from converting to bounding boxes point out of bounds
        image_and_boxes = {'image': image.numpy(), 'bboxes': boxes,
                                    'bbox_class_ids': bboxes[:,-1].numpy().flatten()}

        output_height, output_width, _ = GlobalValues.FEATURE_EXTRACTOR_INPUT_SHAPE
        aug = Compose([HorizontalFlip(),
                       ShiftScaleRotate(p=1., scale_limit=0.2, border_mode=cv2.BORDER_CONSTANT, value=0),
                       OneOf([
                            CLAHE(clip_limit=2),
                            IAASharpen(),
                            IAAEmboss(),
                            RandomBrightnessContrast(),
                       ], p=1.0),
                       HueSaturationValue(p=1.0),
                       RGBShift(p=1.0),
                       ChannelShuffle(p=0.5),
                       Resize(output_height, output_width, always_apply=True)],
                       bbox_params=BboxParams(format='yolo', min_visibility=0.55,
                                                    label_fields=['bbox_class_ids']))
        
        try:
            result = aug(**image_and_boxes)
            augment_success = True
        except ValueError as e:
            augment_success = False
            print('\nBounding box error occured in augmentation. Augmenting step skipped.')

        if augment_success:
            image = result['image']
            bboxes = np.reshape(np.array(result['bboxes'], dtype=np.float32), (-1,4))
            labels = np.reshape(result['bbox_class_ids'], (-1,1))

            bboxes = np.concatenate((bboxes, labels), axis=1)

            return tf.constant(image), tf.constant(bboxes, dtype=tf.float32)
        else:
            shape = GlobalValues.FEATURE_EXTRACTOR_INPUT_SHAPE
            shape = (shape[0], shape[1])

            return tf.cast(tf.image.resize(image, shape), tf.uint8), bboxes

    bboxes_shape = bboxes.shape

    [image, bboxes, ] = tf.py_function(func=augment, inp=[image, bboxes], Tout=[tf.uint8, tf.float32])

    image.set_shape(GlobalValues.FEATURE_EXTRACTOR_INPUT_SHAPE)
    bboxes.set_shape(bboxes_shape)
    return image, bboxes

def tf_labeltensor_from_bboxes(bboxes):
    labeltensor = tf.py_function(func=labeltensor_from_bboxes, inp=[bboxes], Tout=tf.float32)

    labeltensor.set_shape((GlobalValues.S, GlobalValues.S, GlobalValues.B*(5+GlobalValues.CLASSES)))
    return labeltensor

def add_preprocessing_definitions(dataset, batch_size=32, num_threads=10, augmentation=True):
    '''
    Adds image preprocessing definitions onto the dataset
    and batch the dataset.
    '''

    def scale_to_255(image):
        scaled_image = tf.math.divide(
           tf.math.subtract(
              image,
              tf.math.reduce_min(image)
           ),
           tf.math.subtract(
              tf.math.reduce_max(image),
              tf.math.reduce_min(image)
           )
        ) * tf.constant(255.)
        return scaled_image

    def resize(image):
        shape = GlobalValues.FEATURE_EXTRACTOR_INPUT_SHAPE
        input_shape = (shape[0], shape[1])

        return tf.image.resize(image, input_shape)

    def feature_extractor__preprocess(image_batch):
        image_batch = tf.cast(image_batch, tf.float32)
        return GlobalValues.FEATURE_EXTRACTOR_PREPROCESSOR(image_batch)

    if augmentation:
        dataset = dataset.map(lambda x,y: tf_augment(x,y),
                                num_parallel_calls=num_threads)
    else:
        dataset = dataset.map(lambda x,y: (resize(x), y),
                                num_parallel_calls=num_threads)

    dataset = dataset.map(lambda x,y: (x, tf_labeltensor_from_bboxes(y)),
                            num_parallel_calls=num_threads)

    dataset = dataset.map(lambda x,y: (scale_to_255(x), y),
                            num_parallel_calls=num_threads)

    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x,y: (feature_extractor__preprocess(x), y),
                            num_parallel_calls=num_threads)

    dataset = dataset.prefetch(1)

    return dataset

def get_dataset(filenames, batch_size=32, num_threads=10, augmentation=True):
    '''
    Reads data from TFRecord files and returns a tf.data.Dataset object
    after applying interleaving, shuffling, preprocessing, and batching.
    '''
    filepaths = [str(TFRECORD_PATH / filename) for filename in filenames]
    dataset = load_tfrecords(filepaths, num_threads)
    dataset = add_preprocessing_definitions(dataset, batch_size, num_threads, augmentation=augmentation)
    return dataset
