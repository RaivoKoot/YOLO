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

def add_preprocessing_definitions(dataset, batch_size=32, num_threads=10, augmentation=False):
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

    def augment(image_batch):

        image_batch = tf.image.random_hue(image_batch, 0.08)
        image_batch = tf.image.random_saturation(image_batch, 0.6, 1.6)
        image_batch = tf.image.random_brightness(image_batch, 0.05)
        image_batch = tf.image.random_contrast(image_batch, 0.7, 1.3)
        image_batch = tf.clip_by_value(image_batch, -1., 1.)
        return image_batch

    dataset = dataset.map(lambda x,y: (scale_to_255(x), y),
                            num_parallel_calls=num_threads)

    dataset = dataset.map(lambda x,y: (resize(x), y),
                            num_parallel_calls=num_threads)

    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x,y: (feature_extractor__preprocess(x), y),
                            num_parallel_calls=num_threads)

    if augmentation:
        dataset = dataset.map(lambda x,y: (augment(x), y),
                                num_parallel_calls=num_threads)

    dataset = dataset.prefetch(1)

    return dataset

def get_dataset(filenames, batch_size=32, num_threads=10, augmentation=False):
    '''
    Reads data from TFRecord files and returns a tf.data.Dataset object
    after applying interleaving, shuffling, preprocessing, and batching.
    '''
    filepaths = [str(TFRECORD_PATH / filename) for filename in filenames]
    dataset = load_tfrecords(filepaths, num_threads)
    dataset = add_preprocessing_definitions(dataset, batch_size, num_threads, augmentation=augmentation)
    return dataset
