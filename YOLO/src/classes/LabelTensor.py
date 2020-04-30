"""
Author: Raivo Koot
Date: 16 April 2020

A class to create YOLO label tensors
from ObjectAnnotationYOLO objects.
"""

import tensorflow as tf
import numpy as np
from YOLO.src.classes.ObjectAnnotations import YOLOObjectAnnotation
from YOLO.src.helper_functions.annotation_type_conversions import \
                                            encode_box, decode_box

import YOLO.GlobalValues as GlobalValues
GlobalValues.initialize()

class YOLOLabelTensor():
    """
    Helper class to create and fill in a YOLO label tensor,
    given raw object annotations.
    Represents an S*S*(BOXES*OUTPUTS_PER_BOX + CLASSES) tensor.

    NOTE: On par with the first YOLO research paper, this class
          assumes no grid cell contains more than one object.
    """

    def __init__(self, GRID_LENGTH, BOXES, CLASSES):
        self.GRID_LENGTH = GRID_LENGTH
        self.BOXES = BOXES
        self.CLASSES = CLASSES

        self.LABEL_SHAPE = (GRID_LENGTH, GRID_LENGTH,
                                BOXES*5 + BOXES*CLASSES)

        self.tensor = np.zeros(self.LABEL_SHAPE, dtype=np.float32)

    def to_tensor(self):
        '''
        Returns the numpy tensor built by this class as a
        Tensorflow tensor.
        '''
        return tf.constant(self.tensor, dtype=tf.float32)

    def add_objects(self, objects):
        '''
        Fills in this label tensor so that it
        contains the given objects.

        params:
        objects - A list of YOLOObjectAnnotation objects.
        '''
        for object in objects:
            self.__add_object(object)

    def __add_object(self, object):
        """
        Fills in the corresponding cells
        of self.tensor referring to the object.

        params:
        object - A YOLOObjectAnnotation object.
        """
        row, column = object.cell_row, object.cell_column

        for anchor_index in object.preferred_anchor_indices:

            if self.tensor[row, column, anchor_index] == 1.0:
                # This anchor box is already taken by a ground truth object
                continue

            chosen_anchor_dimension = GlobalValues.ANCHOR_BOXES[anchor_index]

            encoded_width, encoded_height = encode_box(object.width, object.height,
                                chosen_anchor_dimension[0], chosen_anchor_dimension[1])

            self.tensor[row, column, anchor_index] = 1.
            self.tensor[row, column, self.BOXES+(anchor_index*2): \
                                     self.BOXES+(anchor_index*2) + 2] = \
                                     [object.x, object.y]

            self.tensor[row, column, self.BOXES+(self.BOXES*2) + (anchor_index*2): \
                                     self.BOXES+(self.BOXES*2) + (anchor_index*2) + 2] = \
                                     [encoded_width, encoded_height]

            self.tensor[row, column, self.BOXES*5 + \
                                     self.CLASSES*anchor_index + \
                                     object.class_label] = 1.

            break
