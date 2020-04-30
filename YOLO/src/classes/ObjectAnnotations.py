"""
Author: Raivo Koot
Date: 16 April 2020

Classes to encapsulate the information of an object annotation in YOLO
and VOC format.
"""

import tensorflow as tf
import numpy as np
from YOLO.src.helper_functions.iou import iou

import YOLO.GlobalValues as GlobalValues
GlobalValues.initialize()

class YOLOObjectAnnotation():
    """
    Holds information about an object present in an image.

    cell_row, cell_column: Indices as to where in the YOLO grid this item lies.
    x, y: A value between 0 and 1 referring to the center coordinate of the object,
          relative to the grid cell.
    width, height: A value between 0 and 1 referring to the height and the width
                   of the object, relative to the entire image.
    class_label: The index of the class of the object.
    """

    def __init__(self, cell_row, cell_column, x, y, class_label,
                                        width=None, height=None,
                                        encoded_width=None, encoded_height=None):
        self.cell_row = int(cell_row)
        self.cell_column = int(cell_column)
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.class_label = class_label
        self.encoded_width = encoded_width
        self.encoded_height = encoded_height

    def find_matching_anchor_boxes(self):
        """
        Finds those anchor boxes which have the highest IoU
        with self. Creates a list of anchor box indices from
        best to worst.
        """
        box = [0,0,self.width, self.height]
        ious = []

        for anchor_dimensions in GlobalValues.ANCHOR_BOXES:
            anchor_box = [0,0,anchor_dimensions[0], anchor_dimensions[1]]
            ious.append(iou(box, anchor_box))

        # Sort them so that the indices of the anchor boxes with
        # the best IoUs come first
        self.preferred_anchor_indices = np.argsort([-x for x in ious])

class VOCAnnotation():
    """
    Holds information about an object present in an image.
    PASCAL VOC style object annotation
    class_label: The index of the class of the object.
    """

    def __init__(self, xmin, ymin, xmax, ymax, class_label):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.class_label = class_label
