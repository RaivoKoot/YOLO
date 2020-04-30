"""
Author: Raivo Koot
Date: 20 April 2020

Helper functions to convert between YOLO and PASCAL VOC annotation formats.
"""

import tensorflow as tf
import math

def gridcell_boundingbox_toglobal(x_center, y_center, grid_row, grid_column,
                                    grid_size_S):
    '''
    This function is the inverse of the below
    function 'global_boundingbox_togridcell'.

    Takes a single YOLO bounding box annotation, where x_center and
    y_center are the object's center coordinates in relation to
    their grid cell. Converts the x and y center coordinates
    to be in reference to the whole image.

    params:
    x_center/y_center - Value between 0 and 1 where 0
                        is the top left corner of the grid cell.
    grid_row/grid_column - Index of the label tensor's grid cell
    grid_size_S - An integer value referring to the height and
                  width that your YOLO model divides an input image
                  into.
    '''
    grid_cell_size = 1. / grid_size_S

    new_x = grid_column * grid_cell_size + grid_cell_size * x_center
    new_y = grid_row * grid_cell_size + grid_cell_size * y_center

    return x_center, y_center

def global_boundingbox_togridcell(x_center, y_center, grid_size_S):
    '''
    This function is the inverse of the above
    function 'gridcell_boundingbox_toglobal'.

    Takes a single YOLO bounding box annotation, where x_center and
    y_center are the object's center coordinates in relation to
    the global image. Converts the x and y center coordinates
    to be in reference to their responsible grid cell, returns
    these along with row and column index for the responsible
    grid cell.

    params:
    x_center/y_center - Value between 0 and 1 where 0
                        is the top left corner of their image.
    grid_size_S - An integer value referring to the height and
                  width that your YOLO model divides an input image
                  into.
    '''
    grid_cell_size = 1. / grid_size_S

    grid_column = x_center // grid_cell_size
    grid_row = y_center // grid_cell_size

    x_grid_cell = x_center % grid_cell_size
    y_grid_cell = y_center % grid_cell_size

    return x_center, y_center, grid_row, grid_column

def yolo_to_voc(x_center, y_center, width, height):
    '''
    Takes a single YOLO bounding box annotation and converts
    it to PASCAL VOC annotation format of x_min, y_min,
    x_max, y_max.

    params:
    x_center/y_center - Value between 0 and 1 where 0
                        is the top left corner of their image.
    width/height - Value between 0 and 1 referring to the width
                   and height of the bounding box in relation to
                   the whole image.
    '''
    x_min = x_center - width/2.
    x_max = x_center + width/2.
    y_min = y_center - height/2.
    y_max = y_center + height/2.

    return x_min, y_min, x_max, y_max

def encode_box(width, height, anchor_width, anchor_height):
    """
    Takes the width and height of a bounding box and an anchor
    box that this bounding box is assigned to, and converts
    the width and height to the two values the neural network
    will try to predict. The equation can be seen in the
    code below. The inverse equation can be used to convert
    from encoded width and height to actual width and height.

    params:
    width/height - Normalized value of a bounding box between 0 and 1.
    anchor_width/anchor_height - Normalized value of an anchor box between 0 and 1
    """

    encoded_width = math.log(width/anchor_width) / 2.
    encoded_height = math.log(height/anchor_height) / 2.

    return encoded_width, encoded_height

def decode_box(encoded_width, encoded_height, anchor_width, anchor_height):
    """
    Takes the encoded width and height of a bounding box
    and that of the anchor box it is assigned to, and
    converts the encoded width and height to the actual
    width and height of this bounding box.
    The equation can be seen in the code below.
    The inverse equation can be used to convert
    from width and height to encoded width and height.

    params:
    encoded_width/encoded_height - Value between -1 and 1 that the neural network
                                   predicts.
    anchor_width/anchor_height - Normalized value of an anchor box between 0 and 1.
    """

    width = anchor_width * math.exp(encoded_width * 2.)
    height = anchor_height * math.exp(encoded_height * 2.)

    return width, height
