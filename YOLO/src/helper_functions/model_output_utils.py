"""
Author: Raivo Koot
Date: 22 April 2020

Functions to help make sense of and displaying predicted
YOLO output tensors.
"""
import tensorflow as tf
from bounding_box import bounding_box as bbox_drawer
from YOLO.src.helper_functions.annotation_type_conversions import \
            gridcell_boundingbox_toglobal, yolo_to_voc, decode_box

import YOLO.GlobalValues as GlobalValues
GlobalValues.initialize()

def extract_boxes(label_tensor, confidence_threshold, B, C, S, image_size=(224,224)):
    '''
    Extracts all the bounding boxes from the YOLO label tensor
    with a higher confidence than confidence_threshold.
    Returns a (boxes,5) shape tensor filled with confidence,
    x, y, width and height.

    params:
    label_tensor - A YOLO label tensor.
    confidence_threshold - The minimum confidence a bounding
                           box must have to be considered
                           a prediction.
    B - Number of bounding box predictions each grid cell
        in the output tensor has.
    C - Number of classes.
    '''

    boxes = []
    for row in range(S):
        for column in range(S):
            for anchor_index in range(B):

                confidence = label_tensor[row, column, anchor_index]
                if confidence < confidence_threshold:
                    continue

                box_xy = label_tensor[row, column, B + anchor_index*2:B+anchor_index*2 + 2]
                box_wh = label_tensor[row, column, B + B*2 + anchor_index*2:B + B*2 + anchor_index*2 + 2]

                x = box_xy[0]
                y = box_xy[1]
                encoded_w = box_wh[0]
                encoded_h = box_wh[1]

                anchor_w = GlobalValues.ANCHOR_BOXES[anchor_index][0]
                anchor_h = GlobalValues.ANCHOR_BOXES[anchor_index][1]

                x, y = gridcell_boundingbox_toglobal(x, y, row, column, S)
                w, h = decode_box(encoded_w, encoded_h, anchor_w, anchor_h)

                x_min, y_min, x_max, y_max = yolo_to_voc(x, y, w, h)

                x_min *= image_size[0]
                y_min *= image_size[0]
                x_max *= image_size[0]
                y_max *= image_size[0]

                if x_min < 0.:
                    x_min = 0.
                if y_min < 0.:
                    y_min = 0.
                if x_max > image_size[0]:
                    x_max = image_size[0]
                if y_max > image_size[0]:
                    y_max = image_size[0]

                class_id = tf.argmax(label_tensor[row, column, B*5 + C*anchor_index: \
                                                               B*5 + C*anchor_index + C])

                boxes.append([float(confidence), int(x_min), int(y_min),
                                                 int(x_max), int(y_max),
                                                 int(class_id)])

    return boxes

def draw_boundingboxes_on_image(image, boxes):
    '''
    Given an image
    '''
    for box in boxes:
        confidence = box[0]
        x_min = box[1]
        y_min = box[2]
        x_max = box[3]
        y_max = box[4]
        class_id = box[5]
        class_ = GlobalValues.CLASS_NAMES[class_id]
        label = '{:03.2f} - {}'.format(confidence, class_)

        bbox_drawer.add(image, x_min, y_min, x_max, y_max, label, 'red')
