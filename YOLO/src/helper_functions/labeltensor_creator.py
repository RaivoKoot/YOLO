import tensorflow as tf

from YOLO.src.classes.LabelTensor import YOLOLabelTensor
from YOLO.src.classes.ObjectAnnotations import YOLOObjectAnnotation
from YOLO.src.helper_functions.annotation_type_conversions import \
                                            global_boundingbox_togridcell

import YOLO.GlobalValues as GlobalValues
GlobalValues.initialize()
GRID_LENGTH = GlobalValues.S
BOXES = GlobalValues.B
CLASSES = GlobalValues.CLASSES

def labeltensor_from_bboxes(bboxes):
    '''
    Returns an S*S*(B*BOX_OUTPUTS + C)
    Tensorflow Tensor representing the annotations.
    '''
    label_tensor = YOLOLabelTensor(GRID_LENGTH, BOXES, CLASSES)

    object_annotations = []

    for box in bboxes:
        x_center = float(box[0])
        y_center = float(box[1])
        width = float(box[2])
        height = float(box[3])
        class_id = int(box[4])

        x_center, y_center, grid_row, grid_column = \
            global_boundingbox_togridcell(x_center, y_center, GRID_LENGTH)

        annotation = YOLOObjectAnnotation(grid_row, grid_column,
                                          x_center, y_center,
                                          class_id,
                                          width, height)
        annotation.find_matching_anchor_boxes()

        object_annotations.append(annotation)

    label_tensor.add_objects(object_annotations)
    return label_tensor.to_tensor()
