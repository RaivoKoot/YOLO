import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from YOLO.src.helper_functions.load_TFRecords import get_dataset
from YOLO.src.classes.Loss import YOLOv1Loss
from YOLO.src.classes.Metrics import YOLOClassAccuracy, YOLOConfidencePrecision, YOLOConfidenceRecall
from YOLO.src.classes.LossMetrics import ConfidenceLossMetric, CoordinateLossMetrics, ClassLossMetrics

from YOLO.src.helper_functions.model_output_utils import extract_boxes, draw_boundingboxes_on_image

import YOLO.GlobalValues as GlobalValues
GlobalValues.initialize()

S = GlobalValues.S
CLASSES = GlobalValues.CLASSES
BOXES = GlobalValues.B
OUTPUTS_PER_BOX = GlobalValues.OUTPUTS_PER_BOX

model = tf.keras.models.load_model('model_checkpoints/RESNET.epoch04-loss0.35.hdf5',
                                  custom_objects={'YOLOv1Loss': YOLOv1Loss,
                                                  'YOLOClassAccuracy': YOLOClassAccuracy,
                                                  'YOLOConfidencePrecision': YOLOConfidencePrecision,
                                                  'YOLOConfidenceRecall': YOLOConfidenceRecall,
                                                  'ConfidenceLossMetric': ConfidenceLossMetric,
                                                  'CoordinateLossMetrics': CoordinateLossMetrics,
                                                  'ClassLossMetrics': ClassLossMetrics})
def preprocess(image):
    MOBILENET_V2_INPUT_SHAPE = (224,224)
    image = tf.image.resize(image, MOBILENET_V2_INPUT_SHAPE)
    image = tf.cast(image, tf.float32)
    return tf.keras.applications.mobilenet_v2.preprocess_input(image)

def predict(image):
    #tf.expand_dims(image, 0)
    image = tf.reshape(image, (1,224,224,3))
    output = model.predict(image)[0]
    boxes = extract_boxes(output, confidence_threshold=0.2, B=BOXES, C=CLASSES, S=S)
    boxes_tf = tf.constant(boxes)

    if len(boxes) != 0:
        boxes_suppr = boxes_tf[:,1:5]
        scores = boxes_tf[:,0]
        indices = tf.image.non_max_suppression(boxes_suppr, scores=scores,
                                    max_output_size=2, iou_threshold=0.3,
                                    score_threshold=0.2)
        indices = list(indices.numpy().flatten())

        boxes = [boxes[i] for i in indices]

    image = ((image[0] + 1.) / 2.) * 255.
    image = tf.cast(image, tf.uint8).numpy()

    draw_boundingboxes_on_image(image, boxes)
    return image

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    frame = preprocess(frame)
    frame = predict(frame)
    cv2.imshow('frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
