"""
Author: Raivo Koot
Date: 28 April 2020

Modified YOLO loss function implementation using the keras API.
The loss function is implemented entirely
with tensorflow operations to be auto-differentiated
as efficiently as possible.
"""

import tensorflow as tf

class YOLOConfidenceLoss(tf.keras.losses.Loss):
    def __init__(self, B, SCALE_FACTOR_NOOBJECT_CONFIDENCE=0.5, **kwargs):
        super().__init__(**kwargs)
        self.B = B
        self.SCALE_FACTOR_NOOBJECT_CONFIDENCE = tf.constant(SCALE_FACTOR_NOOBJECT_CONFIDENCE)

    def call(self, batch_y_true, batch_y_pred):
        # Extract only the confidences for each box
        labels = batch_y_true[:,:,:,:self.B]
        predictions = batch_y_pred[:,:,:,:self.B]

        # Binary cross entropy over the last axis of our batch*S*S*B Tensors
        loss = labels * -tf.math.log(predictions) + \
               (1. - labels) * -tf.math.log(1. - predictions) * self.SCALE_FACTOR_NOOBJECT_CONFIDENCE

        return tf.reduce_sum(loss)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'B': self.B,
               'SCALE_FACTOR_NOOBJECT_CONFIDENCE': self.SCALE_FACTOR_NOOBJECT_CONFIDENCE}

class YOLOBoxLoss(tf.keras.losses.Loss):
    def __init__(self, B, SCALE_FACTOR_BOUNDINGBOX=5., **kwargs):
        super().__init__(**kwargs)
        self.B = B
        self.SCALE_FACTOR_BOUNDINGBOX = tf.constant(SCALE_FACTOR_BOUNDINGBOX)

    def call(self, batch_y_true, batch_y_pred):
        # Extract only the B (x,y,w,h) labels
        labels = batch_y_true[:,:,:,self.B:(self.B+self.B*4)]
        predictions = batch_y_pred[:,:,:,self.B:(self.B+self.B*4)]

        loss = tf.math.squared_difference(labels, predictions)

        xy_loss = loss[:,:,:,:self.B*2]
        wh_loss = loss[:,:,:, self.B*2:]

        # Indicator with a 1. for boxes responsible for an object detection
        # prediction and a 0. for boxes not responsible for an object detection
        responsibility_indicator = batch_y_true[:,:,:,:self.B]
        # Convert the indicator from having a single value per box to repeating
        # this value 4 times per box for convenience so that we can multiply
        # this indicator matrix element-wise with the loss
        responsibility_indicator_reshaped = tf.repeat(responsibility_indicator, [2], axis=3)


        return tf.reduce_sum(xy_loss * responsibility_indicator_reshaped +
                             wh_loss * responsibility_indicator_reshaped) * self.SCALE_FACTOR_BOUNDINGBOX

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'B': self.B,
               'SCALE_FACTOR_NOOBJECT_CONFIDENCE': self.SCALE_FACTOR_BOUNDINGBOX}

class YOLOClassLoss(tf.keras.losses.Loss):
    def __init__(self, B, C, **kwargs):
        super().__init__(**kwargs)
        self.B = B
        self.C = C

    def call(self, batch_y_true, batch_y_pred):
        # Extract only the 20 Class labels for each box
        labels = batch_y_true[:,:,:,-self.C*self.B:]
        predictions = batch_y_pred[:,:,:,-self.C*self.B:]

        # Multiclass Cross Entropy over the classes in our batch*S*S*(B*C) tensor
        loss = labels * tf.math.log(predictions)

        # Indicator with a 1. for boxes responsible for an object detection
        # prediction and a 0. for boxes not responsible for an object detection
        responsibility_indicator = batch_y_true[:,:,:,:self.B]
        # Convert the indicator from having a single value per box to repeating
        # this value C times per box for convenience so that we can multiply
        # this indicator matrix element-wise with the loss
        responsibility_indicator_reshaped = tf.repeat(responsibility_indicator, [self.C], axis=3)

        return tf.reduce_sum(loss * responsibility_indicator_reshaped)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'B': self.B, 'C': self.C}
