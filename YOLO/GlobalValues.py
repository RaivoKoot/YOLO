import tensorflow as tf

def initialize():
    global S # YOLO Grid size S*S
    global B # Number of Box predictions per YOLO Grid
    global OUTPUTS_PER_BOX # Number of values associated with each bounding box prediction.
    global CLASSES # Number of classes
    global CLASS_NAMES
    global ANCHOR_BOXES

    # Folder inside of the 'data' folder where your images and
    # annotations lie.
    global DATA_FOLDER_NAME
    global CLASS_IDS_TO_SKIP

    # Names of the txt files in the data directory containing
    # names of the images that you want to convert to TFRecords
    global IMAGE_NAMES_FILE
    global TFRECORD_OUTPUT_FILENAME


    # Select the weights for different Loss terms
    global SCALE_FACTOR_BOUNDINGBOX
    global SCALE_FACTOR_NOOBJECT_CONFIDENCE
    global SCALE_FACTOR_OBJECT_CONFIDENCE

    # Select the base model you want to use
    global FEATURE_EXTRACTOR
    global FEATURE_EXTRACTOR_PREPROCESSOR
    global FEATURE_EXTRACTOR_INPUT_SHAPE




    S = 7
    B = 8
    CLASS_NAMES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    CLASSES = len(CLASS_NAMES)

    # Folder inside of the 'data' folder where your images and
    # annotations lie.
    DATA_FOLDER_NAME = 'files'

    # Names of the txt files in the data directory containing
    # names of the images that you want to convert to TFRecords
    IMAGE_NAMES_FILE = 'val_image_names_abundant.txt'
    TFRECORD_OUTPUT_FILENAME = 'val'


    # Select the weights for different Loss terms
    SCALE_FACTOR_BOUNDINGBOX = 5.
    SCALE_FACTOR_NOOBJECT_CONFIDENCE = 0.5
    SCALE_FACTOR_OBJECT_CONFIDENCE = 2.

    # Select the base model you want to use
    FEATURE_EXTRACTOR = tf.keras.applications.mobilenet_v2.MobileNetV2
    FEATURE_EXTRACTOR_PREPROCESSOR = tf.keras.applications.mobilenet_v2.preprocess_input
    FEATURE_EXTRACTOR_INPUT_SHAPE = (224,224,3)

    # Taken from YOLOv3 Paper
    ANCHOR_BOXES = [(10,13), (16,30), (33,23), (30,61), (62,45),
                    (59,119), (116, 90), (156, 198)]#, (373, 326)]

    ANCHOR_BOXES = [(width/FEATURE_EXTRACTOR_INPUT_SHAPE[0],
                     height/FEATURE_EXTRACTOR_INPUT_SHAPE[0]) for width, height
                     in ANCHOR_BOXES]
