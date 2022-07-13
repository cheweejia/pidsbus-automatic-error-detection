import os
import tensorflow as tf
import cv2 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
matplotlib.use( 'tkagg' )

# Modify the checkpoint and paths according to your own folder configuration
CUSTOM_MODEL_NAME = 'bus_detection_model'
LABEL_MAP_NAME = 'label_map.pbtxt'
CHECKPOINT = 'ckpt-4'

paths = {
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
}

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

def build_model():
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], CHECKPOINT)).expect_partial()
    return detection_model

# Helper function for detection
def detect_fn(model, image):
    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)
    return detections

# Detect bus and EDS from image
def detect_bus(image):
    # Build model
    detection_model = build_model()

    # Obtain image from path
    category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
    IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', image)
    img = cv2.imread(IMAGE_PATH)
    image_np = np.array(img)

    # Convert image to tensor
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    
    # Run detection
    detections = detect_fn(detection_model, input_tensor)

    # Extract information from detection
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    # Visualise image with bounding boxes
    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)

    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    plt.show()
    return detections

# Run detection here
detect_bus('13.png')