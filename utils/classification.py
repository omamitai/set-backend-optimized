import numpy as np
import cv2
import pandas as pd
from collections import Counter
from utils.detection import detect_cards_from_image

# Define HSV color ranges as constants to avoid recreating them on each function call
# This saves memory and computation time
GREEN_LOWER = np.array([40, 50, 50])
GREEN_UPPER = np.array([80, 255, 255])
PURPLE_LOWER = np.array([120, 50, 50])
PURPLE_UPPER = np.array([160, 255, 255])
RED_LOWER1 = np.array([0, 50, 50])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 50, 50])
RED_UPPER2 = np.array([180, 255, 255])

# Predefine classification labels to avoid repeated list creation
FILL_LABELS = ['empty', 'full', 'striped']
SHAPE_LABELS = ['diamond', 'oval', 'squiggle']

def predict_color(shape_image):
    """
    Predict the dominant color in the shape image using HSV thresholds.

    Args:
        shape_image (numpy.ndarray): Image in BGR format.

    Returns:
        str: Predicted color ('green', 'purple', or 'red').
    """
    # Convert to HSV once
    hsv_image = cv2.cvtColor(shape_image, cv2.COLOR_BGR2HSV)
    
    # Create color masks more efficiently
    green_count = cv2.countNonZero(cv2.inRange(hsv_image, GREEN_LOWER, GREEN_UPPER))
    purple_count = cv2.countNonZero(cv2.inRange(hsv_image, PURPLE_LOWER, PURPLE_UPPER))
    
    # Combine the red masks to handle HSV wraparound
    red_mask1 = cv2.inRange(hsv_image, RED_LOWER1, RED_UPPER1)
    red_mask2 = cv2.inRange(hsv_image, RED_LOWER2, RED_UPPER2)
    red_count = cv2.countNonZero(cv2.bitwise_or(red_mask1, red_mask2))
    
    # Direct comparison instead of dictionary and max() operation
    if green_count > purple_count and green_count > red_count:
        return 'green'
    elif purple_count > red_count:
        return 'purple'
    else:
        return 'red'

def predict_card_features(card_image, shape_detection_model, fill_model, shape_model, box):
    """
    Predict card features (count, color, fill, shape) from a card image.

    Args:
        card_image (numpy.ndarray): Card image in BGR format.
        shape_detection_model: YOLO model for detecting shapes on the card.
        fill_model: Keras model for fill classification.
        shape_model: Keras model for shape classification.
        box (array-like): Bounding box of the card.

    Returns:
        dict: Dictionary containing predicted card features.
    """
    # Detect shapes on the card
    shape_results = shape_detection_model(card_image)
    
    # Calculate card area for filtering small detections
    card_height, card_width = card_image.shape[:2]
    card_area = card_width * card_height
    min_area_threshold = 0.03 * card_area  # Pre-compute threshold
    
    # Extract and filter boxes in one pass
    filtered_boxes = []
    for detected_box in shape_results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = detected_box.astype(int)
        shape_area = (x2 - x1) * (y2 - y1)
        if shape_area > min_area_threshold:
            filtered_boxes.append(detected_box)
    
    count = min(len(filtered_boxes), 3)
    
    # Early return if no shapes detected
    if count == 0:
        return {
            'count': 0,
            'color': 'unknown',
            'fill': 'unknown',
            'shape': 'unknown',
            'box': box
        }
    
    # Get input shapes once to avoid repeated access
    fill_input_shape = fill_model.input_shape[1:3]
    shape_input_shape = shape_model.input_shape[1:3]
    
    # Prepare data for batch prediction
    color_predictions = []
    fill_batch = np.zeros((count, *fill_input_shape, 3), dtype=np.float32)
    shape_batch = np.zeros((count, *shape_input_shape, 3), dtype=np.float32)
    
    # Process each shape
    for i, shape_box in enumerate(filtered_boxes[:count]):
        shape_box = shape_box.astype(int)
        # Extract shape region
        shape_img = card_image[shape_box[1]:shape_box[3], shape_box[0]:shape_box[2]]
        
        # Predict color (can't be batched due to HSV conversion)
        color_predictions.append(predict_color(shape_img))
        
        # Preprocess for model input - resize and normalize in one step
        fill_batch[i] = cv2.resize(shape_img, fill_input_shape) / 255.0
        shape_batch[i] = cv2.resize(shape_img, shape_input_shape) / 255.0
    
    # Batch predictions for better CPU utilization
    fill_preds = fill_model.predict(fill_batch, verbose=0)
    shape_preds = shape_model.predict(shape_batch, verbose=0)
    
    # Process predictions
    fill_predictions = [FILL_LABELS[np.argmax(pred)] for pred in fill_preds]
    shape_predictions = [SHAPE_LABELS[np.argmax(pred)] for pred in shape_preds]
    
    # Use Counter for efficient frequency counting
    color_counter = Counter(color_predictions)
    fill_counter = Counter(fill_predictions)
    shape_counter = Counter(shape_predictions)
    
    return {
        'count': count,
        'color': color_counter.most_common(1)[0][0],
        'fill': fill_counter.most_common(1)[0][0],
        'shape': shape_counter.most_common(1)[0][0],
        'box': box
    }

def classify_cards_from_board_image(board_image, card_detection_model, shape_detection_model, fill_model, shape_model):
    """
    Classify cards on a board image by detecting each card and predicting its features.

    Args:
        board_image (numpy.ndarray): The corrected board image.
        card_detection_model: YOLO model for card detection.
        shape_detection_model: YOLO model for shape detection.
        fill_model: Keras model for fill classification.
        shape_model: Keras model for shape classification.

    Returns:
        pandas.DataFrame: DataFrame containing card features.
    """
    # Detect all cards in the image
    cards = detect_cards_from_image(board_image, card_detection_model)
    
    # Process each card to extract features
    card_data = []
    for card_image, box in cards:
        features = predict_card_features(card_image, shape_detection_model, fill_model, shape_model, box)
        
        # Create dictionary with only the needed fields
        card_data.append({
            "Count": features['count'],
            "Color": features['color'],
            "Fill": features['fill'],
            "Shape": features['shape'],
            "Coordinates": f"{box[0]}, {box[1]}, {box[2]}, {box[3]}"
        })
    
    # Create DataFrame in one operation
    return pd.DataFrame(card_data)
