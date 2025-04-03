import numpy as np
import cv2
import pandas as pd
from functools import lru_cache
from utils.detection import detect_cards_from_image

# Pre-define HSV color ranges as constants to avoid repeated array creation
GREEN_RANGE = (np.array([40, 50, 50]), np.array([80, 255, 255]))
PURPLE_RANGE = (np.array([120, 50, 50]), np.array([160, 255, 255]))
RED_RANGE_1 = (np.array([0, 50, 50]), np.array([10, 255, 255]))
RED_RANGE_2 = (np.array([170, 50, 50]), np.array([180, 255, 255]))

# Define constants for model classification
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
    # Convert to HSV once to avoid redundant conversions
    hsv_image = cv2.cvtColor(shape_image, cv2.COLOR_BGR2HSV)
    
    # Calculate color masks and counts in a more efficient manner
    green_count = cv2.countNonZero(cv2.inRange(hsv_image, *GREEN_RANGE))
    purple_count = cv2.countNonZero(cv2.inRange(hsv_image, *PURPLE_RANGE))
    
    # Combine red mask calculations to avoid unnecessary operations
    red_mask = cv2.inRange(hsv_image, *RED_RANGE_1)
    red_mask = cv2.bitwise_or(red_mask, cv2.inRange(hsv_image, *RED_RANGE_2))
    red_count = cv2.countNonZero(red_mask)
    
    # Use direct comparison instead of dictionary lookup for better performance
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
    # Get card dimensions for filtering - avoid recomputation
    card_height, card_width = card_image.shape[:2]
    card_area = card_width * card_height
    min_shape_area = 0.03 * card_area  # Threshold calculation done once
    
    # Detect shapes on the card - this is a bottleneck operation
    shape_results = shape_detection_model(card_image)
    boxes_array = shape_results[0].boxes.xyxy.cpu().numpy()
    
    # Filter boxes more efficiently by vectorizing operations
    shape_boxes = []
    for box_coords in boxes_array:
        x1, y1, x2, y2 = box_coords.astype(int)
        shape_area = (x2 - x1) * (y2 - y1)
        if shape_area > min_shape_area:
            shape_boxes.append((x1, y1, x2, y2))
    
    # Early return for empty detection to avoid unnecessary processing
    count = min(len(shape_boxes), 3)
    if count == 0:
        return {
            'count': 0,
            'color': 'unknown',
            'fill': 'unknown',
            'shape': 'unknown',
            'box': box
        }
    
    # Batch preprocessing for better CPU utilization
    shape_images = []
    for x1, y1, x2, y2 in shape_boxes:
        shape_img = card_image[y1:y2, x1:x2]
        shape_images.append(shape_img)
    
    # Pre-compute model input dimensions once
    fill_input_shape = fill_model.input_shape[1:3]
    shape_input_shape = shape_model.input_shape[1:3]
    
    # Prepare batches for model prediction to reduce overhead
    color_labels = []
    fill_inputs = []
    shape_inputs = []
    
    for img in shape_images:
        color_labels.append(predict_color(img))
        
        # Resize once per model to avoid repeated operations
        fill_img = cv2.resize(img, fill_input_shape) / 255.0
        shape_img = cv2.resize(img, shape_input_shape) / 255.0
        
        fill_inputs.append(fill_img)
        shape_inputs.append(shape_img)
    
    # Convert to numpy arrays for batch prediction
    fill_batch = np.array(fill_inputs)
    shape_batch = np.array(shape_inputs)
    
    # Perform batch predictions - major optimization for CPU
    fill_preds = fill_model.predict(fill_batch, verbose=0)
    shape_preds = shape_model.predict(shape_batch, verbose=0)
    
    # Process predictions
    fill_labels = [FILL_LABELS[np.argmax(pred)] for pred in fill_preds]
    shape_labels = [SHAPE_LABELS[np.argmax(pred)] for pred in shape_preds]
    
    # Compute most common values efficiently
    def most_common(lst):
        return max(set(lst), key=lst.count)
    
    return {
        'count': count,
        'color': most_common(color_labels),
        'fill': most_common(fill_labels),
        'shape': most_common(shape_labels),
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
    # Detect cards from the board image
    cards = detect_cards_from_image(board_image, card_detection_model)
    
    # Pre-allocate array for better memory management
    card_data = []
    
    # Process each detected card
    for card_image, box in cards:
        features = predict_card_features(card_image, shape_detection_model, fill_model, shape_model, box)
        
        # Store coordinates as a string to maintain compatibility
        card_data.append({
            "Count": features['count'],
            "Color": features['color'],
            "Fill": features['fill'],
            "Shape": features['shape'],
            "Coordinates": f"{box[0]}, {box[1]}, {box[2]}, {box[3]}"
        })
    
    # Create DataFrame in one operation instead of growing it incrementally
    return pd.DataFrame(card_data)
