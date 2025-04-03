import numpy as np
import cv2
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Any
import logging

from utils.detection import detect_cards_from_image

logger = logging.getLogger(__name__)

# Pre-define HSV color ranges as constants for better performance
# These are optimized ranges based on domain knowledge of the Set game
GREEN_LOWER = np.array([40, 50, 50])
GREEN_UPPER = np.array([80, 255, 255])
PURPLE_LOWER = np.array([120, 50, 50])
PURPLE_UPPER = np.array([160, 255, 255])
RED_LOWER1 = np.array([0, 50, 50])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 50, 50])
RED_UPPER2 = np.array([180, 255, 255])

def predict_color(shape_image: np.ndarray) -> str:
    """
    Efficiently predict the dominant color in the shape image using HSV thresholds.
    
    Optimizations:
    1. Uses pre-defined color ranges as constants
    2. Downsamples image first for faster processing
    3. Uses numpy sum instead of cv2.countNonZero for better performance
    
    Args:
        shape_image (numpy.ndarray): Image in BGR format
        
    Returns:
        str: Predicted color ('green', 'purple', or 'red')
    """
    # Resize to smaller dimensions for faster processing
    # This significantly improves speed with minimal impact on accuracy
    h, w = shape_image.shape[:2]
    if max(h, w) > 100:
        scale = 100 / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        shape_image = cv2.resize(shape_image, new_size, interpolation=cv2.INTER_AREA)
    
    # Convert to HSV color space once
    hsv_image = cv2.cvtColor(shape_image, cv2.COLOR_BGR2HSV)
    
    # Create masks for each color (using pre-defined ranges)
    green_mask = cv2.inRange(hsv_image, GREEN_LOWER, GREEN_UPPER)
    purple_mask = cv2.inRange(hsv_image, PURPLE_LOWER, PURPLE_UPPER)
    
    # Combine red masks (red hue wraps around in HSV space)
    red_mask1 = cv2.inRange(hsv_image, RED_LOWER1, RED_UPPER1)
    red_mask2 = cv2.inRange(hsv_image, RED_LOWER2, RED_UPPER2)
    red_mask = cv2.add(red_mask1, red_mask2)
    
    # Use numpy sum for faster counting (divide by 255 to get pixel count)
    green_count = np.sum(green_mask) // 255
    purple_count = np.sum(purple_mask) // 255
    red_count = np.sum(red_mask) // 255
    
    # Determine color with most pixels
    if green_count > purple_count and green_count > red_count:
        return 'green'
    elif purple_count > red_count:
        return 'purple'
    else:
        return 'red'

def preprocess_for_classification(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Efficiently preprocess an image for neural network classification.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        target_shape (tuple): Target dimensions (width, height)
        
    Returns:
        numpy.ndarray: Preprocessed image ready for model input
    """
    # Resize using INTER_AREA for downsampling (better quality)
    resized = cv2.resize(image, target_shape, interpolation=cv2.INTER_AREA)
    
    # Normalize to [0,1] range (faster method)
    normalized = resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    batched = np.expand_dims(normalized, axis=0)
    
    return batched

def predict_card_features(card_image: np.ndarray, 
                          shape_detection_model: Any, 
                          fill_model: Any, 
                          shape_model: Any, 
                          box: np.ndarray) -> Dict[str, Any]:
    """
    Predict card features using batch processing for multiple shapes.
    
    Optimizations:
    1. Batched preprocessing and inference for all shapes at once
    2. Early filtering of shape detections by size and aspect ratio
    3. More efficient model inference with lower verbosity
    
    Args:
        card_image (numpy.ndarray): Card image in BGR format
        shape_detection_model: YOLO model for detecting shapes on the card
        fill_model: Keras model for fill classification
        shape_model: Keras model for shape classification
        box (array-like): Bounding box of the card
        
    Returns:
        dict: Dictionary containing predicted card features
    """
    # Get input shapes for models once
    fill_input_shape = tuple(fill_model.input_shape[1:3])
    shape_input_shape = tuple(shape_model.input_shape[1:3])
    
    # Detect shapes on the card with appropriate confidence
    shape_results = shape_detection_model(card_image, conf=0.4)
    
    # Calculate card area for filtering small detections
    card_height, card_width = card_image.shape[:2]
    card_area = card_width * card_height
    
    # Filter shapes by size and aspect ratio
    filtered_boxes = []
    for detected_box in shape_results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = detected_box.astype(int)
        shape_width = x2 - x1
        shape_height = y2 - y1
        shape_area = shape_width * shape_height
        
        # Skip if too small (less than 3% of card area)
        if shape_area < 0.03 * card_area:
            continue
        
        # Filter by aspect ratio (most Set game shapes have reasonable aspect ratios)
        if shape_width > 0 and shape_height > 0:
            aspect = shape_width / shape_height
            if 0.5 <= aspect <= 2.0:  # Reasonable aspect ratio range
                filtered_boxes.append(detected_box)
    
    # In Set game, cards have 1-3 shapes
    count = min(len(filtered_boxes), 3)
    
    # If no shapes detected, return defaults
    if count == 0:
        return {
            'count': 0,
            'color': 'unknown',
            'fill': 'unknown',
            'shape': 'unknown',
            'box': box
        }
        
    # Ensure we have valid shapes to process (avoid empty images)
    filtered_boxes = [box for box in filtered_boxes if 
                    box[2] - box[0] > 5 and box[3] - box[1] > 5]
    if not filtered_boxes:
        return {
            'count': 0,
            'color': 'unknown',
            'fill': 'unknown',
            'shape': 'unknown',
            'box': box
        }
    
    # Prepare batches for efficient processing
    shape_images = []
    shape_inputs_fill = []
    shape_inputs_shape = []
    
    for shape_box in filtered_boxes:
        shape_box = shape_box.astype(int)
        # Crop the card image to the shape
        shape_img = card_image[shape_box[1]:shape_box[3], shape_box[0]:shape_box[2]]
        shape_images.append(shape_img)
        
        # Preprocess for classification models
        shape_inputs_fill.append(preprocess_for_classification(shape_img, fill_input_shape)[0])
        shape_inputs_shape.append(preprocess_for_classification(shape_img, shape_input_shape)[0]) 
    
    # Batch predictions for better CPU utilization
    fill_batch = np.array(shape_inputs_fill)
    shape_batch = np.array(shape_inputs_shape)
    
    # Make batch predictions with reduced verbosity
    fill_preds = fill_model.predict(fill_batch, verbose=0)
    shape_preds = shape_model.predict(shape_batch, verbose=0)
    
    # Get color predictions (parallel processing if enough shapes)
    if count >= 3:
        with ThreadPoolExecutor(max_workers=3) as executor:
            color_labels = list(executor.map(predict_color, shape_images))
    else:
        color_labels = [predict_color(img) for img in shape_images]
    
    # Convert predictions to labels
    fill_labels = [['empty', 'full', 'striped'][np.argmax(pred)] for pred in fill_preds]
    shape_labels = [['diamond', 'oval', 'squiggle'][np.argmax(pred)] for pred in shape_preds]
    
    # Use voting to get the most common prediction for each feature
    def most_common(lst):
        return max(set(lst), key=lst.count)
        
    color_label = most_common(color_labels)
    fill_label = most_common(fill_labels)
    shape_label = most_common(shape_labels)
    
    return {
        'count': count,
        'color': color_label,
        'fill': fill_label,
        'shape': shape_label,
        'box': box
    }

def classify_cards_from_board_image(board_image: np.ndarray, 
                                   card_detection_model: Any, 
                                   shape_detection_model: Any, 
                                   fill_model: Any, 
                                   shape_model: Any) -> pd.DataFrame:
    """
    Classify cards on a board image with parallel processing.
    
    Optimizations:
    1. Uses ThreadPoolExecutor for parallel card feature prediction
    2. Limits thread count to avoid CPU oversubscription
    3. Constructs DataFrame once at the end
    
    Args:
        board_image (numpy.ndarray): The corrected board image
        card_detection_model: YOLO model for card detection
        shape_detection_model: YOLO model for shape detection
        fill_model: Keras model for fill classification
        shape_model: Keras model for shape classification
        
    Returns:
        pandas.DataFrame: DataFrame containing card features
    """
    # Detect cards from image 
    cards = detect_cards_from_image(board_image, card_detection_model)
    
    # Early exit if no cards detected
    if not cards:
        return pd.DataFrame(columns=["Count", "Color", "Fill", "Shape", "Coordinates"])
        
    # Determine optimal thread count (don't oversubscribe CPU)
    # ThreadPoolExecutor is better than ProcessPoolExecutor for I/O bound tasks
    max_workers = min(4, len(cards))
    
    # Process cards in parallel
    card_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create futures list
        futures = []
        for card_image, box in cards:
            future = executor.submit(
                predict_card_features, 
                card_image, 
                shape_detection_model, 
                fill_model, 
                shape_model, 
                box
            )
            futures.append(future)
        
        # Collect results as they complete
        for future in futures:
            features = future.result()
            card_data.append({
                "Count": features['count'],
                "Color": features['color'],
                "Fill": features['fill'],
                "Shape": features['shape'],
                "Coordinates": f"{features['box'][0]}, {features['box'][1]}, {features['box'][2]}, {features['box'][3]}"
            })
    
    # Create DataFrame once (more efficient than appending)
    return pd.DataFrame(card_data)
