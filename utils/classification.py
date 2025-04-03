import numpy as np
import cv2
import pandas as pd
from functools import lru_cache
import concurrent.futures
from typing import List, Dict, Tuple, Any
from utils.detection import detect_cards_from_image

# Pre-define color ranges as constants to avoid repeated array creation
GREEN_RANGE = (np.array([40, 50, 50]), np.array([80, 255, 255]))
PURPLE_RANGE = (np.array([120, 50, 50]), np.array([160, 255, 255]))
RED_RANGE_1 = (np.array([0, 50, 50]), np.array([10, 255, 255]))
RED_RANGE_2 = (np.array([170, 50, 50]), np.array([180, 255, 255]))

# Constants for feature labels
FILL_LABELS = ['empty', 'full', 'striped']
SHAPE_LABELS = ['diamond', 'oval', 'squiggle']

# Thread pool for parallel processing
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

@lru_cache(maxsize=32)
def get_model_input_shapes(fill_model, shape_model):
    """
    Get input shapes for models and cache them to avoid repeated queries.
    
    Args:
        fill_model: Keras model for fill classification
        shape_model: Keras model for shape classification
        
    Returns:
        tuple: (fill_input_shape, shape_input_shape)
    """
    fill_input_shape = fill_model.input_shape[1:3]
    shape_input_shape = shape_model.input_shape[1:3]
    return fill_input_shape, shape_input_shape

def predict_color_batch(shape_images: List[np.ndarray]) -> List[str]:
    """
    Predict the dominant color for a batch of shape images using HSV thresholds.
    
    Args:
        shape_images: List of shape images in BGR format
        
    Returns:
        List of predicted colors ('green', 'purple', or 'red')
    """
    results = []
    
    for shape_image in shape_images:
        # Convert to HSV once
        hsv_image = cv2.cvtColor(shape_image, cv2.COLOR_BGR2HSV)
        
        # Calculate all masks and counts efficiently
        green_count = cv2.countNonZero(cv2.inRange(hsv_image, *GREEN_RANGE))
        purple_count = cv2.countNonZero(cv2.inRange(hsv_image, *PURPLE_RANGE))
        
        # Combine red mask calculations to avoid creating intermediate masks
        red_mask1 = cv2.inRange(hsv_image, *RED_RANGE_1)
        red_mask2 = cv2.inRange(hsv_image, *RED_RANGE_2)
        red_count = cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2)
        
        # Direct comparison instead of dict lookup
        if green_count > purple_count and green_count > red_count:
            results.append('green')
        elif purple_count > red_count:
            results.append('purple')
        else:
            results.append('red')
    
    return results

def prepare_shape_image_batch(card_image: np.ndarray, filtered_boxes: List[np.ndarray], 
                              fill_input_shape: Tuple[int, int], shape_input_shape: Tuple[int, int]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Extract and preprocess shape images for batch prediction.
    
    Args:
        card_image: The card image in BGR format
        filtered_boxes: List of bounding boxes for shapes
        fill_input_shape: Input shape for the fill model
        shape_input_shape: Input shape for the shape model
        
    Returns:
        Tuple of (original_shape_images, fill_inputs, shape_inputs)
    """
    original_shape_images = []
    fill_inputs = []
    shape_inputs = []
    
    for shape_box in filtered_boxes:
        x1, y1, x2, y2 = shape_box.astype(int)
        
        # Ensure within bounds
        h, w = card_image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Skip invalid boxes
        if x1 >= x2 or y1 >= y2:
            continue
            
        # Extract shape image
        shape_img = card_image[y1:y2, x1:x2]
        
        # Skip empty or invalid images
        if shape_img.size == 0 or shape_img.shape[0] == 0 or shape_img.shape[1] == 0:
            continue
            
        original_shape_images.append(shape_img)
        
        # Preprocess for models - resize once per model
        fill_img = cv2.resize(shape_img, fill_input_shape) / 255.0
        shape_img_resized = cv2.resize(shape_img, shape_input_shape) / 255.0
        
        # Store inputs for batch prediction
        fill_inputs.append(fill_img)
        shape_inputs.append(shape_img_resized)
    
    return original_shape_images, fill_inputs, shape_inputs

def predict_card_features(card_image: np.ndarray, shape_detection_model, fill_model, shape_model, box) -> Dict[str, Any]:
    """
    Predict card features (count, color, fill, shape) from a card image.
    
    Args:
        card_image: Card image in BGR format
        shape_detection_model: YOLO model for detecting shapes on the card
        fill_model: Keras model for fill classification
        shape_model: Keras model for shape classification
        box: Bounding box of the card
        
    Returns:
        Dictionary containing predicted card features
    """
    # Get model input shapes from cache
    fill_input_shape, shape_input_shape = get_model_input_shapes(fill_model, shape_model)
    
    # Get card dimensions for filtering
    card_height, card_width = card_image.shape[:2]
    card_area = card_width * card_height
    min_shape_area = 0.03 * card_area
    
    # Detect shapes on the card
    shape_results = shape_detection_model(card_image)
    boxes_array = shape_results[0].boxes.xyxy.cpu().numpy()
    
    # Calculate shape areas efficiently using vectorized operations
    if len(boxes_array) > 0:
        widths = boxes_array[:, 2] - boxes_array[:, 0]
        heights = boxes_array[:, 3] - boxes_array[:, 1]
        areas = widths * heights
        
        # Filter small boxes (vectorized)
        mask = areas > min_shape_area
        filtered_boxes = boxes_array[mask]
    else:
        filtered_boxes = []
    
    # Early return for empty detections
    count = min(len(filtered_boxes), 3)
    if count == 0:
        return {
            'count': 0,
            'color': 'unknown',
            'fill': 'unknown',
            'shape': 'unknown',
            'box': box
        }
    
    # Prepare images for batch prediction
    shape_images, fill_inputs, shape_inputs = prepare_shape_image_batch(
        card_image, filtered_boxes, fill_input_shape, shape_input_shape
    )
    
    # Skip if no valid shapes were extracted
    if not shape_images:
        return {
            'count': 0,
            'color': 'unknown',
            'fill': 'unknown',
            'shape': 'unknown',
            'box': box
        }
    
    # Convert to numpy arrays for batch prediction
    fill_batch = np.array(fill_inputs)
    shape_batch = np.array(shape_inputs)
    
    # Process color prediction in a separate thread while models are running
    color_future = executor.submit(predict_color_batch, shape_images)
    
    # Batch predict fill and shape (runs on main thread)
    fill_preds = fill_model.predict(fill_batch, verbose=0)
    shape_preds = shape_model.predict(shape_batch, verbose=0)
    
    # Get color results from thread
    color_labels = color_future.result()
    
    # Convert predictions to labels
    fill_labels = [FILL_LABELS[np.argmax(pred)] for pred in fill_preds]
    shape_labels = [SHAPE_LABELS[np.argmax(pred)] for pred in shape_preds]
    
    # Determine most common values using numpy (faster than Python's max())
    def most_common(labels):
        values, counts = np.unique(labels, return_counts=True)
        return values[np.argmax(counts)]
    
    # Build feature dictionary
    return {
        'count': count,
        'color': most_common(color_labels),
        'fill': most_common(fill_labels),
        'shape': most_common(shape_labels),
        'box': box
    }

def process_card(args):
    """
    Process a single card for parallel execution
    
    Args:
        args: Tuple of (card_image, box, models)
        
    Returns:
        Dictionary of card features
    """
    card_image, box, models = args
    shape_detection_model, fill_model, shape_model = models
    return predict_card_features(card_image, shape_detection_model, fill_model, shape_model, box)

def classify_cards_from_board_image(board_image: np.ndarray, card_detection_model, shape_detection_model, fill_model, shape_model) -> pd.DataFrame:
    """
    Classify cards on a board image by detecting each card and predicting its features.
    Uses parallel processing for multiple cards.
    
    Args:
        board_image: The corrected board image
        card_detection_model: YOLO model for card detection
        shape_detection_model: YOLO model for shape detection
        fill_model: Keras model for fill classification
        shape_model: Keras model for shape classification
        
    Returns:
        DataFrame containing card features
    """
    # Detect cards from the board image
    cards = detect_cards_from_image(board_image, card_detection_model)
    
    # Skip processing if no cards found
    if not cards:
        return pd.DataFrame()
    
    # Group the models for parallel processing
    models = (shape_detection_model, fill_model, shape_model)
    
    # Prepare arguments for parallel processing
    card_args = [(card_image, box, models) for card_image, box in cards]
    
    # Process cards in parallel if there are multiple cards
    if len(cards) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(cards), 4)) as executor:
            features_list = list(executor.map(process_card, card_args))
    else:
        # Process single card without overhead of parallelism
        features_list = [process_card(card_args[0])]
    
    # Convert results to DataFrame format
    card_data = []
    for features in features_list:
        if features['count'] > 0:  # Skip invalid cards
            card_data.append({
                "Count": features['count'],
                "Color": features['color'],
                "Fill": features['fill'],
                "Shape": features['shape'],
                "Coordinates": f"{features['box'][0]}, {features['box'][1]}, {features['box'][2]}, {features['box'][3]}"
            })
    
    # Create DataFrame from processed data
    return pd.DataFrame(card_data)
