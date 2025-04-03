import numpy as np
import cv2
from typing import Tuple, List, Any
import logging

logger = logging.getLogger(__name__)

def check_and_rotate_input_image(board_image: np.ndarray, card_detection_model: Any) -> Tuple[np.ndarray, bool]:
    """
    Efficiently checks the orientation of the board image and rotates if necessary.
    
    Optimized with:
    1. Fast dimension-based heuristic as first check
    2. Optimized model inference with confidence thresholds
    3. Card aspect ratio analysis
    
    Args:
        board_image (numpy.ndarray): The original board image (BGR format)
        card_detection_model: YOLO model for card detection
        
    Returns:
        tuple: (rotated_image, was_rotated) where rotated_image is either the original
               image or rotated 90° clockwise, and was_rotated is a boolean flag
    """
    h, w = board_image.shape[:2]
    
    # Fast path: Use image dimensions as initial heuristic
    # Most Set game photos are in landscape orientation
    if h > w * 1.2:  # Image is clearly in portrait orientation
        logger.debug("Image appears to be in portrait orientation based on dimensions")
        rotated_image = cv2.rotate(board_image, cv2.ROTATE_90_CLOCKWISE)
        return rotated_image, True
    
    # Second check: Use card detection and aspect ratios
    # Standard playing cards have width/height ratio of ~0.7
    card_results = card_detection_model(board_image, conf=0.4)  # Lower confidence to detect more cards
    card_boxes = card_results[0].boxes.xyxy.cpu().numpy().astype(int)
    
    # If no cards detected, keep original orientation
    if len(card_boxes) == 0:
        return board_image, False
    
    # Calculate average aspect ratio of detected cards
    aspect_ratios = []
    for box in card_boxes:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        if width > 0 and height > 0:  # Avoid division by zero
            aspect_ratios.append(width / height)
    
    # If mean aspect ratio < 0.8, cards are likely in portrait orientation
    if aspect_ratios and np.mean(aspect_ratios) < 0.8:
        logger.debug(f"Rotating image based on card aspect ratios (mean: {np.mean(aspect_ratios):.2f})")
        rotated_image = cv2.rotate(board_image, cv2.ROTATE_90_CLOCKWISE)
        return rotated_image, True
    
    return board_image, False

def detect_cards_from_image(board_image: np.ndarray, card_detection_model: Any) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Efficiently detect card regions on the board image using the YOLO model.
    
    Optimizations:
    1. Uses optimized confidence thresholds
    2. Sorts detections by area to prioritize clearer cards
    3. Adds minimal padding to ensure complete card capture
    
    Args:
        board_image (numpy.ndarray): The (corrected) board image
        card_detection_model: YOLO model for card detection
        
    Returns:
        list: List of tuples containing (card_image, bounding_box)
    """
    # Run detection with optimized parameters for CPU
    card_results = card_detection_model(
        board_image,
        conf=0.5,  # Balanced confidence for precision/recall
        iou=0.45   # Slightly higher IoU for better NMS
    )
    
    # Extract and sort bounding boxes by area (largest first)
    card_boxes = card_results[0].boxes.xyxy.cpu().numpy().astype(int)
    
    # If too many cards detected (likely false positives), keep only top confidence ones
    if len(card_boxes) > 15:  # Set games typically have at most 12-15 cards
        confidences = card_results[0].boxes.conf.cpu().numpy()
        # Get indices sorted by confidence (highest first)
        sorted_indices = np.argsort(confidences)[::-1][:15]
        card_boxes = card_boxes[sorted_indices]
    
    # Sort boxes by area (largest first) for better feature detection
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in card_boxes]
    sorted_indices = np.argsort(areas)[::-1]
    card_boxes = card_boxes[sorted_indices]
    
    # Process each card
    cards = []
    h, w = board_image.shape[:2]
    
    for box in card_boxes:
        x1, y1, x2, y2 = box
        
        # Add small padding (5px) to ensure whole card is captured
        x1 = max(0, x1 - 5)
        y1 = max(0, y1 - 5)
        x2 = min(w, x2 + 5)
        y2 = min(h, y2 + 5)
        
        # Only process reasonable sized cards (filter out tiny detections)
        card_width, card_height = x2 - x1, y2 - y1
        if card_width < 20 or card_height < 20:
            continue
            
        # Use copy() to ensure we have a contiguous array for better memory access
        card_img = board_image[y1:y2, x1:x2].copy()
        cards.append((card_img, box))
    
    return cards
