import numpy as np
import cv2

def check_and_rotate_input_image(board_image, card_detection_model):
    """
    Checks the orientation of the board image by detecting card bounding boxes.
    If the average detected card height is greater than the average card width,
    rotates the board image 90° clockwise.

    Args:
        board_image (numpy.ndarray): The original board image (BGR format).
        card_detection_model: YOLO model for card detection.

    Returns:
        tuple: (rotated_image, was_rotated, detected_boxes) where:
               - rotated_image is either the original image or rotated 90° clockwise
               - was_rotated is a boolean flag
               - detected_boxes contains the detected card boxes (if not rotated)
    """
    # Run card detection on the board image
    card_results = card_detection_model(board_image)
    
    # Extract bounding boxes efficiently
    card_boxes = card_results[0].boxes.xyxy.cpu().numpy().astype(int)

    # If no cards are detected, return the original image
    if len(card_boxes) == 0:
        return board_image, False, None

    # Vectorized calculation of widths and heights - more efficient on CPU
    widths = card_boxes[:, 2] - card_boxes[:, 0]
    heights = card_boxes[:, 3] - card_boxes[:, 1]
    
    # Calculate averages in one operation
    avg_width = np.mean(widths)
    avg_height = np.mean(heights)

    # Determine if rotation is needed
    if avg_height > avg_width:
        # Cards are taller than wide - rotate the image
        rotated_image = cv2.rotate(board_image, cv2.ROTATE_90_CLOCKWISE)
        return rotated_image, True, None
    else:
        # Cards are wider than tall - keep original orientation and return detected boxes
        return board_image, False, card_boxes

def detect_cards_from_image(board_image, card_detection_model, pre_detected_boxes=None):
    """
    Detect card regions on the board image using the YOLO card detection model.

    Args:
        board_image (numpy.ndarray): The (corrected) board image.
        card_detection_model: YOLO model for card detection.
        pre_detected_boxes (numpy.ndarray, optional): Pre-detected card boxes to avoid
                                                     redundant detection.

    Returns:
        list: List of tuples containing (card_image, bounding_box).
    """
    # Use pre-detected boxes if available to avoid redundant detection
    if pre_detected_boxes is not None:
        card_boxes = pre_detected_boxes
    else:
        # Run detection only if needed
        card_results = card_detection_model(board_image)
        card_boxes = card_results[0].boxes.xyxy.cpu().numpy().astype(int)
    
    # Pre-allocate list for better memory efficiency
    cards = []
    
    # Extract card images using the detected boxes
    for box in card_boxes:
        x1, y1, x2, y2 = box
        
        # Ensure bounds are within image dimensions
        h, w = board_image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue
            
        # Extract card image
        card_img = board_image[y1:y2, x1:x2]
        
        # Only add valid cards (non-empty images)
        if card_img.size > 0:
            cards.append((card_img, box))
    
    return cards
