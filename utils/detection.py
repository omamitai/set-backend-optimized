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
        tuple: (rotated_image, was_rotated) where rotated_image is either the original
               image or rotated 90° clockwise, and was_rotated is a boolean flag.
    """
    # Run card detection on the board image
    card_results = card_detection_model(board_image)
    
    # Handle empty results case
    if len(card_results) == 0 or len(card_results[0].boxes) == 0:
        return board_image, False
        
    # Extract bounding boxes
    card_boxes = card_results[0].boxes.xyxy.cpu().numpy().astype(int)

    # If no cards are detected, return the original image
    if len(card_boxes) == 0:
        return board_image, False

    # Use vectorized operations for better performance
    # Calculate width and height of all boxes in one operation
    widths = card_boxes[:, 2] - card_boxes[:, 0]
    heights = card_boxes[:, 3] - card_boxes[:, 1]
    
    # Calculate averages with numpy for better efficiency
    avg_width = np.mean(widths)
    avg_height = np.mean(heights)

    # If cards are generally vertical (taller than wide), rotate the image
    if avg_height > avg_width:
        # Perform the rotation
        rotated_image = cv2.rotate(board_image, cv2.ROTATE_90_CLOCKWISE)
        return rotated_image, True
    else:
        # Return original image if no rotation needed
        return board_image, False

def detect_cards_from_image(board_image, card_detection_model):
    """
    Detect card regions on the board image using the YOLO card detection model.

    Args:
        board_image (numpy.ndarray): The (corrected) board image.
        card_detection_model: YOLO model for card detection.

    Returns:
        list: List of tuples containing (card_image, bounding_box).
    """
    # Get image dimensions once
    img_height, img_width = board_image.shape[:2]
    
    # Run the card detection model
    card_results = card_detection_model(board_image)
    
    # Handle empty results
    if len(card_results) == 0 or len(card_results[0].boxes) == 0:
        return []
        
    # Extract boxes from results
    card_boxes = card_results[0].boxes.xyxy.cpu().numpy().astype(int)
    
    # Pre-allocate list with estimated capacity for better memory efficiency
    cards = []
    
    # Extract each card image based on its bounding box
    for box in card_boxes:
        x1, y1, x2, y2 = box
        
        # Ensure bounds are within image dimensions
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_width, x2), min(img_height, y2)
        
        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue
            
        # Extract the card image - use direct slicing for efficiency
        card_img = board_image[y1:y2, x1:x2]
        cards.append((card_img, box))
    
    return cards
