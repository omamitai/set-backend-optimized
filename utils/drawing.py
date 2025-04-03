import cv2
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def draw_sets_on_image(board_image: np.ndarray, sets_info: List[Dict[str, Any]]) -> np.ndarray:
    """
    Efficiently draw bounding boxes and labels for the detected sets.
    
    Optimizations:
    1. Early return if no sets found
    2. Pre-calculates dimensions and visual parameters
    3. Uses adaptive thickness and font size based on image dimensions
    4. Optimizes text rendering with proper placement
    
    Args:
        board_image (numpy.ndarray): Board image in BGR format
        sets_info (list): List of dictionaries containing set information
        
    Returns:
        numpy.ndarray: The annotated board image
    """
    # Early return if no sets found
    if not sets_info:
        return board_image
    
    # Create a copy of the image to avoid modifying the original
    result_image = board_image.copy()
    
    # Define visually distinct colors for better separation
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 0),    # Dark Blue
        (0, 128, 0),    # Dark Green
        (0, 0, 128),    # Dark Red
    ]
    
    # Calculate image dimensions once
    img_height, img_width = result_image.shape[:2]
    
    # Scale visual parameters based on image size for consistent appearance
    # Ensure minimum values for very small images
    base_thickness = max(1, min(3, img_width // max(640, 1)))  # Adaptive thickness
    base_expansion = max(3, min(15, img_width // max(320, 1)))  # Adaptive expansion
    font_scale = max(0.5, min(2.0, img_width / max(1000, 1)))  # Adaptive font size
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Process each set
    for index, set_info in enumerate(sets_info):
        # Cycle through colors
        color = colors[index % len(colors)]
        
        # Scale thickness and expansion based on set index for visual distinction
        thickness = base_thickness + (index % 3)
        expansion = base_expansion + (index * 3)
        
        # Process each card in the set
        for i, card in enumerate(set_info['cards']):
            # Parse coordinates efficiently
            coords = [int(x) for x in card['Coordinates'].split(',')]
            x1, y1, x2, y2 = coords
            
            # Apply expansion with bounds checking
            x1_expanded = max(0, x1 - expansion)
            y1_expanded = max(0, y1 - expansion)
            x2_expanded = min(img_width, x2 + expansion)
            y2_expanded = min(img_height, y2 + expansion)
            
            # Draw rectangle
            cv2.rectangle(
                result_image, 
                (x1_expanded, y1_expanded), 
                (x2_expanded, y2_expanded), 
                color, 
                thickness
            )
            
            # Only add text label to first card in each set
            if i == 0:
                # Calculate text size for better placement
                text = f"Set {index + 1}"
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                
                # Ensure text is visible (above the card)
                text_x = x1_expanded
                text_y = max(text_size[1] + 10, y1_expanded - 10)
                
                # Draw text
                cv2.putText(
                    result_image, 
                    text, 
                    (text_x, text_y), 
                    font, 
                    font_scale, 
                    color, 
                    thickness
                )
    
    return result_image
