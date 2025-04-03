import cv2
import numpy as np
from collections import defaultdict

# Pre-define colors for better performance and readability
COLORS = [
    (255, 0, 0),   # Blue
    (0, 255, 0),   # Green
    (0, 0, 255),   # Red
    (255, 255, 0), # Cyan
    (255, 0, 255), # Magenta
    (0, 255, 255)  # Yellow
]

def parse_coordinates(coord_str):
    """
    Parse coordinate string to integer values.
    
    Args:
        coord_str (str): Comma-separated coordinate string "x1, y1, x2, y2"
    
    Returns:
        tuple: (x1, y1, x2, y2) as integers
    """
    return tuple(map(int, coord_str.split(',')))

def draw_sets_on_image(board_image, sets_info):
    """
    Draw bounding boxes and labels for the detected sets on the provided board image.
    
    Args:
        board_image (numpy.ndarray): Board image in BGR format.
        sets_info (list): List of dictionaries containing set information.
        
    Returns:
        numpy.ndarray: The annotated board image.
    """
    # Early return for empty input - avoids unnecessary processing
    if not sets_info:
        return board_image.copy()
        
    # Get image dimensions and calculate diagonal for proportional scaling
    img_height, img_width = board_image.shape[:2]
    img_diagonal = np.sqrt(img_width**2 + img_height**2)
    
    # Scale parameters based on image size
    base_thickness = max(1, int(img_diagonal * 0.004))  # 0.4% of diagonal
    base_expansion = max(2, int(img_diagonal * 0.005))  # 0.5% of diagonal
    font_scale = max(0.5, img_diagonal * 0.0007)        # 0.07% of diagonal
    
    # Create a copy of the image to avoid modifying the original
    result_image = board_image.copy()
    
    # Track card appearance count to apply the exact specified expansion
    # We'll use defaultdict to avoid explicit initialization
    card_appearance_count = defaultdict(int)
    
    # First pass: Count card appearances across all sets before drawing anything
    for set_info in sets_info:
        for card in set_info['cards']:
            coord_key = card['Coordinates'] if not isinstance(card['Coordinates'], str) else card['Coordinates']
            card_appearance_count[coord_key] += 1
    
    # Process each set using vectorized operations where possible
    for index, set_info in enumerate(sets_info):
        color = COLORS[index % len(COLORS)]
        thickness = base_thickness
        
        # Process each card in the set
        for i, card in enumerate(set_info['cards']):
            # Extract coordinates efficiently
            if isinstance(card['Coordinates'], str):
                coord_str = card['Coordinates']
                x1, y1, x2, y2 = parse_coordinates(coord_str)
            else:
                x1, y1, x2, y2 = card['Coordinates']
                coord_str = f"{x1}, {y1}, {x2}, {y2}"
            
            # Calculate expansion based on how many sets this card belongs to
            appearances = card_appearance_count[coord_str]
            
            # Precisely targeted expansion based on requirement:
            # - 0 expansion for cards in exactly 1 set
            # - 1 unit for cards in exactly 2 sets
            # - 2 units for cards in exactly 3 sets
            expansion = 0
            if appearances == 2:
                expansion = base_expansion
            elif appearances >= 3:
                expansion = 2 * base_expansion
            
            # Calculate expanded box with bounds checking
            x1_expanded = max(0, x1 - expansion)
            y1_expanded = max(0, y1 - expansion)
            x2_expanded = min(img_width, x2 + expansion)
            y2_expanded = min(img_height, y2 + expansion)
            
            # Draw bounding box
            cv2.rectangle(
                result_image, 
                (x1_expanded, y1_expanded), 
                (x2_expanded, y2_expanded), 
                color, 
                thickness
            )
            
            # Only draw label for the first card in each set
            if i == 0:
                # Scale text margin proportionally to image size
                text_margin = max(5, int(img_diagonal * 0.005))
                text_y = max(text_margin, y1_expanded - text_margin)
                
                cv2.putText(
                    result_image,
                    f"Set {index + 1}",
                    (x1_expanded, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    thickness
                )
    
    return result_image
