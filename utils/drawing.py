import cv2
import numpy as np

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
    # Get image dimensions once - avoid repeated property access
    img_height, img_width = board_image.shape[:2]
    
    # Base parameters
    base_thickness = 8
    base_expansion = 5
    
    # Create a copy of the image to avoid modifying the original
    result_image = board_image.copy()
    
    # Process each set
    for index, set_info in enumerate(sets_info):
        # Calculate parameters for this set
        color = COLORS[index % len(COLORS)]
        thickness = base_thickness + 2 * index
        expansion = base_expansion + 15 * index
        
        # Process each card in the set
        for i, card in enumerate(set_info['cards']):
            # Parse coordinates once - optimization to avoid repeated parsing
            if isinstance(card['Coordinates'], str):
                x1, y1, x2, y2 = parse_coordinates(card['Coordinates'])
            else:
                x1, y1, x2, y2 = card['Coordinates']
            
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
                # Position label above the box
                label_position = (x1_expanded, max(0, y1_expanded - 10))
                cv2.putText(
                    result_image,
                    f"Set {index + 1}",
                    label_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    thickness
                )
    
    return result_image
