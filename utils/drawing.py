import cv2

def draw_sets_on_image(board_image, sets_info):
    """
    Draw bounding boxes and labels for the detected sets on the provided board image.

    Args:
        board_image (numpy.ndarray): Board image in BGR format.
        sets_info (list): List of dictionaries containing set information.

    Returns:
        numpy.ndarray: The annotated board image.
    """
    # Make a copy to avoid modifying the original image
    result_image = board_image.copy()
    
    # Get image dimensions once for bounds checking
    img_height, img_width = result_image.shape[:2]
    
    # Predefined colors for better visibility (BGR format)
    # Using a tuple for faster access compared to a list
    colors = (
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255)   # Yellow
    )
    
    # Base parameters
    base_thickness = 8
    base_expansion = 5
    color_count = len(colors)
    
    # Process each set
    for index, set_info in enumerate(sets_info):
        # Select color and parameters for this set
        color = colors[index % color_count]
        thickness = base_thickness + 2 * index
        expansion = base_expansion + 15 * index
        
        # Process each card in the set
        for i, card in enumerate(set_info['cards']):
            # Parse coordinates efficiently
            if isinstance(card['Coordinates'], str):
                # Split once and convert all at once
                coords = card['Coordinates'].split(',')
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            else:
                # Handle case where coordinates might already be numeric
                x1, y1, x2, y2 = card['Coordinates']
            
            # Calculate expanded box with bounds checking
            x1_expanded = max(0, x1 - expansion)
            y1_expanded = max(0, y1 - expansion)
            x2_expanded = min(img_width, x2 + expansion)
            y2_expanded = min(img_height, y2 + expansion)
            
            # Draw the rectangle in one call
            cv2.rectangle(
                result_image,
                (x1_expanded, y1_expanded),
                (x2_expanded, y2_expanded),
                color,
                thickness
            )
            
            # Add label to first card of each set
            if i == 0:
                # Position text above the box with safe margin
                text_y = max(30, y1_expanded - 10)
                cv2.putText(
                    result_image,
                    f"Set {index + 1}",
                    (x1_expanded, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    thickness
                )
    
    return result_image
