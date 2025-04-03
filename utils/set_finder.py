import pandas as pd
from itertools import combinations
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def is_set(cards: List[Dict[str, Any]]) -> bool:
    """
    Efficiently determine if a group of cards forms a valid set.
    
    In the Set game, for each feature (Count, Color, Fill, Shape), the values
    must be either all the same or all different.
    
    Optimizations:
    1. Fast path for invalid cards
    2. Early termination as soon as any feature fails validation
    3. Efficient set operations for feature checking
    
    Args:
        cards (list): List of card feature dictionaries
        
    Returns:
        bool: True if the cards form a valid set, False otherwise
    """
    # Fast path: If any card has unknown features or 0 count, it's not a valid set
    for card in cards:
        if ('unknown' in card.values() or 
            card['Count'] == 0 or  # Undetected shapes
            None in card.values()):
            return False
    
    # Check each feature - exit early if any feature fails
    for feature in ['Count', 'Color', 'Fill', 'Shape']:
        # Use set for O(1) uniqueness check
        values = set(card[feature] for card in cards)
        
        # Set game rule: all same (len=1) or all different (len=3)
        if len(values) == 2 or len(values) > 3:
            return False
            
    return True

def find_sets(card_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Find all valid sets from the card DataFrame.
    
    Optimizations:
    1. Convert DataFrame to list of dicts for faster access
    2. Pre-filter invalid cards before generating combinations
    3. Return early if not enough valid cards
    
    Args:
        card_df (pandas.DataFrame): DataFrame containing card features
        
    Returns:
        list: List of dictionaries with set details
    """
    sets_found = []
    
    # Early exit if not enough cards
    if len(card_df) < 3:
        return sets_found
    
    # Convert DataFrame to list of dictionaries once for faster access
    # Use copy to avoid modifying original data
    cards = card_df.copy().to_dict('records')
    
    # Filter out cards with unknown features (optimization)
    valid_cards = []
    for i, card in enumerate(cards):
        if ('unknown' not in card.values() and 
            card['Count'] != 0 and 
            None not in card.values()):
            # Add index to card for later reconstruction
            card['_index'] = i
            valid_cards.append(card)
    
    # Early exit if not enough valid cards
    if len(valid_cards) < 3:
        return sets_found
    
    # Generate combinations of 3 cards (combinations is already efficient)
    for combo in combinations(valid_cards, 3):
        if is_set(combo):
            # Create set info with original indices and card data
            set_info = {
                'set_indices': [card['_index'] for card in combo],
                'cards': [{k: card[k] for k in ['Count', 'Color', 'Fill', 'Shape', 'Coordinates']} 
                          for card in combo]
            }
            sets_found.append(set_info)
    
    logger.info(f"Found {len(sets_found)} valid sets")
    return sets_found
