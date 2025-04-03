from itertools import combinations

# Define constants for feature names to avoid string literals
FEATURES = ['Count', 'Color', 'Fill', 'Shape']

def is_set(cards):
    """
    Determine if a group of cards forms a valid set.
    For each feature (Count, Color, Fill, Shape), the values must be either all the same or all different.

    Args:
        cards (list of dict): List of card feature dictionaries.

    Returns:
        bool: True if the cards form a valid set, False otherwise.
    """
    # Fast-path: Verify we have exactly 3 cards (a Set requires 3 cards)
    if len(cards) != 3:
        return False
        
    # Check each feature efficiently
    for feature in FEATURES:
        # Use a set for O(1) lookups and size calculation
        unique_values = {card[feature] for card in cards}
        
        # A valid set must have either all same (1) or all different (3) values
        # Any other count means it's not a set
        if len(unique_values) == 2:  # The only invalid case for 3 cards
            return False
            
    # If we got here, all features passed the test
    return True

def find_sets(card_df):
    """
    Find all valid sets from the card DataFrame by checking every combination of three cards.

    Args:
        card_df (pandas.DataFrame): DataFrame containing card features.

    Returns:
        list: List of dictionaries with set details.
    """
    # Skip processing if fewer than 3 cards
    if len(card_df) < 3:
        return []
        
    # Convert DataFrame to list of records for faster access
    # This avoids repeated DataFrame lookups which can be slow
    cards = card_df.to_dict('records')
    
    # Pre-allocate result list
    sets_found = []
    
    # Use combinations to generate all possible triplets
    # This is more readable and often faster than nested loops
    for i, (idx1, idx2, idx3) in enumerate(combinations(range(len(cards)), 3)):
        card_combo = [cards[idx1], cards[idx2], cards[idx3]]
        
        # Check if this combo forms a set
        if is_set(card_combo):
            # Store set information
            sets_found.append({
                'set_indices': [idx1, idx2, idx3],
                'cards': card_combo
            })
    
    return sets_found

def organize_sets_by_overlap(sets_found):
    """
    Organize found sets by minimizing overlap between consecutive sets.
    This helps with visualization by grouping distinct sets together.
    
    Args:
        sets_found (list): List of found sets as returned by find_sets().
        
    Returns:
        list: Reorganized list of sets to minimize card overlap between consecutive sets.
    """
    if not sets_found:
        return []
        
    # Already organized if only one set
    if len(sets_found) <= 1:
        return sets_found
        
    # Start with the first set
    organized = [sets_found[0]]
    remaining = sets_found[1:]
    
    while remaining:
        # Get the indices of cards in the most recently added set
        last_set_indices = set(organized[-1]['set_indices'])
        
        # Find the set with minimal overlap with the last set
        best_overlap = float('inf')
        best_idx = 0
        
        for i, candidate in enumerate(remaining):
            candidate_indices = set(candidate['set_indices'])
            overlap = len(last_set_indices.intersection(candidate_indices))
            
            if overlap < best_overlap:
                best_overlap = overlap
                best_idx = i
        
        # Add the best set to our organized list
        organized.append(remaining.pop(best_idx))
    
    return organized
