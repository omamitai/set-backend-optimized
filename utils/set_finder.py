from itertools import combinations

def is_set(cards):
    """
    Determine if a group of cards forms a valid set.
    For each feature (Count, Color, Fill, Shape), the values must be either all the same or all different.

    Args:
        cards (list of dict): List of card feature dictionaries.

    Returns:
        bool: True if the cards form a valid set, False otherwise.
    """
    # Check all features in one pass for efficiency
    for feature in ('Count', 'Color', 'Fill', 'Shape'):
        # Use a set for O(1) lookups and unique value counting
        # This is faster than using list.count() for each value
        unique_values = {card[feature] for card in cards}
        
        # For a valid set: must have either 1 unique value (all same) or 3 unique values (all different)
        if len(unique_values) not in (1, 3):
            return False
    
    return True

def find_sets(card_df):
    """
    Find all valid sets from the card DataFrame by checking every combination of three cards.

    Args:
        card_df (pandas.DataFrame): DataFrame containing card features.

    Returns:
        list: List of dictionaries with set details.
    """
    # Early return if there are fewer than 3 cards (not enough to form a set)
    if len(card_df) < 3:
        return []
    
    # Convert DataFrame to list of dictionaries for faster access
    # This avoids slow DataFrame indexing operations in the loop
    cards_list = card_df.to_dict('records')
    
    # Pre-allocate result list
    sets_found = []
    
    # Define features to extract once to avoid repeated string creation
    features_to_extract = ('Count', 'Color', 'Fill', 'Shape', 'Coordinates')
    
    # Iterate through all possible combinations of 3 cards
    # Using index combinations is faster than using DataFrame.iterrows()
    for idx_combo in combinations(range(len(cards_list)), 3):
        # Extract the three cards for this combination
        combo_cards = [cards_list[idx] for idx in idx_combo]
        
        # Check if these cards form a valid set
        if is_set(combo_cards):
            # Create set info with minimal dictionary creation
            cards_data = []
            for card in combo_cards:
                # Extract just the needed features for each card
                card_data = {feature: card[feature] for feature in features_to_extract}
                cards_data.append(card_data)
            
            set_info = {
                'set_indices': idx_combo,
                'cards': cards_data
            }
            sets_found.append(set_info)
    
    return sets_found
