import numpy as np
from sklearn_extra.cluster import KMedoids
from scipy.stats import wasserstein_distance
from phevaluator import evaluate_cards
from collections import defaultdict
from itertools import combinations
import sys

# Import the HandIsomorphism class (assuming it's in hand_isomorphism.py)
from hand_isomorphism import HandIsomorphism

def card_string_to_tuple(card_str):
    """Convert card string format to tuple format
    '2h' -> (0, 1) where 2=0, ..., A=12 and h=1
    """
    rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, 
                '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
    suit_map = {'s': 0, 'h': 1, 'c': 2, 'd': 3}
    
    rank = rank_map[card_str[0]]
    suit = suit_map[card_str[1]]
    return (rank, suit)

def card_tuple_to_string(card_tuple):
    """Convert tuple format back to string format for phevaluator
    (0, 1) -> '2h'
    """
    rank_map = {0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8',
                7: '9', 8: 'T', 9: 'J', 10: 'Q', 11: 'K', 12: 'A'}
    suit_map = {0: 's', 1: 'h', 2: 'c', 3: 'd'}
    
    rank, suit = card_tuple
    return rank_map[rank] + suit_map[suit]

def generate_cards(num_cards, exclude_cards=None):
    """Generate all possible combinations of num_cards from a standard deck"""
    # All cards in a standard deck
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['h', 'd', 'c', 's']
    deck = [rank + suit for rank in ranks for suit in suits]
    
    # Remove excluded cards
    if exclude_cards:
        deck = [card for card in deck if card not in exclude_cards]
    
    # Generate all combinations of num_cards from remaining deck
    return [list(combo) for combo in combinations(deck, num_cards)]

def calculate_hand_strength_distribution():
    """Calculate hand strength distribution for all canonical flop combinations"""
    
    # Initialize hand isomorphism calculator
    iso = HandIsomorphism()
    
    hand_strength = {}
    
    num_board_cards = 3
    num_runout_cards = 5 - num_board_cards  # 2 (turn + river)
    num_hole_cards = 2
    
    # Generate all possible hole cards
    hero_cards_list = generate_cards(num_hole_cards, None)
    
    total_processed = 0
    canonical_count = 0
    
    for hero_cards_str in hero_cards_list:
        # Generate all possible flops excluding hero cards
        board_cards_list = generate_cards(num_board_cards, hero_cards_str)
        
        for board_cards_str in board_cards_list:
            # Convert to tuple format for canonicalization
            hero_cards_tuple = [card_string_to_tuple(c) for c in hero_cards_str]
            board_cards_tuple = [card_string_to_tuple(c) for c in board_cards_str]
            
            # Get canonical form
            canonical_hero, canonical_board = iso.canonicalize_with_private_public(
                hero_cards_tuple, board_cards_tuple
            )
            
            # Create key
            key = (tuple(canonical_hero), tuple(canonical_board))
            
            # Skip if already processed
            if key in hand_strength:
                total_processed += 1
                continue
            
            # Calculate hand strength distribution
            runout_cards_list = generate_cards(num_runout_cards, hero_cards_str + board_cards_str)
            winrates = []
            
            for runout_cards_str in runout_cards_list:
                wins, ties, total = 0, 0, 0
                
                # Generate all possible opponent hands
                opponent_cards_list = generate_cards(
                    num_hole_cards, 
                    hero_cards_str + board_cards_str + runout_cards_str
                )
                
                for opponent_cards_str in opponent_cards_list:
                    # Evaluate hands using phevaluator
                    hero_hand = hero_cards_str + board_cards_str + runout_cards_str
                    opponent_hand = opponent_cards_str + board_cards_str + runout_cards_str
                    
                    hero_value = evaluate_cards(*hero_hand)
                    opponent_value = evaluate_cards(*opponent_hand)
                    
                    # Lower value is better in phevaluator
                    if hero_value < opponent_value:
                        wins += 1
                    elif hero_value == opponent_value:
                        ties += 1
                    total += 1
                
                # Calculate winrate for this runout
                winrate = (wins + (ties * 0.5)) / total
                winrates.append(winrate)
            
            # Store histogram of winrates
            hist, bin_edges = np.histogram(winrates, bins=20, range=(0, 1))
            hand_strength[key] = (hist, bin_edges)
            
            canonical_count += 1
            total_processed += 1
            
            # Progress indicator
            if canonical_count % 1000 == 0:
                print(f"Processed {total_processed:,} hands, {canonical_count:,} canonical forms")
    
    print(f"\nFinal results:")
    print(f"Total hands processed: {total_processed:,}")
    print(f"Unique canonical forms: {canonical_count:,}")
    print(f"Expected canonical forms: 1,286,792")
    
    return hand_strength

def estimate_memory_usage():
    """Estimate memory usage for the hand strength table"""
    
    # Key size estimation
    # Each (rank, suit) tuple: ~56 bytes (2 ints in a tuple)
    # Key structure: ((rank, suit), (rank, suit)), ((rank, suit), (rank, suit), (rank, suit))
    # Approximately: 2 * 56 + 3 * 56 + tuple overhead ≈ 300-400 bytes per key
    
    # Value size estimation
    # Histogram: 20 bins (int64) + 21 bin edges (float64)
    # 20 * 8 + 21 * 8 = 328 bytes per value
    
    # Total per entry: ~700 bytes
    # For 1,286,792 entries: ~900 MB
    
    key_example = ((12, 0), (11, 0)), ((10, 0), (9, 1), (8, 2))
    
    print("Memory usage estimation:")
    print(f"Example key: {key_example}")
    print(f"Estimated size per key: ~350 bytes")
    print(f"Estimated size per value (histogram): ~330 bytes")
    print(f"Total per entry: ~680 bytes")
    print(f"For 1,286,792 canonical flops: ~{1286792 * 680 / 1024 / 1024:.0f} MB")
    print(f"Actual usage will be higher due to Python overhead")

def lookup_hand_strength(hand_strength_table, hero_cards_str, board_cards_str):
    """Look up hand strength for a specific hand"""
    iso = HandIsomorphism()
    
    # Convert to tuple format
    hero_cards_tuple = [card_string_to_tuple(c) for c in hero_cards_str]
    board_cards_tuple = [card_string_to_tuple(c) for c in board_cards_str]
    
    # Get canonical form
    canonical_hero, canonical_board = iso.canonicalize_with_private_public(
        hero_cards_tuple, board_cards_tuple
    )
    
    # Create key
    key = (tuple(canonical_hero), tuple(canonical_board))
    
    # Look up in table
    if key in hand_strength_table:
        hist, bin_edges = hand_strength_table[key]
        mean_winrate = sum(hist * (bin_edges[:-1] + bin_edges[1:]) / 2) / sum(hist)
        return mean_winrate, hist, bin_edges
    else:
        return None, None, None

# Example usage
if __name__ == "__main__":
    # Estimate memory usage
    estimate_memory_usage()
    
    # Small example (not full calculation due to time)
    print("\nSmall example calculation:")
    iso = HandIsomorphism()
    
    # Example hand
    hero = ['Ah', 'Kh']
    board = ['Qh', 'Jd', 'Tc']
    
    # Convert and canonicalize
    hero_tuple = [card_string_to_tuple(c) for c in hero]
    board_tuple = [card_string_to_tuple(c) for c in board]
    
    canonical_hero, canonical_board = iso.canonicalize_with_private_public(
        hero_tuple, board_tuple
    )
    
    print(f"Original: {hero} | {board}")
    print(f"Canonical: {canonical_hero} | {canonical_board}")
    
    hand_strength_table = calculate_hand_strength_distribution()