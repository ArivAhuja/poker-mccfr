import numpy as np
from sklearn_extra.cluster import KMedoids
from scipy.stats import wasserstein_distance
from phevaluator.evaluator import evaluate_cards
from collections import defaultdict
from suit_abstraction import canonicalize_cards
from itertools import combinations

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

hand_strength = defaultdict(list)

num_board_cards = 3
num_runout_cards = 5 - num_board_cards
num_hole_cards = 2

hero_cards = generate_cards(num_hole_cards, None)
for hero_card in hero_cards:
    board_cards = generate_cards(num_board_cards, hero_card)
    for board_card in board_cards:
        canonicalized_hero_card, canonicalized_board_card = canonicalize_cards(hero_card, board_card)
        key = (tuple(canonicalized_hero_card), tuple(canonicalized_board_card))
        if (tuple(canonicalized_hero_card), tuple(canonicalized_board_card)) in hand_strength:
            continue
        runout_cards = generate_cards(num_runout_cards, hero_card + board_card)
        winrates = []
        for runout_card in runout_cards:
            wins, ties, total = 0, 0, 0
            opponent_cards = generate_cards(num_hole_cards, hero_card + board_card + runout_card)
            for opponent_card in opponent_cards:
                hero_hand = hero_card + board_card + runout_card
                opponent_hand = opponent_card + board_card + runout_card
                hero_value = evaluate_cards(*hero_hand)
                opponent_value = evaluate_cards(*opponent_hand)
                wins += hero_value < opponent_value 
                ties += hero_value == opponent_value
                total += 1
            winrate = (wins + (ties * .5)) / total
            winrates.append(winrate)
        hand_strength[key] = np.histogram(winrates, bins=20)
            
        
            
            

            


                    


            