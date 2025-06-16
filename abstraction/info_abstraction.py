from sklearn_extra.cluster import KMedoids
from scipy.stats import wasserstein_distance
import numpy as np
from suit_abstraction import canonicalize_cards

hand_strength_distributions = {}

num_board_cards = 3
num_runout_cards = 2
hero_hands = hands(2, None)
for hero_hand in hero_hands:
    board_card_list = hands(num_board_cards, hero_hand)
    for board_cards in board_cards_list:
        opponent_hands = weighted_canolical_hands(2, hero_hand + board_cards)
        for opponent_cards in opponent_hands:
            runout_cards = runout_cards(hole_cards + board_cards + opp)
            cards = sample
            for 
            