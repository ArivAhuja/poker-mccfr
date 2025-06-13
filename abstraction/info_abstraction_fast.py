import numpy as np
from sklearn_extra.cluster import KMedoids
from scipy.stats import wasserstein_distance
from phevaluator.evaluator import evaluate_cards
from collections import defaultdict
from suit_abstraction import canonicalize_cards
from itertools import combinations
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pickle
from functools import partial

def generate_cards(num_cards, exclude_cards=None):
    """Generate all possible combinations of num_cards from a standard deck"""
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['h', 'd', 'c', 's']
    deck = [rank + suit for rank in ranks for suit in suits]
    
    if exclude_cards:
        deck = [card for card in deck if card not in exclude_cards]
    
    return [list(combo) for combo in combinations(deck, num_cards)]

def process_single_hand(args):
    """Process a single hero/board combination"""
    hero_card, board_card, num_runout_cards, num_hole_cards = args
    
    # Canonicalize cards
    canonicalized_hero_card, canonicalized_board_card = canonicalize_cards(hero_card, board_card)
    key = (tuple(canonicalized_hero_card), tuple(canonicalized_board_card))
    
    # Generate runouts
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
            
        winrate = (wins + (ties * 0.5)) / total
        winrates.append(winrate)
    
    histogram = np.histogram(winrates, bins=20)
    return key, histogram

def main():
    num_board_cards = 3
    num_runout_cards = 5 - num_board_cards
    num_hole_cards = 2
    
    # Generate all hero cards
    hero_cards = generate_cards(num_hole_cards, None)
    
    # Prepare all tasks
    tasks = []
    seen_canonical = set()
    
    print("Preparing tasks...")
    for hero_card in tqdm(hero_cards, desc="Hero cards"):
        board_cards = generate_cards(num_board_cards, hero_card)
        for board_card in board_cards:
            # Check if we've seen this canonical form
            canon_hero, canon_board = canonicalize_cards(hero_card, board_card)
            canon_key = (tuple(canon_hero), tuple(canon_board))
            
            if canon_key not in seen_canonical:
                seen_canonical.add(canon_key)
                tasks.append((hero_card, board_card, num_runout_cards, num_hole_cards))
    
    print(f"\nTotal unique canonical forms: {len(tasks)}")
    print(f"Using {cpu_count()} CPU cores")
    
    # Process in parallel with progress bar
    hand_strength = {}
    
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(process_single_hand, tasks),
            total=len(tasks),
            desc="Processing hands"
        ))
    
    # Convert results to dictionary
    for key, histogram in results:
        hand_strength[key] = histogram
    
    # Save results
    with open('hand_strength_histograms.pkl', 'wb') as f:
        pickle.dump(hand_strength, f)
    
    print(f"\nCompleted! Processed {len(hand_strength)} unique canonical forms")
    print("Results saved to hand_strength_histograms.pkl")
    
    return hand_strength

def main_sampled(sample_rate=0.1):
    """Faster version using sampling instead of exhaustive enumeration"""
    num_board_cards = 3
    num_runout_cards = 5 - num_board_cards
    num_hole_cards = 2
    
    # Generate all hero cards
    hero_cards = generate_cards(num_hole_cards, None)
    
    # Sample hero cards
    sampled_heroes = np.random.choice(len(hero_cards), 
                                      int(len(hero_cards) * sample_rate), 
                                      replace=False)
    
    tasks = []
    seen_canonical = set()
    
    print(f"Sampling {sample_rate*100}% of hero cards...")
    for idx in tqdm(sampled_heroes, desc="Preparing sampled tasks"):
        hero_card = hero_cards[idx]
        board_cards = generate_cards(num_board_cards, hero_card)
        
        # Sample board cards too
        sampled_boards = np.random.choice(len(board_cards), 
                                          max(1, int(len(board_cards) * sample_rate)), 
                                          replace=False)
        
        for board_idx in sampled_boards:
            board_card = board_cards[board_idx]
            canon_hero, canon_board = canonicalize_cards(hero_card, board_card)
            canon_key = (tuple(canon_hero), tuple(canon_board))
            
            if canon_key not in seen_canonical:
                seen_canonical.add(canon_key)
                tasks.append((hero_card, board_card, num_runout_cards, num_hole_cards))
    
    print(f"\nProcessing {len(tasks)} sampled canonical forms...")
    print(f"Using {cpu_count()} CPU cores")
    
    # Process in parallel
    hand_strength = {}
    
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(process_single_hand, tasks),
            total=len(tasks),
            desc="Processing sampled hands"
        ))
    
    for key, histogram in results:
        hand_strength[key] = histogram
    
    # Save results
    with open('hand_strength_histograms_sampled.pkl', 'wb') as f:
        pickle.dump(hand_strength, f)
    
    print(f"\nCompleted! Processed {len(hand_strength)} sampled canonical forms")
    return hand_strength

if __name__ == "__main__":
    # For full computation (will take very long)
    hand_strength = main()
    
    # For faster sampled version (10% sampling)
    # hand_strength = main_sampled(sample_rate=0.1)
    
    # To use even faster sampling
    # hand_strength = main_sampled(sample_rate=0.01)  # 1% sampling