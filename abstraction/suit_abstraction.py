"""
Suit Isomorphism Converter for OpenSpiel Poker States

This module converts OpenSpiel poker state strings to use canonical card representations
based on suit isomorphism principles, combining private and public cards.
"""

import re
from typing import List, Dict, Tuple, Iterator, Optional
import itertools


def parse_card(card_str: str) -> Tuple[int, int]:
    """Parse card string (e.g. 'As', '2d') into (rank, suit) indices."""
    rank_map = {'a': 12, 'k': 11, 'q': 10, 'j': 9, 't': 8,
                '9': 7, '8': 6, '7': 5, '6': 4, '5': 3,
                '4': 2, '3': 1, '2': 0}
    suit_map = {'s': 0, 'h': 1, 'd': 2, 'c': 3}
    
    rank = rank_map[card_str[0].lower()]
    suit = suit_map[card_str[1].lower()]
    
    return rank, suit


def card_to_string(rank: int, suit: int) -> str:
    """Convert (rank, suit) indices back to card string."""
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['s', 'h', 'd', 'c']
    
    return ranks[rank] + suits[suit]


def extract_cards_from_state(state_str: str) -> Tuple[List[str], List[str]]:
    """Extract private and public cards from OpenSpiel state string."""
    # Extract private cards
    private_match = re.search(r'\[Private: ([^\]]*)\]', state_str)
    private_cards = []
    if private_match and private_match.group(1):
        private_str = private_match.group(1)
        # Split by space or no space between cards
        private_cards = re.findall(r'[2-9tjqka][shdc]', private_str.lower())
    
    # Extract public cards
    public_match = re.search(r'\[Public: ([^\]]*)\]', state_str)
    public_cards = []
    if public_match and public_match.group(1):
        public_str = public_match.group(1)
        public_cards = re.findall(r'[2-9tjqka][shdc]', public_str.lower())
    
    return private_cards, public_cards


def compute_suit_signature(cards: List[str]) -> List[Tuple[int, int, int]]:
    """Compute suit signature (count, highest_rank, rank_sum) for each suit."""
    suit_info = {0: [], 1: [], 2: [], 3: []}  # s, h, d, c
    
    for card in cards:
        rank, suit = parse_card(card)
        suit_info[suit].append(rank)
    
    signature = []
    for suit in range(4):
        ranks = suit_info[suit]
        if ranks:
            signature.append((len(ranks), max(ranks), sum(ranks)))
        else:
            signature.append((0, -1, 0))
    
    return signature


def find_canonical_permutation(cards: List[str]) -> Dict[int, int]:
    """Find suit permutation for lexicographically smallest representation."""
    if not cards:
        return {0: 0, 1: 1, 2: 2, 3: 3}
    
    # Get suit signatures
    signatures = compute_suit_signature(cards)
    
    # Group suits by their signatures
    signature_to_suits = {}
    for suit, sig in enumerate(signatures):
        sig_tuple = tuple(sig)  # Convert to tuple for use as dict key
        if sig_tuple not in signature_to_suits:
            signature_to_suits[sig_tuple] = []
        signature_to_suits[sig_tuple].append(suit)
    
    # Sort signature groups by signature value
    sorted_sig_groups = sorted(signature_to_suits.items(),
                              key=lambda x: (-x[0][0], -x[0][1], -x[0][2]))
    
    # Assign canonical suits
    permutation = {}
    canonical_suit = 0
    

    for signature, suits in sorted_sig_groups:
        # Within each signature group, assign canonical suits in a fixed order
        # Always assign to the lowest indices first (s before h before d before c)
        for suit in sorted(suits):
            permutation[suit] = canonical_suit
            canonical_suit += 1
    
    return permutation


def apply_suit_permutation(cards: List[str], permutation: Dict[int, int]) -> List[str]:
    """Apply suit permutation to cards."""
    canonical_cards = []
    
    for card in cards:
        rank, suit = parse_card(card)
        new_suit = permutation[suit]
        canonical_cards.append(card_to_string(rank, new_suit))
    
    return canonical_cards


def sort_cards_by_rank(cards: List[str]) -> List[str]:
    """Sort cards by rank in descending order, then by suit in ascending order."""
    # Parse cards and sort by rank (descending) then suit (ascending)
    card_tuples = [(parse_card(card), card) for card in cards]
    # Sort by rank descending (-rank) and suit ascending (suit)
    card_tuples.sort(key=lambda x: (-x[0][0], x[0][1]))
    
    # Return sorted cards
    return [card_to_string(rank, suit) for (rank, suit), _ in card_tuples]


def canonicalize_and_combine_cards(private_cards: List[str], public_cards: List[str]) -> List[str]:
    """Convert cards to canonical form using suit isomorphism and combine them."""
    # Combine all cards to determine the permutation
    all_cards = private_cards + public_cards
    
    # Find the canonical permutation
    permutation = find_canonical_permutation(all_cards)
    
    # Apply to all cards
    canonical_cards = apply_suit_permutation(all_cards, permutation)
    
    # Sort by rank (descending)
    sorted_cards = sort_cards_by_rank(canonical_cards)
    
    return sorted_cards


def get_cards_string(state_string: str) -> str:
    """Extract, canonicalize, and return just the combined cards string."""
    # Extract cards from state
    private_cards, public_cards = extract_cards_from_state(state_string)
    
    # Canonicalize and combine
    combined_cards = canonicalize_and_combine_cards(private_cards, public_cards)
    
    # Return as concatenated string
    return ''.join(combined_cards)


def reconstruct_state_with_combined_cards(original_state: str, combined_cards: List[str]) -> str:
    """Reconstruct state string with combined canonical cards."""
    # Create the combined cards string
    cards_str = ''.join(combined_cards)
    
    # Replace [Private: ...][Public: ...] with [Cards: ...]
    # First, find and replace the Private section
    new_state = re.sub(
        r'\[Private: [^\]]*\]',
        f'[Cards: {cards_str}]',
        original_state
    )
    
    # Then remove the Public section
    new_state = re.sub(
        r'\[Public: [^\]]*\]',
        '',
        new_state
    )
    
    return new_state


def apply_suit_isomorphism_combined(state_string: str) -> str:
    """Apply suit isomorphism and combine cards in OpenSpiel poker state string."""
    # Extract cards from state
    private_cards, public_cards = extract_cards_from_state(state_string)
    
    # Canonicalize and combine
    combined_cards = canonicalize_and_combine_cards(private_cards, public_cards)
    
    # Reconstruct state string
    canonical_state = reconstruct_state_with_combined_cards(state_string, combined_cards)
    
    return canonical_state

# Example usage and test cases
if __name__ == "__main__":
    # Test case 1: Example from the prompt
    state1 = "[Round 0][Player: 1][Pot: 600][Money: 9900 9950 10000 10000 10000 10000][Private: 2s4s][Public: AsKs3s][Sequences: ]"
    canonical1 = apply_suit_isomorphism_combined(state1)
    cards1 = get_cards_string(state1)
    print(f"Original:  {state1}")
    print(f"Canonical: {canonical1}")
    print(f"Cards only: {cards1}")
    print()
    
    # Test case 2: Pocket pairs
    state2 = "[Round 0][Player: 1][Pot: 600][Money: 9900 9950 10000 10000 10000 10000][Private: 2d2c][Public: ][Sequences: ]"
    canonical2 = apply_suit_isomorphism_combined(state2)
    cards2 = get_cards_string(state2)
    print(f"Original:  {state2}")
    print(f"Canonical: {canonical2}")
    print(f"Cards only: {cards2}")
    print()
    
    # Test case 3: Suited cards with public cards
    state3 = "[Round 1][Player: 0][Pot: 1200][Money: 9700 9750 10000 10000 10000 10000][Private: AhKh][Public: QhJhTh][Sequences: cc]"
    canonical3 = apply_suit_isomorphism_combined(state3)
    cards3 = get_cards_string(state3)
    print(f"Original:  {state3}")
    print(f"Canonical: {canonical3}")
    print(f"Cards only: {cards3}")
    print()

    # Test case 4: Mixed suits
    state4 = "[Round 1][Player: 0][Pot: 1200][Money: 9700 9750 10000 10000 10000 10000][Private: KsAd][Public: QhJc9d][Sequences: cc]"
    canonical4 = apply_suit_isomorphism_combined(state4)
    cards4 = get_cards_string(state4)
    print(f"Original:  {state4}")
    print(f"Canonical: {canonical4}")
    print(f"Cards only: {cards4}")
    print()
    
    # Test case 5: Two states that should have the same canonical form
    print("Testing suit isomorphism equivalence:")
    state5a = "[Round 0][Player: 1][Pot: 600][Money: 9900 9950 10000 10000 10000 10000][Private: 2d4s][Public: AsKs2s][Sequences: ]"
    state5b = "[Round 0][Player: 1][Pot: 600][Money: 9900 9950 10000 10000 10000 10000][Private: 2s4s][Public: AsKs2h][Sequences: ]"
    
    canonical5a = apply_suit_isomorphism_combined(state5a)
    canonical5b = apply_suit_isomorphism_combined(state5b)
    cards5a = get_cards_string(state5a)
    cards5b = get_cards_string(state5b)
    
    print(f"State A Original:  {state5a}")
    print(f"State A Canonical: {canonical5a}")
    print(f"State A Cards: {cards5a}")
    print()
    print(f"State B Original:  {state5b}")
    print(f"State B Canonical: {canonical5b}")
    print(f"State B Cards: {cards5b}")
    print()
    print(f"Cards match: {cards5a == cards5b}")

    hole_cards_list = canolical_hands(2, None)
    hand_strength = {}

    
