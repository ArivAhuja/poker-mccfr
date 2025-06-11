"""
Suit Isomorphism Converter for OpenSpiel Poker States

This module converts OpenSpiel poker state strings to use canonical card representations
based on suit isomorphism principles.
"""

import re
from typing import List, Dict, Tuple, Optional


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
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 't', 'j', 'q', 'k', 'a']
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
    
    # Create (signature, original_suit) pairs and sort
    suit_pairs = [(sig, suit) for suit, sig in enumerate(signatures)]
    # Sort by: count (desc), highest rank (desc), sum of ranks (desc), original suit (asc)
    suit_pairs.sort(key=lambda x: (-x[0][0], -x[0][1], -x[0][2], x[1]))
    
    # Create permutation mapping
    permutation = {}
    for canonical_suit, (_, original_suit) in enumerate(suit_pairs):
        permutation[original_suit] = canonical_suit
    
    return permutation


def apply_suit_permutation(cards: List[str], permutation: Dict[int, int]) -> List[str]:
    """Apply suit permutation to cards."""
    canonical_cards = []
    
    for card in cards:
        rank, suit = parse_card(card)
        new_suit = permutation[suit]
        canonical_cards.append(card_to_string(rank, new_suit))
    
    return canonical_cards


def canonicalize_cards(private_cards: List[str], public_cards: List[str]) -> Tuple[List[str], List[str]]:
    """Convert cards to canonical form using suit isomorphism."""
    # Combine all cards to determine the permutation
    all_cards = private_cards + public_cards
    
    # Find the canonical permutation
    permutation = find_canonical_permutation(all_cards)
    
    # Apply to both private and public cards
    canonical_private = apply_suit_permutation(private_cards, permutation)
    canonical_public = apply_suit_permutation(public_cards, permutation)
    
    return canonical_private, canonical_public


def reconstruct_state_string(original_state: str, canonical_private: List[str], 
                           canonical_public: List[str]) -> str:
    """Reconstruct state string with canonical cards."""
    # Replace private cards
    private_str = ''.join(canonical_private)
    new_state = re.sub(
        r'\[Private: [^\]]*\]',
        f'[Private: {private_str}]',
        original_state
    )
    
    # Replace public cards
    if canonical_public:
        public_str = ''.join(canonical_public)
    else:
        public_str = ''
    
    new_state = re.sub(
        r'\[Public: [^\]]*\]',
        f'[Public: {public_str}]',
        new_state
    )
    
    return new_state


def apply_suit_isomorphism(state_string: str) -> str:
    """Apply suit isomorphism to OpenSpiel poker state string, returning canonical form."""
    # Extract cards from state
    private_cards, public_cards = extract_cards_from_state(state_string)
    
    # Apply suit isomorphism
    canonical_private, canonical_public = canonicalize_cards(private_cards, public_cards)
    
    # Reconstruct state string
    canonical_state = reconstruct_state_string(state_string, canonical_private, canonical_public)
    
    return canonical_state


# Example usage and test cases
if __name__ == "__main__":
    # Test case 1: Pocket pairs
    state1 = "[Round 0][Player: 1][Pot: 600][Money: 9900 9950 10000 10000 10000 10000][Private: 2d2c][Public: ][Sequences: ]"
    canonical1 = apply_suit_isomorphism(state1)
    print(f"Original:  {state1}")
    print(f"Canonical: {canonical1}")
    print()
    
    # Test case 2: Suited cards with public cards
    state2 = "[Round 1][Player: 0][Pot: 1200][Money: 9700 9750 10000 10000 10000 10000][Private: AhKh][Public: QhJhTh][Sequences: cc]"
    canonical2 = apply_suit_isomorphism(state2)
    print(f"Original:  {state2}")
    print(f"Canonical: {canonical2}")
    print()
    
    # Test case 3: Mixed suits
    state3 = "[Round 1][Player: 0][Pot: 1200][Money: 9700 9750 10000 10000 10000 10000][Private: AsKd][Public: QhJc9d][Sequences: cc]"
    canonical3 = apply_suit_isomorphism(state3)
    print(f"Original:  {state3}")
    print(f"Canonical: {canonical3}")