"""
Suit Isomorphism Converter for OpenSpiel Poker States

This module converts OpenSpiel poker state strings to use canonical card representations
based on suit isomorphism principles, maintaining separate private and public cards.
"""

import re
from typing import List, Dict, Tuple


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
    """Convert (rank, suit) indices back to card string with capital ranks."""
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
        sig_tuple = tuple(sig)
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
        for suit in sorted(suits):
            permutation[suit] = canonical_suit
            canonical_suit += 1
    
    return permutation


def apply_suit_permutation(cards: List[str], permutation: Dict[int, int]) -> List[str]:
    """Apply suit permutation to cards, keeping ranks unchanged."""
    canonical_cards = []
    
    for card in cards:
        rank, suit = parse_card(card)
        new_suit = permutation[suit]
        canonical_cards.append(card_to_string(rank, new_suit))
    
    return canonical_cards


def sort_cards_by_rank(cards: List[str]) -> List[str]:
    """Sort cards by rank in descending order, then by suit in ascending order."""
    card_tuples = [(parse_card(card), card) for card in cards]
    # Sort by rank descending (-rank) and suit ascending (suit)
    card_tuples.sort(key=lambda x: (-x[0][0], x[0][1]))
    
    return [card_to_string(rank, suit) for (rank, suit), _ in card_tuples]


def canonicalize_cards(private_cards: List[str], public_cards: List[str]) -> Tuple[List[str], List[str]]:
    """
    Convert cards to canonical form using suit isomorphism.
    
    Args:
        private_cards: List of private card strings
        public_cards: List of public card strings
    
    Returns:
        Tuple of (canonicalized_private_cards, canonicalized_public_cards)
    """
    # Combine all cards to determine the permutation
    all_cards = private_cards + public_cards
    
    # Find the canonical permutation based on all cards
    permutation = find_canonical_permutation(all_cards)
    
    # Apply to private and public cards separately
    canonical_private = apply_suit_permutation(private_cards, permutation)
    canonical_public = apply_suit_permutation(public_cards, permutation)
    
    # Sort each group by rank (descending)
    sorted_private = sort_cards_by_rank(canonical_private)
    sorted_public = sort_cards_by_rank(canonical_public)
    
    return sorted_private, sorted_public


def apply_suit_isomorphism(state_string: str) -> str:
    """Apply suit isomorphism to OpenSpiel poker state string, keeping Private and Public separate."""
    # Extract cards from state
    private_cards, public_cards = extract_cards_from_state(state_string)
    
    # Canonicalize cards
    canonical_private, canonical_public = canonicalize_cards(private_cards, public_cards)
    
    # Create the card strings
    private_str = ''.join(canonical_private)
    public_str = ''.join(canonical_public)
    
    # Replace Private section
    new_state = re.sub(
        r'\[Private: [^\]]*\]',
        f'[Private: {private_str}]',
        state_string
    )
    
    # Replace Public section
    new_state = re.sub(
        r'\[Public: [^\]]*\]',
        f'[Public: {public_str}]',
        new_state
    )
    
    return new_state


# Example usage and test cases
if __name__ == "__main__":
    # Test case 1: Basic example
    state1 = "[Round 0][Player: 1][Pot: 600][Money: 9900 9950 10000 10000 10000 10000][Private: 2s4s][Public: AsKs3s][Sequences: ]"
    canonical1 = apply_suit_isomorphism(state1)
    print(f"Original:  {state1}")
    print(f"Canonical: {canonical1}")
    print()
    
    # Test case 2: Direct function call with simple example
    print("Test case 2 - Direct function call:")
    private = ['ks', 'ad']
    public = ['qh']
    print(f"Input private: {private}")
    print(f"Input public: {public}")
    
    # Show permutation
    all_cards = private + public
    perm = find_canonical_permutation(all_cards)
    print(f"Suit permutation: {perm}")
    
    canon_private, canon_public = canonicalize_cards(private, public)
    print(f"Canonical private: {canon_private}")
    print(f"Canonical public: {canon_public}")
    print()
    
    # Test case 3: Mixed suits with detailed output
    state3 = "[Round 1][Player: 0][Pot: 1200][Money: 9700 9750 10000 10000 10000 10000][Private: KsAd][Public: QhJc9d][Sequences: cc]"
    
    # Extract and show step by step
    private3, public3 = extract_cards_from_state(state3)
    print(f"Test case 3 - Mixed suits:")
    print(f"Original state: {state3}")
    print(f"Extracted private: {private3}")
    print(f"Extracted public: {public3}")
    
    # Canonicalize
    canon_priv3, canon_pub3 = canonicalize_cards(private3, public3)
    print(f"Canonical private: {canon_priv3}")
    print(f"Canonical public: {canon_pub3}")
    
    # Apply to full state
    canonical3 = apply_suit_isomorphism(state3)
    print(f"Final canonical: {canonical3}")
    print()
    
    # Test case 4: Empty public cards
    state4 = "[Round 0][Player: 1][Pot: 600][Money: 9900 9950 10000 10000 10000 10000][Private: AsKh][Public: ][Sequences: ]"
    canonical4 = apply_suit_isomorphism(state4)
    print(f"Original:  {state4}")
    print(f"Canonical: {canonical4}")
    print()
    
    # Test case 6: Verify the specific example
    print("Test case 6 - Verifying suit isomorphism (ranks should never change):")
    test_state = "[Round 1][Player: 0][Pot: 1200][Money: 9700 9750 10000 10000 10000 10000][Private: KsAd][Public: QhJc9d][Sequences: cc]"
    
    # Manual step-by-step
    priv_cards = ['ks', '2d']
    pub_cards = ['qh', '2c', '2s']
    
    print(f"Original: Private={priv_cards}, Public={pub_cards}")
    
    # The permutation should only change suits, never ranks!
    # K should stay K, A should stay A, Q should stay Q, etc.
    canonical_priv, canonical_pub = canonicalize_cards(priv_cards, pub_cards)
    
    print(f"After canonicalization:")
    print(f"  Private cards: {canonical_priv} (should still have K and A)")
    print(f"  Public cards: {canonical_pub} (should still have Q, J, and 9)")
    
    # Show the full state transformation
    canonical_state = apply_suit_isomorphism(test_state)
    print(f"\nFull state transformation:")
    print(f"Original:  {test_state}")
    print(f"Canonical: {canonical_state}")