import itertools
from typing import List, Tuple, Set
from collections import defaultdict
import math

# Card representation: (rank, suit) where:
# - rank: 0=2, 1=3, ..., 8=T, 9=J, 10=Q, 11=K, 12=A
# - suit: 0=♠, 1=♥, 2=♣, 3=♦ (can be any mapping, will be remapped in canonical form)
Card = Tuple[int, int]

class HandIsomorphism:
    def __init__(self, num_ranks=13, num_suits=4):
        self.num_ranks = num_ranks
        self.num_suits = num_suits
        
        # Precompute binomial coefficients
        self.binomial_cache = {}
        for n in range(max(num_ranks + 1, 20)):
            for k in range(n + 1):
                self.binomial_cache[(n, k)] = self._compute_binomial(n, k)
    
    def _compute_binomial(self, n: int, k: int) -> int:
        """Compute binomial coefficient n choose k"""
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result
    
    def _binomial(self, n: int, k: int) -> int:
        """Get binomial coefficient from cache"""
        return self.binomial_cache.get((n, k), 0)
    
    def index_rank_set(self, ranks: List[int], M: int) -> int:
        """Index M-rank set using colex function"""
        if not ranks:
            return 0
        
        # Sort in decreasing order
        sorted_ranks = sorted(ranks, reverse=True)
        
        # Apply formula from equation (3)
        index = 0
        for i in range(M):
            index += self._binomial(sorted_ranks[i], M - i)
        
        return index
    
    def unindex_rank_set(self, idx: int, M: int, N: int) -> List[int]:
        """Unindex to get M-rank set"""
        if M == 0:
            return []
        
        result = []
        remaining_idx = idx
        
        for i in range(M):
            # Find largest x such that C(x, M-i) <= remaining_idx
            x = M - i - 1
            while x < N and self._binomial(x + 1, M - i) <= remaining_idx:
                x += 1
            
            result.append(x)
            remaining_idx -= self._binomial(x, M - i)
        
        return result
    
    def index_rank_group(self, groups: List[List[int]], sizes: List[int], used: Set[int] = None) -> int:
        """Index rank groups with sizes M1, M2, ..., MK"""
        if not groups:
            return 0
        
        if used is None:
            used = set()
        
        # Index first group
        A1 = groups[0]
        M1 = sizes[0]
        
        # Shift ranks down by counting used slots below each rank
        shifted_ranks = []
        for rank in A1:
            shift = sum(1 for u in used if u < rank)
            shifted_ranks.append(rank - shift)
        
        # Get index for first set
        idx = self.index_rank_set(shifted_ranks, M1)
        
        # Recursively index remaining groups
        if len(groups) > 1:
            new_used = used.union(set(A1))
            next_idx = self.index_rank_group(groups[1:], sizes[1:], new_used)
            group_size = self._binomial(self.num_ranks - len(used), M1)
            idx += group_size * next_idx
        
        return idx
    
    def unindex_rank_group(self, idx: int, sizes: List[int], used: Set[int] = None) -> List[List[int]]:
        """Unindex to get rank groups"""
        if not sizes:
            return []
        
        if used is None:
            used = set()
        
        M1 = sizes[0]
        N = self.num_ranks - len(used)
        
        # Compute size of first group
        group_size = self._binomial(N, M1)
        
        # Split index
        this_idx = idx % group_size
        next_idx = idx // group_size
        
        # Unindex first set
        shifted_ranks = self.unindex_rank_set(this_idx, M1, N)
        
        # Unshift ranks
        A1 = []
        available = sorted([r for r in range(self.num_ranks) if r not in used])
        for shifted_rank in shifted_ranks:
            A1.append(available[shifted_rank])
        
        # Recursively unindex remaining groups
        result = [A1]
        if len(sizes) > 1:
            new_used = used.union(set(A1))
            remaining = self.unindex_rank_group(next_idx, sizes[1:], new_used)
            result.extend(remaining)
        
        return result
    
    def multiset_colex(self, indices: List[int], sizes: List[int]) -> int:
        """Combine indices using multiset colex when suits have same configuration"""
        if not indices:
            return 0
        
        # Sort indices in decreasing order
        sorted_indices = sorted(enumerate(indices), key=lambda x: x[1], reverse=True)
        
        result = 0
        for i, (suit_idx, group_idx) in enumerate(sorted_indices):
            # Add current index
            result += self._binomial(group_idx + i, i + 1)
        
        return result
    
    def get_suit_configuration(self, cards: List[Card], round_sizes: List[int]) -> dict:
        """Get suit configuration for each suit"""
        # Group cards by suit and round
        suit_groups = defaultdict(lambda: [[] for _ in round_sizes])
        
        round_start = 0
        for round_idx, round_size in enumerate(round_sizes):
            round_cards = cards[round_start:round_start + round_size]
            for rank, suit in round_cards:
                suit_groups[suit][round_idx].append(rank)
            round_start += round_size
        
        # Create configuration
        config = {}
        for suit in range(self.num_suits):
            config[suit] = tuple(len(suit_groups[suit][i]) for i in range(len(round_sizes)))
        
        return config, suit_groups
    
    def canonicalize_hand(self, cards: List[Card], round_sizes: List[int] = None) -> List[Card]:
        """Convert hand to canonical form following the paper's algorithm
        
        IMPORTANT: Within each suit/round group, cards are ordered by DECREASING rank.
        This matches the paper's examples (e.g., A♠2♠ not 2♠A♠) and aligns with
        the colex indexing approach. This ordering is critical for dictionary key
        compatibility between implementations.
        """
        if round_sizes is None:
            round_sizes = [len(cards)]
        
        # Get suit configurations
        config, suit_groups = self.get_suit_configuration(cards, round_sizes)
        
        # Group suits by configuration
        config_to_suits = defaultdict(list)
        for suit, cfg in config.items():
            config_to_suits[cfg].append(suit)
        
        # Sort configurations lexicographically (descending)
        sorted_configs = sorted(config_to_suits.keys(), reverse=True)
        
        # Create suit mapping and compute group indices
        suit_info = []  # (old_suit, new_suit, group_index)
        new_suit = 0
        
        for cfg in sorted_configs:
            suits = config_to_suits[cfg]
            
            if len(suits) == 1:
                # No ties
                suit = suits[0]
                groups = [g for g in suit_groups[suit] if g]
                sizes = [len(g) for g in groups]
                idx = self.index_rank_group(groups, sizes) if groups else 0
                suit_info.append((suit, new_suit, idx, groups))
                new_suit += 1
            else:
                # Break ties using group indices
                suit_data = []
                for suit in suits:
                    groups = [g for g in suit_groups[suit] if g]
                    sizes = [len(g) for g in groups]
                    idx = self.index_rank_group(groups, sizes) if groups else 0
                    suit_data.append((suit, idx, groups))
                
                # Sort by index (descending) to break ties
                suit_data.sort(key=lambda x: x[1], reverse=True)
                
                for suit, idx, groups in suit_data:
                    suit_info.append((suit, new_suit, idx, groups))
                    new_suit += 1
        
        # Build canonical hand with proper ordering
        suit_map = {old: new for old, new, _, _ in suit_info}
        canonical = []
        
        # Process each round
        round_start = 0
        for round_idx, round_size in enumerate(round_sizes):
            # Collect cards for this round by new suit
            round_cards_by_suit = defaultdict(list)
            
            for i in range(round_start, round_start + round_size):
                rank, old_suit = cards[i]
                new_suit = suit_map[old_suit]
                round_cards_by_suit[new_suit].append(rank)
            
            # Add cards in new suit order
            for new_suit in sorted(round_cards_by_suit.keys()):
                ranks = round_cards_by_suit[new_suit]
                # Sort ranks in DECREASING order (matching the paper's colex approach)
                for rank in sorted(ranks, reverse=True):
                    canonical.append((rank, new_suit))
            
            round_start += round_size
        
        return canonical
    
    def canonicalize_with_private_public(self, private: List[Card], public: List[Card]) -> Tuple[List[Card], List[Card]]:
        """Canonicalize keeping private and public cards separate"""
        # Combine for canonicalization
        all_cards = private + public
        round_sizes = [len(private), len(public)]
        
        # Get canonical form
        canonical = self.canonicalize_hand(all_cards, round_sizes)
        
        # Split back
        canonical_private = canonical[:len(private)]
        canonical_public = canonical[len(private):]
        
        return canonical_private, canonical_public


def test_canonical_hands():
    """Test that we generate the correct number of canonical hands
    
    NOTE: The canonical form orders cards within each suit/round by DECREASING rank.
    This is important for dictionary key compatibility - different orderings will
    produce different "canonical" forms and incompatible dictionary keys!
    """
    iso = HandIsomorphism()
    
    # Generate all possible hands of different sizes
    all_cards = [(rank, suit) for rank in range(13) for suit in range(4)]
    
    for num_cards in range(1, 6):
        canonical_hands = set()
        
        # Generate all possible hands
        for hand in itertools.combinations(all_cards, num_cards):
            # Convert to canonical form
            canonical = iso.canonicalize_hand(list(hand))
            # Convert to tuple for hashing
            canonical_tuple = tuple(canonical)
            canonical_hands.add(canonical_tuple)
        
        print(f"{num_cards} cards: {len(canonical_hands)} canonical hands")
        
        # Known values for verification
        expected = {
            1: 13,      # 13 ranks
            2: 169,     # 13 pairs + 13*12/2 * 2 (suited/offsuit) = 13 + 78*2 = 169
            3: 1755,    # Calculated value
            4: 16432,   # Calculated value
            5: 134459   # Calculated value
        }
        
        if num_cards in expected:
            if len(canonical_hands) == expected[num_cards]:
                print(f"  ✓ Correct!")
            else:
                print(f"  ✗ Expected {expected[num_cards]}")


def example_usage():
    """Example usage of the hand isomorphism algorithm"""
    iso = HandIsomorphism()
    
    # Example 1: Simple hand showing rank ordering
    hand1 = [(0, 0), (12, 0)]  # 2♠ A♠
    canonical1 = iso.canonicalize_hand(hand1)
    print(f"Hand: 2♠ A♠ = {hand1}")
    print(f"Canonical: {canonical1} (Note: A before 2, decreasing rank order)")
    print()
    
    # Example 2: Paper's example - A♣2♣|6♣J♥K♥
    # Note: Using (rank, suit) where rank: A=12, K=11, Q=10, J=9, T=8, ..., 6=4, ..., 2=0
    # Original: A♣=12,2  2♣=0,2  6♣=4,2  J♥=9,1  K♥=11,1
    hand_paper = [(12, 2), (0, 2), (4, 2), (9, 1), (11, 1)]
    canonical_paper = iso.canonicalize_hand(hand_paper, [2, 3])
    print(f"Paper example: A♣2♣|6♣J♥K♥ = {hand_paper}")
    print(f"Canonical (should be A♠2♠|6♠K♥J♥): {canonical_paper}")
    print()
    
    # Example 3: With private and public cards
    private = [(12, 1), (11, 1)]  # A♥ K♥
    public = [(10, 2), (9, 2), (8, 0)]  # Q♣ J♣ T♠
    
    canonical_private, canonical_public = iso.canonicalize_with_private_public(private, public)
    print(f"Private: A♥ K♥ = {private}")
    print(f"Public: Q♣ J♣ T♠ = {public}")
    print(f"Canonical private: {canonical_private}")
    print(f"Canonical public: {canonical_public}")
    print()
    
    # Example 4: Same hand, different suits
    hand2 = [(12, 1), (11, 3)]  # A♥ K♦
    hand3 = [(11, 0), (12, 2)]  # K♠ A♣
    
    canonical2 = iso.canonicalize_hand(hand2)
    canonical3 = iso.canonicalize_hand(hand3)
    
    print(f"Hand 2: A♥ K♦ = {hand2} -> {canonical2}")
    print(f"Hand 3: K♠ A♣ = {hand3} -> {canonical3}")
    print(f"Same canonical? {canonical2 == canonical3}")


def test_texas_holdem_combinations():
    """Test that we generate the correct number of canonical hands for Texas Hold'em
    
    This tests the specific case where we have private and public cards:
    - Flop: 2 private + 3 public = 1,286,792 canonical forms
    - Turn: 2 private + 4 public = 55,190,538 canonical forms (if time permits)
    - River: 2 private + 5 public = 2,428,287,420 canonical forms (too large to test)
    """
    iso = HandIsomorphism()
    
    # Generate all possible cards
    all_cards = [(rank, suit) for rank in range(13) for suit in range(4)]
    
    print("Testing Texas Hold'em canonical hand counts...")
    print("=" * 50)
    
    # Test Flop (2 private + 3 public)
    print("Testing Flop (2 private + 3 public cards)...")
    flop_canonical = set()
    count = 0
    
    # Generate all possible private hands
    for private in itertools.combinations(all_cards, 2):
        # Get remaining cards
        remaining = [c for c in all_cards if c not in private]
        
        # Generate all possible flops
        for public in itertools.combinations(remaining, 3):
            count += 1
            
            # Get canonical form preserving private/public structure
            canonical_private, canonical_public = iso.canonicalize_with_private_public(
                list(private), list(public)
            )
            
            # Create tuple for hashing (private cards first, then public)
            canonical_tuple = (tuple(canonical_private), tuple(canonical_public))
            flop_canonical.add(canonical_tuple)
            
            # Progress indicator
            if count % 100000 == 0:
                print(f"  Processed {count:,} hands, found {len(flop_canonical):,} canonical forms...")
    
    print(f"\nFlop Results:")
    print(f"  Total hands processed: {count:,}")
    print(f"  Canonical forms found: {len(flop_canonical):,}")
    print(f"  Expected (from paper): 1,286,792")
    print(f"  Match: {'✓ YES!' if len(flop_canonical) == 1286792 else '✗ NO'}")
    
    # Optionally test Turn (warning: this takes longer)
    test_turn = False  # Set to True if you want to test turn
    
    if test_turn:
        print("\nTesting Turn (2 private + 4 public cards)...")
        print("WARNING: This will take several minutes...")
        turn_canonical = set()
        count = 0
        
        for private in itertools.combinations(all_cards, 2):
            remaining = [c for c in all_cards if c not in private]
            
            for public in itertools.combinations(remaining, 4):
                count += 1
                
                canonical_private, canonical_public = iso.canonicalize_with_private_public(
                    list(private), list(public)
                )
                
                canonical_tuple = (tuple(canonical_private), tuple(canonical_public))
                turn_canonical.add(canonical_tuple)
                
                if count % 1000000 == 0:
                    print(f"  Processed {count:,} hands, found {len(turn_canonical):,} canonical forms...")
        
        print(f"\nTurn Results:")
        print(f"  Total hands processed: {count:,}")
        print(f"  Canonical forms found: {len(turn_canonical):,}")
        print(f"  Expected (from paper): 55,190,538")
        print(f"  Match: {'✓ YES!' if len(turn_canonical) == 55190538 else '✗ NO'}")


def verify_paper_examples():
    """Verify specific examples from the paper to ensure our implementation matches"""
    iso = HandIsomorphism()
    
    print("\nVerifying paper examples...")
    print("=" * 50)
    
    # Example from page 3: A♣2♣|6♣J♥K♥ should map to A♠2♠|6♠K♥J♥
    # A=12, K=11, J=9, 6=4, 2=0
    # ♠=0, ♥=1, ♣=2, ♦=3
    
    hand = [(12, 2), (0, 2), (4, 2), (9, 1), (11, 1)]  # A♣2♣|6♣J♥K♥
    canonical_private, canonical_public = iso.canonicalize_with_private_public(
        hand[:2], hand[2:]
    )
    
    print("Example 1: A♣2♣|6♣J♥K♥")
    print(f"  Input: {hand[:2]}|{hand[2:]}")
    print(f"  Canonical: {canonical_private}|{canonical_public}")
    
    # Expected: A♠2♠|6♠K♥J♥ = [(12,0), (0,0)]|[(4,0), (11,1), (9,1)]
    expected_private = [(12, 0), (0, 0)]
    expected_public = [(4, 0), (11, 1), (9, 1)]
    
    if canonical_private == expected_private and canonical_public == expected_public:
        print("  ✓ Matches expected canonical form!")
    else:
        print("  ✗ Does not match expected form")
        print(f"    Expected: {expected_private}|{expected_public}")


if __name__ == "__main__":
    print("Running canonical hand count test...")
    print("=" * 40)
    test_canonical_hands()
    print()
    print("Testing Texas Hold'em combinations...")
    print("=" * 40)
    test_texas_holdem_combinations()
    print()
    verify_paper_examples()
    print()
    print("Example usage:")
    print("=" * 40)
    example_usage()