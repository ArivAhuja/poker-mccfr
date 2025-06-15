import numpy as np
from sklearn_extra.cluster import KMedoids
from scipy.stats import wasserstein_distance
from phevaluator.evaluator import evaluate_cards
from collections import defaultdict
from itertools import combinations
import pickle
import random
from typing import List, Tuple, Dict, Optional
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
import os

# Import the hand isomorphism class from your implementation
# You'll need to save the HandIsomorphism class from paste.txt to a file
from hand_isomorphism import HandIsomorphism


def compute_single_hand_strength(args):
    """
    Helper function for parallel computation of hand strength distributions.
    Separated out for multiprocessing.
    """
    hero_cards, board_cards, num_runout_cards, n_bins, max_runouts, max_opponents = args
    
    # Generate all cards
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['h', 'd', 'c', 's']
    deck = [rank + suit for rank in ranks for suit in suits]
    
    # Remove hero and board cards
    exclude_cards = hero_cards + board_cards
    remaining_deck = [card for card in deck if card not in exclude_cards]
    
    # Generate runout cards
    runout_cards = [list(combo) for combo in combinations(remaining_deck, num_runout_cards)]
    
    # Sample runouts if there are too many
    if len(runout_cards) > max_runouts:
        runout_cards = random.sample(runout_cards, max_runouts)
    
    winrates = []
    
    for runout_card in runout_cards:
        wins, ties, total = 0, 0, 0
        
        # Get remaining deck for opponent cards
        remaining_for_opponents = [card for card in remaining_deck if card not in runout_card]
        opponent_cards = [list(combo) for combo in combinations(remaining_for_opponents, 2)]
        
        # Sample opponents if there are too many
        if len(opponent_cards) > max_opponents:
            opponent_cards = random.sample(opponent_cards, max_opponents)
        
        for opponent_card in opponent_cards:
            hero_hand = hero_cards + board_cards + runout_card
            opponent_hand = opponent_card + board_cards + runout_card
            
            hero_value = evaluate_cards(*hero_hand)
            opponent_value = evaluate_cards(*opponent_hand)
            
            if hero_value < opponent_value:  # Lower is better in phevaluator
                wins += 1
            elif hero_value == opponent_value:
                ties += 1
            total += 1
        
        if total > 0:
            winrate = (wins + (ties * 0.5)) / total
            winrates.append(winrate)
    
    # Create histogram
    hist, _ = np.histogram(winrates, bins=n_bins, range=(0, 1))
    # Normalize to get probability distribution
    hist = hist.astype(float) / hist.sum() if hist.sum() > 0 else hist
    
    return hist


class PokerAbstractionPipeline:
    def __init__(self, 
                 n_clusters_flop: int = 200,
                 n_clusters_turn: int = 200,
                 n_clusters_river: int = 200,
                 n_preflop_clusters: int = 8,
                 sample_size_flop: int = 50000,
                 sample_size_turn: int = 50000,
                 sample_size_river: int = 100000,
                 n_bins: int = 20,
                 random_seed: int = 42,
                 n_workers: int = None,
                 max_runouts: int = 100,
                 max_opponents: int = 50):
        """
        Initialize the poker abstraction pipeline.
        
        Args:
            n_clusters_flop: Number of clusters for flop
            n_clusters_turn: Number of clusters for turn
            n_clusters_river: Number of clusters for river
            n_preflop_clusters: Number of preflop clusters for river abstraction
            sample_size_flop: Number of samples for flop clustering
            sample_size_turn: Number of samples for turn clustering
            sample_size_river: Number of samples for river clustering
            n_bins: Number of bins for hand strength histogram
            random_seed: Random seed for reproducibility
            n_workers: Number of parallel workers (None = use all CPUs)
            max_runouts: Maximum number of runouts to sample
            max_opponents: Maximum number of opponents to sample
        """
        self.n_clusters_flop = n_clusters_flop
        self.n_clusters_turn = n_clusters_turn
        self.n_clusters_river = n_clusters_river
        self.n_preflop_clusters = n_preflop_clusters
        self.sample_size_flop = sample_size_flop
        self.sample_size_turn = sample_size_turn
        self.sample_size_river = sample_size_river
        self.n_bins = n_bins
        self.random_seed = random_seed
        self.n_workers = n_workers or mp.cpu_count()
        self.max_runouts = max_runouts
        self.max_opponents = max_opponents
        
        # Initialize hand isomorphism
        self.iso = HandIsomorphism()
        
        # Models
        self.flop_model = None
        self.turn_model = None
        self.river_model = None
        self.preflop_model = None
        
        # Preflop cluster centers and assignments
        self.preflop_centers = None
        self.preflop_assignments = {}  # canonical hand -> cluster
        
        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Card conversion mappings
        self.rank_to_int = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, 
                           '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        self.suit_to_int = {'s': 0, 'h': 1, 'c': 2, 'd': 3}
        self.int_to_rank = {v: k for k, v in self.rank_to_int.items()}
        self.int_to_suit = {v: k for k, v in self.suit_to_int.items()}
        
        # Cache for canonical preflop hands
        self._canonical_preflop_cache = None
    
    def card_str_to_tuple(self, card_str: str) -> Tuple[int, int]:
        """Convert card string (e.g., 'As') to tuple format (12, 0)"""
        rank = self.rank_to_int[card_str[0]]
        suit = self.suit_to_int[card_str[1]]
        return (rank, suit)
    
    def card_tuple_to_str(self, card_tuple: Tuple[int, int]) -> str:
        """Convert card tuple (12, 0) to string format ('As')"""
        rank, suit = card_tuple
        return self.int_to_rank[rank] + self.int_to_suit[suit]
    
    def cards_to_tuples(self, cards: List[str]) -> List[Tuple[int, int]]:
        """Convert list of card strings to list of tuples"""
        return [self.card_str_to_tuple(card) for card in cards]
    
    def tuples_to_cards(self, tuples: List[Tuple[int, int]]) -> List[str]:
        """Convert list of card tuples to list of strings"""
        return [self.card_tuple_to_str(t) for t in tuples]
    
    def canonicalize_cards(self, hero_cards: List[str], board_cards: List[str]) -> Tuple[List[str], List[str]]:
        """Canonicalize cards using the hand isomorphism algorithm"""
        # Convert to tuple format
        hero_tuples = self.cards_to_tuples(hero_cards)
        board_tuples = self.cards_to_tuples(board_cards)
        
        # Use the hand isomorphism canonicalization
        if board_cards:
            canon_hero_tuples, canon_board_tuples = self.iso.canonicalize_with_private_public(
                hero_tuples, board_tuples
            )
        else:
            # For preflop (no board cards)
            canon_all = self.iso.canonicalize_hand(hero_tuples)
            canon_hero_tuples = canon_all
            canon_board_tuples = []
        
        # Convert back to string format
        canon_hero = self.tuples_to_cards(canon_hero_tuples)
        canon_board = self.tuples_to_cards(canon_board_tuples) if canon_board_tuples else []
        
        return canon_hero, canon_board
    
    def generate_all_canonical_preflop(self) -> List[Tuple[List[str], str]]:
        """Generate all canonical preflop hands (169 total)"""
        if self._canonical_preflop_cache is not None:
            return self._canonical_preflop_cache
        
        print("Generating all canonical preflop hands...")
        canonical_hands = {}
        
        # Generate all possible 2-card combinations
        all_cards = [(rank, suit) for rank in range(13) for suit in range(4)]
        
        for combo in tqdm(combinations(all_cards, 2), total=1326, desc="Processing preflop hands"):
            # Canonicalize
            canon = self.iso.canonicalize_hand(list(combo))
            canon_tuple = tuple(canon)
            
            # Convert to string format
            canon_str = self.tuples_to_cards(canon)
            
            # Store with a representative string (for display)
            if canon_tuple not in canonical_hands:
                # Create a human-readable representation
                rank1, suit1 = canon[0]
                rank2, suit2 = canon[1]
                
                if rank1 == rank2:  # Pocket pair
                    hand_type = f"{self.int_to_rank[rank1]}{self.int_to_rank[rank2]}"
                elif suit1 == suit2:  # Suited
                    hand_type = f"{self.int_to_rank[rank1]}{self.int_to_rank[rank2]}s"
                else:  # Offsuit
                    hand_type = f"{self.int_to_rank[rank1]}{self.int_to_rank[rank2]}o"
                
                canonical_hands[canon_tuple] = (canon_str, hand_type)
        
        print(f"Generated {len(canonical_hands)} canonical preflop hands")
        
        # Cache and return
        self._canonical_preflop_cache = list(canonical_hands.values())
        return self._canonical_preflop_cache
    
    def generate_cards(self, num_cards: int, exclude_cards: Optional[List[str]] = None) -> List[List[str]]:
        """Generate all possible combinations of num_cards from a standard deck"""
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['h', 'd', 'c', 's']
        deck = [rank + suit for rank in ranks for suit in suits]
        
        if exclude_cards:
            deck = [card for card in deck if card not in exclude_cards]
        
        return [list(combo) for combo in combinations(deck, num_cards)]
    
    def compute_hand_strength_distribution(self, 
                                         hero_cards: List[str], 
                                         board_cards: List[str],
                                         num_runout_cards: int) -> np.ndarray:
        """
        Compute hand strength distribution for a given hand and board.
        Returns histogram of win rates.
        """
        args = (hero_cards, board_cards, num_runout_cards, self.n_bins, 
                self.max_runouts, self.max_opponents)
        return compute_single_hand_strength(args)
    
    def compute_hand_strength_batch_parallel(self, hand_batch: List[Tuple]) -> List[Tuple]:
        """
        Compute hand strength distributions for a batch of hands in parallel.
        """
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Prepare arguments
            args_list = []
            for hero_cards, board_cards, num_runout_cards in hand_batch:
                args = (hero_cards, board_cards, num_runout_cards, self.n_bins, 
                       self.max_runouts, self.max_opponents)
                args_list.append(args)
            
            # Submit all tasks
            futures = [executor.submit(compute_single_hand_strength, args) for args in args_list]
            
            # Collect results with progress bar
            results = []
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Computing hand strengths", leave=False):
                results.append(future.result())
        
        return results
    
    def compute_preflop_distribution(self, hero_cards: List[str]) -> np.ndarray:
        """
        Compute preflop hand strength distribution (no board cards).
        """
        return self.compute_hand_strength_distribution(hero_cards, [], 5)
    
    def train_preflop_clusters(self):
        """
        Train preflop clusters for river abstraction.
        Computes all 169 canonical hands without sampling.
        """
        print("\n" + "="*60)
        print("TRAINING PREFLOP CLUSTERS")
        print("="*60)
        
        # Get all canonical preflop hands
        canonical_hands = self.generate_all_canonical_preflop()
        
        # Compute distributions for all canonical hands
        print(f"\nComputing hand strength distributions for {len(canonical_hands)} canonical hands...")
        
        distributions = []
        canonical_hand_strs = []
        
        # Process in batches for parallel computation
        batch_size = 20
        hand_batch = []
        
        for canon_cards, hand_type in tqdm(canonical_hands, desc="Preparing hands"):
            hand_batch.append((canon_cards, [], 5))  # 5 runout cards for preflop
            canonical_hand_strs.append(canon_cards)
            
            if len(hand_batch) >= batch_size:
                # Compute batch in parallel
                batch_results = self.compute_hand_strength_batch_parallel(hand_batch)
                distributions.extend(batch_results)
                hand_batch = []
        
        # Process remaining hands
        if hand_batch:
            batch_results = self.compute_hand_strength_batch_parallel(hand_batch)
            distributions.extend(batch_results)
        
        # Convert to numpy array
        X = np.array(distributions)
        
        # Train KMedoids with Wasserstein distance
        print(f"\nTraining KMedoids with {self.n_preflop_clusters} clusters...")
        print("Using 'alternate' method with 'k-medoids++' initialization...")
        
        start_time = time.time()
        self.preflop_model = KMedoids(
            n_clusters=self.n_preflop_clusters,
            metric=wasserstein_distance,
            method='alternate',  # Much faster than default 'pam' method
            init='k-medoids++',  # Better initialization for faster convergence
            random_state=self.random_seed
        )
        self.preflop_model.fit(X)
        
        # Store cluster centers and assignments
        self.preflop_centers = self.preflop_model.cluster_centers_
        
        # Store cluster assignments for each canonical hand
        for i, (canon_cards, _) in enumerate(canonical_hands):
            canon_tuple = tuple(canon_cards)
            self.preflop_assignments[canon_tuple] = self.preflop_model.labels_[i]
        
        elapsed = time.time() - start_time
        print(f"Preflop clustering complete in {elapsed:.1f} seconds!")
        
        # Show cluster distribution
        cluster_counts = defaultdict(int)
        for cluster in self.preflop_model.labels_:
            cluster_counts[cluster] += 1
        
        print("\nCluster distribution:")
        for cluster in sorted(cluster_counts.keys()):
            print(f"  Cluster {cluster}: {cluster_counts[cluster]} hands")
    
    def get_preflop_cluster(self, hero_cards: List[str]) -> int:
        """
        Get preflop cluster for a hand using cached assignments.
        """
        # Canonicalize
        canon_hero, _ = self.canonicalize_cards(hero_cards, [])
        canon_tuple = tuple(canon_hero)
        
        # Look up in cache
        if canon_tuple in self.preflop_assignments:
            return self.preflop_assignments[canon_tuple]
        
        # If not in cache, compute (shouldn't happen with proper canonicalization)
        dist = self.compute_preflop_distribution(hero_cards)
        return self.preflop_model.predict([dist])[0]
    
    def compute_river_features(self, hero_cards: List[str], board_cards: List[str]) -> np.ndarray:
        """
        Compute 8-dimensional feature vector for river hands.
        Shows performance against each preflop cluster.
        """
        features = np.zeros(self.n_preflop_clusters)
        cluster_counts = np.zeros(self.n_preflop_clusters)
        
        # Get all possible opponent hands
        opponent_samples = self.generate_cards(2, hero_cards + board_cards)
        
        # Sample if too many
        if len(opponent_samples) > 100:
            opponent_samples = random.sample(opponent_samples, 100)
        
        # Process each opponent hand
        for opponent_cards in opponent_samples:
            # Get preflop cluster for this opponent hand
            opp_cluster = self.get_preflop_cluster(opponent_cards)
            cluster_counts[opp_cluster] += 1
            
            # Evaluate showdown
            hero_hand = hero_cards + board_cards
            opponent_hand = opponent_cards + board_cards
            
            hero_value = evaluate_cards(*hero_hand)
            opponent_value = evaluate_cards(*opponent_hand)
            
            if hero_value < opponent_value:  # Win
                features[opp_cluster] += 1
            elif hero_value == opponent_value:  # Tie
                features[opp_cluster] += 0.5
        
        # Normalize by cluster counts
        for i in range(self.n_preflop_clusters):
            if cluster_counts[i] > 0:
                features[i] /= cluster_counts[i]
            else:
                features[i] = 0.5  # Default to 50% if no hands from this cluster
        
        return features
    
    def sample_hands_flop_turn(self, num_board_cards: int, sample_size: int) -> Dict[Tuple, np.ndarray]:
        """
        Sample hands and compute their strength distributions for flop/turn.
        """
        street = "flop" if num_board_cards == 3 else "turn"
        print(f"\n{'='*60}")
        print(f"SAMPLING {street.upper()} HANDS")
        print(f"{'='*60}")
        print(f"Target sample size: {sample_size:,}")
        
        samples = {}
        num_hole_cards = 2
        num_runout_cards = 5 - num_board_cards
        
        # Generate all possible hole cards
        all_hero_cards = self.generate_cards(num_hole_cards)
        
        count = 0
        start_time = time.time()
        batch_data = []  # Store (key, hero_cards, board_cards) for batch processing
        batch_size = 50  # Process in batches
        
        pbar = tqdm(total=sample_size, desc=f"Sampling {street} hands")
        
        while count < sample_size:
            # Sample random hole cards
            hero_cards = random.choice(all_hero_cards)
            
            # Generate all possible boards for these hole cards
            possible_boards = self.generate_cards(num_board_cards, hero_cards)
            
            # Sample a random board
            board_cards = random.choice(possible_boards)
            
            # Canonicalize
            canon_hero, canon_board = self.canonicalize_cards(hero_cards, board_cards)
            key = (tuple(canon_hero), tuple(canon_board))
            
            # Skip if already sampled
            if key in samples:
                continue
            
            # Add to batch
            batch_data.append((key, hero_cards, board_cards, num_runout_cards))
            
            # Process batch when full
            if len(batch_data) >= batch_size or count + len(batch_data) >= sample_size:
                # Compute distributions in parallel
                hand_batch = [(h, b, r) for _, h, b, r in batch_data]
                batch_results = self.compute_hand_strength_batch_parallel(hand_batch)
                
                # Store results
                for i, (key, _, _, _) in enumerate(batch_data):
                    samples[key] = batch_results[i]
                    count += 1
                    pbar.update(1)
                
                batch_data = []
                
                # Update progress
                if count % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = count / elapsed
                    eta = (sample_size - count) / rate
                    pbar.set_postfix({
                        'rate': f'{rate:.1f} hands/sec',
                        'eta': f'{eta:.0f}s'
                    })
        
        pbar.close()
        elapsed = time.time() - start_time
        print(f"\nSampling complete: {count:,} hands in {elapsed:.1f}s ({count/elapsed:.1f} hands/sec)")
        
        return samples
    
    def sample_hands_river(self, sample_size: int) -> Dict[Tuple, np.ndarray]:
        """
        Sample river hands and compute their feature vectors.
        """
        print(f"\n{'='*60}")
        print("SAMPLING RIVER HANDS")
        print(f"{'='*60}")
        print(f"Target sample size: {sample_size:,}")
        
        # Make sure preflop clusters are trained
        if self.preflop_model is None:
            self.train_preflop_clusters()
        
        samples = {}
        count = 0
        start_time = time.time()
        
        # Generate all possible hole cards
        all_hero_cards = self.generate_cards(2)
        
        pbar = tqdm(total=sample_size, desc="Sampling river hands")
        
        while count < sample_size:
            # Sample random hole cards
            hero_cards = random.choice(all_hero_cards)
            
            # Generate all possible boards (5 cards) for these hole cards
            possible_boards = self.generate_cards(5, hero_cards)
            
            # Sample a random board
            board_cards = random.choice(possible_boards)
            
            # Canonicalize
            canon_hero, canon_board = self.canonicalize_cards(hero_cards, board_cards)
            key = (tuple(canon_hero), tuple(canon_board))
            
            # Skip if already sampled
            if key in samples:
                continue
            
            # Compute river features
            features = self.compute_river_features(hero_cards, board_cards)
            samples[key] = features
            
            count += 1
            pbar.update(1)
            
            if count % 100 == 0:
                elapsed = time.time() - start_time
                rate = count / elapsed
                eta = (sample_size - count) / rate
                pbar.set_postfix({
                    'rate': f'{rate:.1f} hands/sec',
                    'eta': f'{eta:.0f}s'
                })
        
        pbar.close()
        elapsed = time.time() - start_time
        print(f"\nSampling complete: {count:,} hands in {elapsed:.1f}s ({count/elapsed:.1f} hands/sec)")
        
        return samples
    
    def train_flop_turn_model(self, num_board_cards: int, n_clusters: int, sample_size: int):
        """
        Train KMedoids model for flop or turn.
        """
        street = "flop" if num_board_cards == 3 else "turn"
        print(f"\n{'='*60}")
        print(f"TRAINING {street.upper()} MODEL")
        print(f"{'='*60}")
        print(f"Clusters: {n_clusters}")
        
        # Sample hands
        samples = self.sample_hands_flop_turn(num_board_cards, sample_size)
        
        # Convert to numpy array
        X = np.array(list(samples.values()))
        keys = list(samples.keys())
        
        print(f"\nTraining KMedoids on {len(X):,} samples...")
        print("Using Wasserstein distance with 'alternate' method and 'k-medoids++' init...")
        
        start_time = time.time()
        model = KMedoids(
            n_clusters=n_clusters,
            metric=wasserstein_distance,
            method='alternate',  # Much faster than default 'pam' method
            init='k-medoids++',  # Better initialization for faster convergence
            random_state=self.random_seed
        )
        
        model.fit(X)
        elapsed = time.time() - start_time
        
        # Store model
        if num_board_cards == 3:
            self.flop_model = model
        else:
            self.turn_model = model
        
        print(f"\n{street.capitalize()} model training complete in {elapsed:.1f} seconds!")
        
        # Show cluster distribution
        cluster_counts = defaultdict(int)
        for cluster in model.labels_:
            cluster_counts[cluster] += 1
        
        print("\nCluster size distribution:")
        sizes = list(cluster_counts.values())
        print(f"  Min size: {min(sizes)}")
        print(f"  Max size: {max(sizes)}")
        print(f"  Avg size: {np.mean(sizes):.1f}")
        print(f"  Std dev: {np.std(sizes):.1f}")
        
        return model, keys
    
    def train_river_model(self):
        """
        Train KMedoids model for river using L2 distance.
        """
        print(f"\n{'='*60}")
        print("TRAINING RIVER MODEL")
        print(f"{'='*60}")
        print(f"Clusters: {self.n_clusters_river}")
        print(f"Using {self.n_preflop_clusters}-dimensional feature vectors")
        
        # Sample hands
        samples = self.sample_hands_river(self.sample_size_river)
        
        # Convert to numpy array
        X = np.array(list(samples.values()))
        keys = list(samples.keys())
        
        print(f"\nTraining KMedoids on {len(X):,} samples...")
        print("Using L2 distance with 'alternate' method and 'k-medoids++' init...")
        
        start_time = time.time()
        self.river_model = KMedoids(
            n_clusters=self.n_clusters_river,
            metric='euclidean',  # L2 distance for river features
            method='alternate',  # Much faster than default 'pam' method
            init='k-medoids++',  # Better initialization for faster convergence
            random_state=self.random_seed
        )
        
        self.river_model.fit(X)
        elapsed = time.time() - start_time
        
        print(f"\nRiver model training complete in {elapsed:.1f} seconds!")
        
        # Show cluster distribution
        cluster_counts = defaultdict(int)
        for cluster in self.river_model.labels_:
            cluster_counts[cluster] += 1
        
        print("\nCluster size distribution:")
        sizes = list(cluster_counts.values())
        print(f"  Min size: {min(sizes)}")
        print(f"  Max size: {max(sizes)}")
        print(f"  Avg size: {np.mean(sizes):.1f}")
        print(f"  Std dev: {np.std(sizes):.1f}")
        
        # Show feature importance
        print("\nAverage feature values by cluster (first 5 clusters):")
        for i in range(min(5, self.n_clusters_river)):
            cluster_mask = self.river_model.labels_ == i
            cluster_features = X[cluster_mask].mean(axis=0)
            print(f"  Cluster {i}: {cluster_features}")
        
        return self.river_model, keys
    
    def train_all_models(self):
        """
        Train all models (flop, turn, river).
        """
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        total_start = time.time()
        
        # Train preflop clusters first (needed for river)
        self.train_preflop_clusters()
        
        # Train flop model
        self.train_flop_turn_model(3, self.n_clusters_flop, self.sample_size_flop)
        
        # Train turn model
        self.train_flop_turn_model(4, self.n_clusters_turn, self.sample_size_turn)
        
        # Train river model
        self.train_river_model()
        
        total_elapsed = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"ALL MODELS TRAINED SUCCESSFULLY!")
        print(f"Total training time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
        print(f"{'='*60}")
    
    def get_flop_cluster(self, hero_cards: List[str], board_cards: List[str]) -> int:
        """
        Get cluster assignment for a flop hand.
        """
        if self.flop_model is None:
            raise ValueError("Flop model not trained yet!")
        
        # Compute hand strength distribution
        hist = self.compute_hand_strength_distribution(hero_cards, board_cards, 2)
        
        # Predict cluster
        cluster = self.flop_model.predict([hist])[0]
        
        return cluster
    
    def get_turn_cluster(self, hero_cards: List[str], board_cards: List[str]) -> int:
        """
        Get cluster assignment for a turn hand.
        """
        if self.turn_model is None:
            raise ValueError("Turn model not trained yet!")
        
        # Compute hand strength distribution
        hist = self.compute_hand_strength_distribution(hero_cards, board_cards, 1)
        
        # Predict cluster
        cluster = self.turn_model.predict([hist])[0]
        
        return cluster
    
    def get_river_cluster(self, hero_cards: List[str], board_cards: List[str]) -> int:
        """
        Get cluster assignment for a river hand.
        """
        if self.river_model is None:
            raise ValueError("River model not trained yet!")
        
        # Compute river features
        features = self.compute_river_features(hero_cards, board_cards)
        
        # Predict cluster
        cluster = self.river_model.predict([features])[0]
        
        return cluster
    
    def save_models(self, filepath_prefix: str):
        """
        Save all trained models to disk.
        """
        print(f"\nSaving models to {filepath_prefix}_models.pkl...")
        
        models = {
            'flop_model': self.flop_model,
            'turn_model': self.turn_model,
            'river_model': self.river_model,
            'preflop_model': self.preflop_model,
            'preflop_centers': self.preflop_centers,
            'preflop_assignments': self.preflop_assignments,
            'config': {
                'n_clusters_flop': self.n_clusters_flop,
                'n_clusters_turn': self.n_clusters_turn,
                'n_clusters_river': self.n_clusters_river,
                'n_preflop_clusters': self.n_preflop_clusters,
                'n_bins': self.n_bins,
                'max_runouts': self.max_runouts,
                'max_opponents': self.max_opponents
            }
        }
        
        with open(f"{filepath_prefix}_models.pkl", 'wb') as f:
            pickle.dump(models, f)
        
        print(f"Models saved successfully!")
        print(f"File size: {os.path.getsize(f'{filepath_prefix}_models.pkl') / 1024 / 1024:.1f} MB")
    
    def load_models(self, filepath_prefix: str):
        """
        Load trained models from disk.
        """
        print(f"\nLoading models from {filepath_prefix}_models.pkl...")
        
        with open(f"{filepath_prefix}_models.pkl", 'rb') as f:
            models = pickle.load(f)
        
        self.flop_model = models['flop_model']
        self.turn_model = models['turn_model']
        self.river_model = models['river_model']
        self.preflop_model = models['preflop_model']
        self.preflop_centers = models['preflop_centers']
        self.preflop_assignments = models['preflop_assignments']
        
        # Update config
        config = models['config']
        self.n_clusters_flop = config['n_clusters_flop']
        self.n_clusters_turn = config['n_clusters_turn']
        self.n_clusters_river = config['n_clusters_river']
        self.n_preflop_clusters = config['n_preflop_clusters']
        self.n_bins = config['n_bins']
        self.max_runouts = config.get('max_runouts', 100)
        self.max_opponents = config.get('max_opponents', 50)
        
        print(f"Models loaded successfully!")


# Example usage
if __name__ == "__main__":
    # Create pipeline with custom parameters
    print("Poker Abstraction Pipeline")
    print("Using 'alternate' method with 'k-medoids++' init")
    print("Expected 60-80% faster training than default settings")
    print("-" * 60)
    
    pipeline = PokerAbstractionPipeline(
        n_clusters_flop=200,
        n_clusters_turn=200,
        n_clusters_river=200,
        n_preflop_clusters=8,
        sample_size_flop=10_000,     # Reduced for testing
        sample_size_turn=20_000,     # Reduced for testing
        sample_size_river=40_000,    # Reduced for testing
        n_bins=20,
        n_workers=None,  # Use all available CPUs
        max_runouts=100,
        max_opponents=50
    )
    
    # Train all models
    pipeline.train_all_models()
    
    # Save models
    pipeline.save_models("poker_abstraction")
    
    # Example: Get cluster for a specific hand
    hero_cards = ['As', 'Kh']
    flop_cards = ['Qd', 'Jc', 'Ts']
    
    flop_cluster = pipeline.get_flop_cluster(hero_cards, flop_cards)
    print(f"\nHand {hero_cards} on flop {flop_cards} belongs to cluster {flop_cluster}")
    
    # Turn example
    turn_cards = flop_cards + ['9h']
    turn_cluster = pipeline.get_turn_cluster(hero_cards, turn_cards)
    print(f"Hand {hero_cards} on turn {turn_cards} belongs to cluster {turn_cluster}")
    
    # River example
    river_cards = turn_cards + ['8s']
    river_cluster = pipeline.get_river_cluster(hero_cards, river_cards)
    print(f"Hand {hero_cards} on river {river_cards} belongs to cluster {river_cluster}")