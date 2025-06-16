#!/usr/bin/env python3
"""
Evaluate how close a trained MCCFR-P strategy is to Nash equilibrium.
Fixed version that handles Config class from training script.
"""

import pickle
import gzip
import numpy as np
import hashlib
import pyspiel
from typing import Dict, List, Tuple, Optional
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
import multiprocessing as mp

# Add the Config class definition to handle unpickling
@dataclass
class Config:
    """Configuration for MCCFR-P solver (needed for unpickling)"""
    prune_threshold: float = -30000.0
    regret_floor: float = -50000.0
    prune_threshold_iterations: int = 10000
    strategy_interval: int = 1000
    discount_interval: int = 10000
    lcfr_threshold: int = 40000
    prune_probability: float = 0.95
    num_processes: int = mp.cpu_count()
    batch_size: int = 100
    update_batch_size: int = 1000
    num_players: int = 2
    max_actions: int = 10
    checkpoint_interval: int = 10000
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 1000
    detailed_log_interval: int = 10000

def load_strategy_safe(filepath: str) -> Dict:
    """Safely load strategy handling different formats"""
    print(f"Loading strategy from {filepath}...")
    
    with gzip.open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different file formats
    if 'strategy' in data:
        # This is a final strategy file
        strategy_data = {
            'strategy': data['strategy'],
            'iteration': data.get('iteration', 0),
            'elapsed_time': data.get('elapsed_time', 0)
        }
    elif 'minimizer_state' in data:
        # This is a checkpoint file - extract strategy
        print("Detected checkpoint file format, extracting strategy...")
        strategy = {}
        minimizer_state = data['minimizer_state']
        
        if 'strategies' in minimizer_state:
            for state_hash, strategy_sum in minimizer_state['strategies'].items():
                total = sum(strategy_sum)
                if total > 0:
                    avg_strategy = [s/total for s in strategy_sum]
                    # Only save non-uniform strategies
                    if not all(abs(s - avg_strategy[0]) < 1e-6 for s in avg_strategy):
                        strategy[state_hash] = avg_strategy
        
        strategy_data = {
            'strategy': strategy,
            'iteration': data.get('iteration', 0),
            'elapsed_time': data.get('elapsed_time', 0)
        }
    else:
        raise ValueError("Unknown file format")
    
    print(f"Strategy loaded successfully!")
    print(f"  Training iterations: {strategy_data['iteration']:,}")
    print(f"  Number of strategies: {len(strategy_data['strategy']):,}")
    
    return strategy_data

def create_game():
    """Create the same game used in training"""
    CUSTOM_LIMIT_HOLDEM_ACPC_GAMEDEF = """\
GAMEDEF
limit
numPlayers = 2
numRounds = 2
blind = 1 2
raiseSize = 2 2
firstPlayer = 1
maxRaises = 2 2
numSuits = 2
numRanks = 6
numHoleCards = 2
numBoardCards = 0 3
stack = 20
END GAMEDEF
"""
    universal_poker = pyspiel.universal_poker
    return universal_poker.load_universal_poker_from_acpc_gamedef(CUSTOM_LIMIT_HOLDEM_ACPC_GAMEDEF)

class StrategyEvaluator:
    """Evaluate strategy exploitability and Nash distance"""
    
    def __init__(self, strategy_data: Dict, game: pyspiel.Game):
        self.strategy = strategy_data['strategy']
        self.game = game
        self.big_blind = 2.0  # From game definition
        
        # Cache for best response values
        self.br_cache = {}
        
        # Statistics
        self.nodes_evaluated = 0
        
    def get_strategy_for_state(self, info_state: str, num_actions: int) -> np.ndarray:
        """Get strategy (action probabilities) for an info state"""
        state_hash = hashlib.md5(info_state.encode()).hexdigest()
        
        if state_hash in self.strategy:
            return np.array(self.strategy[state_hash])
        else:
            # Uniform strategy for unseen states
            return np.ones(num_actions) / num_actions
    
    def compute_best_response_value(self, state: pyspiel.State, 
                                  br_player: int, 
                                  prob_reach_1: float = 1.0,
                                  prob_reach_2: float = 1.0) -> float:
        """
        Compute value of best response against fixed strategy.
        Uses recursive tree traversal to find optimal counter-strategy.
        """
        self.nodes_evaluated += 1
        
        if state.is_terminal():
            return state.returns()[br_player]
        
        if state.is_chance_node():
            value = 0.0
            for action, prob in state.chance_outcomes():
                new_state = state.clone()
                new_state.apply_action(action)
                value += prob * self.compute_best_response_value(
                    new_state, br_player, prob_reach_1, prob_reach_2)
            return value
        
        current_player = state.current_player()
        info_state = state.information_state_string()
        legal_actions = state.legal_actions()
        
        # Create cache key including reach probabilities
        cache_key = (info_state, br_player, 
                    round(prob_reach_1, 6), round(prob_reach_2, 6))
        
        if cache_key in self.br_cache:
            return self.br_cache[cache_key]
        
        if current_player == br_player:
            # Best response player - choose action with highest value
            action_values = []
            
            for action in legal_actions:
                new_state = state.clone()
                new_state.apply_action(action)
                
                if br_player == 0:
                    new_reach_1 = prob_reach_1
                    new_reach_2 = prob_reach_2
                else:
                    new_reach_1 = prob_reach_1
                    new_reach_2 = prob_reach_2
                
                value = self.compute_best_response_value(
                    new_state, br_player, new_reach_1, new_reach_2)
                action_values.append(value)
            
            best_value = max(action_values)
            self.br_cache[cache_key] = best_value
            return best_value
        
        else:
            # Fixed strategy player - use trained strategy
            strategy = self.get_strategy_for_state(info_state, len(legal_actions))
            expected_value = 0.0
            
            for i, action in enumerate(legal_actions):
                new_state = state.clone()
                new_state.apply_action(action)
                
                if current_player == 0:
                    new_reach_1 = prob_reach_1 * strategy[i]
                    new_reach_2 = prob_reach_2
                else:
                    new_reach_1 = prob_reach_1
                    new_reach_2 = prob_reach_2 * strategy[i]
                
                value = self.compute_best_response_value(
                    new_state, br_player, new_reach_1, new_reach_2)
                expected_value += strategy[i] * value
            
            self.br_cache[cache_key] = expected_value
            return expected_value
    
    def compute_exploitability(self) -> Tuple[float, float, float]:
        """
        Compute exploitability of the strategy.
        Returns: (exploitability_mbb, player0_br_value, player1_br_value)
        
        Exploitability is measured in milli-big-blinds per game (mbb/g).
        Nash equilibrium has exploitability = 0.
        """
        print("Computing exploitability (this may take a few minutes)...")
        
        # Reset statistics
        self.nodes_evaluated = 0
        self.br_cache.clear()
        
        # Compute best response value for each player
        initial_state = self.game.new_initial_state()
        
        print("Computing best response for Player 0...")
        start_time = time.time()
        br_value_0 = self.compute_best_response_value(initial_state, 0)
        time_0 = time.time() - start_time
        nodes_0 = self.nodes_evaluated
        
        print("Computing best response for Player 1...")
        self.nodes_evaluated = 0
        self.br_cache.clear()
        start_time = time.time()
        br_value_1 = self.compute_best_response_value(initial_state, 1)
        time_1 = time.time() - start_time
        nodes_1 = self.nodes_evaluated
        
        # Game value should be 0 for symmetric game at Nash
        # Exploitability = (BR0 + BR1) / 2
        exploitability = (br_value_0 + br_value_1) / 2.0
        
        # Convert to milli-big-blinds per game
        exploitability_mbb = (exploitability / self.big_blind) * 1000
        
        print(f"\nEvaluation complete:")
        print(f"  Player 0 best response: {br_value_0:.4f} ({nodes_0:,} nodes in {time_0:.1f}s)")
        print(f"  Player 1 best response: {br_value_1:.4f} ({nodes_1:,} nodes in {time_1:.1f}s)")
        
        return exploitability_mbb, br_value_0, br_value_1
    
    def compute_strategy_statistics(self) -> Dict:
        """Compute various statistics about the strategy"""
        stats = {
            'total_strategies': len(self.strategy),
            'pure_strategies': 0,
            'nearly_pure_strategies': 0,
            'mixed_strategies': 0,
            'max_entropy_strategies': 0,
            'avg_entropy': 0.0,
            'action_distribution': defaultdict(int)
        }
        
        total_entropy = 0.0
        
        for state_hash, probs in self.strategy.items():
            probs = np.array(probs)
            
            # Count strategy types
            max_prob = np.max(probs)
            if max_prob >= 0.999:
                stats['pure_strategies'] += 1
            elif max_prob >= 0.95:
                stats['nearly_pure_strategies'] += 1
            else:
                stats['mixed_strategies'] += 1
            
            # Compute entropy
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            total_entropy += entropy
            
            # Check if maximum entropy (uniform)
            max_possible_entropy = np.log(len(probs))
            if abs(entropy - max_possible_entropy) < 0.01:
                stats['max_entropy_strategies'] += 1
            
            # Track most likely action
            most_likely_action = np.argmax(probs)
            stats['action_distribution'][most_likely_action] += 1
        
        stats['avg_entropy'] = total_entropy / len(self.strategy) if self.strategy else 0
        
        return stats

def main():
    """Evaluate a trained strategy"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Evaluate MCCFR-P strategy')
    parser.add_argument('--strategy', type=str, 
                       default='limit_holdem_strategy_parallel.pkl.gz',
                       help='Path to strategy file')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to checkpoint file (alternative to strategy)')
    args = parser.parse_args()
    
    # Determine which file to load
    if args.checkpoint:
        filepath = args.checkpoint
    else:
        filepath = args.strategy
        
    if not os.path.exists(filepath):
        # Try checkpoint directory
        checkpoint_path = os.path.join('checkpoints', 'mccfr_checkpoint.pkl.gz')
        if os.path.exists(checkpoint_path):
            print(f"Strategy file not found, trying checkpoint: {checkpoint_path}")
            filepath = checkpoint_path
        else:
            print(f"Error: Could not find {filepath}")
            print("Make sure the training has completed and the file exists.")
            return
    
    # Create game
    game = create_game()
    
    try:
        # Load strategy with safe loading
        strategy_data = load_strategy_safe(filepath)
        
        # Create evaluator
        evaluator = StrategyEvaluator(strategy_data, game)
        
        # Compute exploitability
        print("\n=== Computing Exploitability ===")
        exploitability_mbb, br0, br1 = evaluator.compute_exploitability()
        
        print(f"\n=== Results ===")
        print(f"Exploitability: {exploitability_mbb:.2f} mbb/g")
        print(f"  (0 = Nash equilibrium, <1 = very strong, <10 = good)")
        
        if exploitability_mbb < 1.0:
            print("\n✓ Excellent! Strategy is within 1 mbb/g of Nash equilibrium!")
        elif exploitability_mbb < 10.0:
            print("\n✓ Good! Strategy is reasonably close to Nash equilibrium")
        elif exploitability_mbb < 50.0:
            print("\n→ Strategy has learned something but needs more training")
        else:
            print("\n→ Strategy is still far from Nash, continue training")
        
        # Compute additional statistics
        print("\n=== Strategy Statistics ===")
        stats = evaluator.compute_strategy_statistics()
        
        print(f"Total strategies: {stats['total_strategies']:,}")
        print(f"Pure strategies (>99.9% on one action): {stats['pure_strategies']:,} "
              f"({100*stats['pure_strategies']/stats['total_strategies']:.1f}%)")
        print(f"Nearly pure (>95% on one action): {stats['nearly_pure_strategies']:,} "
              f"({100*stats['nearly_pure_strategies']/stats['total_strategies']:.1f}%)")
        print(f"Mixed strategies: {stats['mixed_strategies']:,} "
              f"({100*stats['mixed_strategies']/stats['total_strategies']:.1f}%)")
        print(f"Average entropy: {stats['avg_entropy']:.3f}")
        
        print("\nAction distribution (most likely actions):")
        for action, count in sorted(stats['action_distribution'].items()):
            percentage = 100 * count / stats['total_strategies']
            print(f"  Action {action}: {count:,} states ({percentage:.1f}%)")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()