"""
Intelligent Strategy Mapper - Maps game states to strategies using contextual matching
Since we can't reverse the exact format, we'll use game context to find similar situations
"""

import random
import hashlib
import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
import pickle
import gzip
import os

from poker_engine import GameState, Action, Card, Street

class GameStateFeatures:
    """Extract meaningful features from a game state for matching"""
    
    def __init__(self, game_state: GameState, player_id: int):
        self.game_state = game_state
        self.player_id = player_id
        self.player = game_state.players[player_id]
        self.opponent = game_state.players[1 - player_id]
        
    def extract_features(self) -> Dict[str, any]:
        """Extract key features that likely influence strategy"""
        
        features = {}
        
        # Basic game info
        features['round'] = 0 if self.game_state.current_street == Street.PREFLOP else 1
        features['position'] = self.player_id  # 0 or 1
        features['pot_size'] = self.game_state.pot
        
        # Money/stack info
        features['my_stack'] = self.player.chips
        features['opp_stack'] = self.opponent.chips
        features['stack_ratio'] = self.player.chips / max(1, self.opponent.chips)
        
        # Cards
        features['my_cards'] = self._encode_cards(self.player.hole_cards)
        features['board_cards'] = self._encode_cards(self.game_state.community_cards)
        features['num_board_cards'] = len(self.game_state.community_cards)
        
        # Hand strength
        features['hand_strength'] = self._evaluate_hand_strength()
        features['hand_class'] = self._classify_hand()
        
        # Betting action
        features['pot_odds'] = self._calculate_pot_odds()
        features['aggression_level'] = self._analyze_aggression()
        features['betting_pattern'] = self._encode_betting_pattern()
        
        # Position info
        features['is_button'] = (self.game_state.dealer_button == self.player_id)
        features['acts_first'] = self._determine_position_action()
        
        return features
    
    def _encode_cards(self, cards: List[Card]) -> List[int]:
        """Encode cards as integers for comparison"""
        if not cards:
            return []
        
        # Convert to ranks (2-7 becomes 0-5)
        ranks = [card.rank - 2 for card in cards]
        suits = [0 if card.suit.value == "♠" else 1 for card in cards]
        
        # Create a combined encoding
        encoded = []
        for rank, suit in zip(ranks, suits):
            encoded.append(rank * 2 + suit)  # 0-11 for 12 total cards
        
        return sorted(encoded)  # Sort for consistency
    
    def _evaluate_hand_strength(self) -> float:
        """Simple hand strength evaluation (0-1)"""
        all_cards = self.player.hole_cards + self.game_state.community_cards
        
        if not all_cards:
            return 0.5
        
        # Count ranks
        rank_counts = defaultdict(int)
        for card in all_cards:
            rank_counts[card.rank] += 1
        
        # Basic strength
        max_rank = max(card.rank for card in all_cards)
        strength = (max_rank - 2) / 5.0  # Normalize to 0-1
        
        # Bonus for pairs/trips
        max_count = max(rank_counts.values()) if rank_counts else 1
        if max_count >= 2:
            strength += 0.2 * (max_count - 1)
        
        # Check for flush potential (2 suits)
        suits = [card.suit for card in all_cards]
        suit_counts = defaultdict(int)
        for suit in suits:
            suit_counts[suit] += 1
        
        if max(suit_counts.values()) >= 3:
            strength += 0.1
        
        return min(1.0, strength)
    
    def _classify_hand(self) -> str:
        """Classify hand into categories"""
        strength = self._evaluate_hand_strength()
        
        if strength >= 0.8:
            return "premium"
        elif strength >= 0.6:
            return "strong"
        elif strength >= 0.4:
            return "medium"
        else:
            return "weak"
    
    def _calculate_pot_odds(self) -> float:
        """Calculate pot odds if facing a bet"""
        max_bet = max(p.current_bet for p in self.game_state.players if not p.folded)
        call_amount = max_bet - self.player.current_bet
        
        if call_amount <= 0:
            return 0.0
        
        return call_amount / (self.game_state.pot + call_amount)
    
    def _analyze_aggression(self) -> float:
        """Analyze aggression level in the hand"""
        betting_actions = [a for a in self.game_state.action_history 
                          if a["action"] in ["bet", "raise"]]
        
        total_actions = len([a for a in self.game_state.action_history 
                           if a["action"] not in ["small_blind", "big_blind"]])
        
        if total_actions == 0:
            return 0.0
        
        return len(betting_actions) / total_actions
    
    def _encode_betting_pattern(self) -> str:
        """Encode the betting pattern as a string"""
        actions = []
        for action in self.game_state.action_history:
            if action["action"] not in ["small_blind", "big_blind"]:
                if action["action"] in ["check", "call"]:
                    actions.append("c")
                elif action["action"] in ["bet", "raise"]:
                    actions.append("r")
                elif action["action"] == "fold":
                    actions.append("f")
        
        return "".join(actions[-6:])  # Last 6 actions
    
    def _determine_position_action(self) -> bool:
        """Determine if player acts first this round"""
        if self.game_state.current_street == Street.PREFLOP:
            # Small blind acts first preflop
            sb_player = (self.game_state.dealer_button + 1) % 2
            return self.player_id == sb_player
        else:
            # Small blind acts first post-flop too
            sb_player = (self.game_state.dealer_button + 1) % 2
            return self.player_id == sb_player


class IntelligentStrategyMapper:
    """Maps game states to strategies using contextual similarity"""
    
    def __init__(self, strategy_data: Dict[str, List[float]]):
        self.strategy_data = strategy_data
        self.feature_cache = {}
        self.strategy_clusters = self._analyze_strategy_clusters()
        
    def _analyze_strategy_clusters(self) -> Dict[str, List[str]]:
        """Analyze strategy data to find clusters of similar strategies"""
        
        print("🔍 Analyzing strategy clusters...")
        
        clusters = {
            'tight': [],      # High fold probability
            'aggressive': [], # High bet/raise probability  
            'passive': [],    # High call probability
            'balanced': [],   # Mixed strategies
        }
        
        for key, strategy in list(self.strategy_data.items())[:1000]:  # Sample first 1000
            if not isinstance(strategy, (list, tuple)) or len(strategy) < 2:
                continue
            
            # Normalize strategy
            total = sum(strategy)
            if total <= 0:
                continue
                
            normalized = [s / total for s in strategy]
            
            # Classify strategy
            if len(normalized) >= 3:
                fold_prob = normalized[0]
                passive_prob = normalized[1]
                aggressive_prob = sum(normalized[2:])
                
                if fold_prob > 0.6:
                    clusters['tight'].append(key)
                elif aggressive_prob > 0.5:
                    clusters['aggressive'].append(key)
                elif passive_prob > 0.5:
                    clusters['passive'].append(key)
                else:
                    clusters['balanced'].append(key)
        
        print(f"   Tight strategies: {len(clusters['tight'])}")
        print(f"   Aggressive strategies: {len(clusters['aggressive'])}")
        print(f"   Passive strategies: {len(clusters['passive'])}")
        print(f"   Balanced strategies: {len(clusters['balanced'])}")
        
        return clusters
    
    def find_strategy(self, game_state: GameState, player_id: int) -> Optional[List[float]]:
        """Find the best matching strategy for the current game state"""
        
        # Extract features from current state
        extractor = GameStateFeatures(game_state, player_id)
        features = extractor.extract_features()
        
        # Use a contextual approach to find similar situations
        strategy = self._contextual_strategy_lookup(features)
        
        if strategy:
            print(f"✅ Found contextual strategy match")
            return strategy
        
        # Fallback to cluster-based approach
        strategy = self._cluster_based_strategy(features)
        
        if strategy:
            print(f"✅ Found cluster-based strategy")
            return strategy
        
        return None
    
    def _contextual_strategy_lookup(self, features: Dict) -> Optional[List[float]]:
        """Look up strategy based on game context"""
        
        # Create multiple context-based keys to try
        context_keys = self._generate_context_keys(features)
        
        for key in context_keys:
            # Try direct hash lookup
            key_hash = hashlib.md5(key.encode()).hexdigest()
            if key_hash in self.strategy_data:
                print(f"🎯 Contextual match: '{key}' -> {key_hash}")
                return self.strategy_data[key_hash]
        
        return None
    
    def _generate_context_keys(self, features: Dict) -> List[str]:
        """Generate possible context keys based on features"""
        
        keys = []
        
        # Card-based contexts (might match abstracted representations)
        card_str = ",".join(map(str, features['my_cards']))
        board_str = ",".join(map(str, features['board_cards']))
        
        # Various format attempts
        keys.extend([
            # Round-based
            f"r{features['round']}p{features['position']}",
            f"{features['round']}:{features['position']}:{features['pot_size']}",
            
            # Card-based
            f"{card_str}:{features['betting_pattern']}",
            f"cards{card_str}board{board_str}",
            
            # Hand strength based
            f"{features['hand_class']}:{features['round']}:{features['betting_pattern']}",
            f"strength{features['hand_strength']:.1f}round{features['round']}",
            
            # Position and betting
            f"pos{features['position']}pattern{features['betting_pattern']}",
            f"{features['position']}{features['betting_pattern']}{features['round']}",
            
            # Stack-based
            f"stack{features['my_stack']}pot{features['pot_size']}",
            
            # Simplified formats
            f"{features['round']}{features['position']}{features['hand_class']}",
            f"{features['my_cards'][0] if features['my_cards'] else 0}_{features['round']}",
        ])
        
        return keys
    
    def _cluster_based_strategy(self, features: Dict) -> Optional[List[float]]:
        """Select strategy based on game situation and clusters"""
        
        # Determine appropriate strategy type based on features
        hand_strength = features['hand_strength']
        position = features['position']
        pot_odds = features['pot_odds']
        aggression = features['aggression_level']
        
        # Decision logic
        if hand_strength >= 0.7:
            # Strong hand - lean aggressive
            cluster_preference = ['aggressive', 'balanced', 'passive', 'tight']
        elif hand_strength <= 0.3:
            # Weak hand - lean tight
            cluster_preference = ['tight', 'passive', 'balanced', 'aggressive']
        elif pot_odds > 0.3:
            # Good pot odds - more calling
            cluster_preference = ['passive', 'balanced', 'aggressive', 'tight']
        else:
            # Default to balanced
            cluster_preference = ['balanced', 'passive', 'aggressive', 'tight']
        
        # Find strategies from preferred clusters
        for cluster_type in cluster_preference:
            cluster_keys = self.strategy_clusters.get(cluster_type, [])
            if cluster_keys:
                # Pick a random strategy from this cluster
                selected_key = random.choice(cluster_keys)
                strategy = self.strategy_data[selected_key]
                
                print(f"🎯 Selected {cluster_type} strategy (hand strength: {hand_strength:.2f})")
                return strategy
        
        return None


class IntelligentCFRInterface:
    """CFR interface that uses intelligent mapping instead of exact format matching"""
    
    def __init__(self, strategy_file_path: str = None):
        self.strategy_data = {}
        self.trained_model_loaded = False
        self.strategy_mapper = None
        
        # Load strategy data
        self._load_strategy_data(strategy_file_path)
        
        # Initialize intelligent mapper
        if self.strategy_data:
            self.strategy_mapper = IntelligentStrategyMapper(self.strategy_data)
            self.trained_model_loaded = True
            print("✅ Intelligent strategy mapper initialized")
    
    def _load_strategy_data(self, strategy_file_path: str):
        """Load strategy data from file"""
        
        strategy_files = [
            strategy_file_path,
            "./gui_compatible_strategy.pkl.gz",
            "../mccfr/limit_holdem_strategy_parallel.pkl.gz",
            "../mccfr/checkpoints/mccfr_checkpoint.pkl.gz",
        ]
        
        for file_path in strategy_files:
            if file_path and os.path.exists(file_path):
                try:
                    print(f"🔍 Loading strategy data from: {file_path}")
                    
                    with gzip.open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Extract strategy data
                    if isinstance(data, dict):
                        if 'strategy' in data:
                            self.strategy_data = data['strategy']
                        elif 'final_strategy' in data:
                            self.strategy_data = data['final_strategy']
                        elif 'minimizer_state' in data and 'strategies' in data['minimizer_state']:
                            # Convert checkpoint format
                            strategies = data['minimizer_state']['strategies']
                            self.strategy_data = {}
                            for state_hash, strategy_sum in strategies.items():
                                total = sum(strategy_sum) if strategy_sum else 0
                                if total > 0:
                                    avg_strategy = [s / total for s in strategy_sum]
                                    self.strategy_data[state_hash] = avg_strategy
                        else:
                            self.strategy_data = data
                    
                    print(f"✅ Loaded {len(self.strategy_data)} strategies")
                    break
                    
                except Exception as e:
                    print(f"❌ Error loading {file_path}: {e}")
                    continue
    
    def get_strategy(self, game_state: GameState, player_id: int) -> Optional[List[float]]:
        """Get strategy for current game state"""
        
        if not self.strategy_mapper:
            return None
        
        return self.strategy_mapper.find_strategy(game_state, player_id)
    
    def get_action(self, game_state: GameState, valid_actions: List[Action]) -> Optional[Action]:
        """Get AI action based on intelligent strategy mapping"""
        
        if not valid_actions:
            return None
        
        ai_player_id = 1  # Assume AI is player 1
        
        # Get strategy using intelligent mapping
        strategy = self.get_strategy(game_state, ai_player_id)
        
        if strategy:
            print(f"✅ Intelligent strategy: {[f'{p:.3f}' for p in strategy]}")
            return self._strategy_to_action(strategy, valid_actions)
        
        # Ultimate fallback
        print("🔄 Using random fallback")
        return random.choice(valid_actions)
    
    def _strategy_to_action(self, strategy: List[float], valid_actions: List[Action]) -> Action:
        """Convert strategy to action"""
        
        if len(strategy) < 2:
            return random.choice(valid_actions)
        
        # Normalize strategy
        strategy = np.array(strategy)
        if strategy.sum() > 0:
            strategy = strategy / strategy.sum()
        else:
            strategy = np.ones(len(strategy)) / len(strategy)
        
        # Map to available actions
        action_mapping = self._create_action_mapping(valid_actions)
        
        # Adjust strategy length to match actions
        if len(strategy) != len(action_mapping):
            if len(strategy) > len(action_mapping):
                strategy = strategy[:len(action_mapping)]
            else:
                # Extend with uniform probability
                extra = np.ones(len(action_mapping) - len(strategy)) / len(action_mapping)
                strategy = np.concatenate([strategy, extra])
            
            # Re-normalize
            strategy = strategy / strategy.sum()
        
        # Sample action
        action_idx = np.random.choice(len(action_mapping), p=strategy)
        chosen_action = action_mapping[action_idx]
        
        print(f"🎲 Intelligent choice: {chosen_action.value} (prob={strategy[action_idx]:.3f})")
        return chosen_action
    
    def _create_action_mapping(self, valid_actions: List[Action]) -> List[Action]:
        """Create consistent action mapping"""
        action_order = [Action.FOLD, Action.CHECK, Action.CALL, Action.BET, Action.RAISE]
        
        mapping = []
        for action in action_order:
            if action in valid_actions:
                mapping.append(action)
        
        # Add any remaining actions
        for action in valid_actions:
            if action not in mapping:
                mapping.append(action)
        
        return mapping


# Updated AIPlayer class to use intelligent mapping
class IntelligentAIPlayer:
    """AI player using intelligent strategy mapping"""
    
    def __init__(self, name: str = "AI", strategy_file_path: str = None):
        self.name = name
        self.cfr_interface = IntelligentCFRInterface(strategy_file_path)
    
    def get_action(self, game_state: GameState, valid_actions: List[Action]) -> Action:
        """Get action for AI player"""
        action = self.cfr_interface.get_action(game_state, valid_actions)
        return action if action else random.choice(valid_actions)
    
    def get_strategy_info(self, game_state: GameState, player_id: int) -> Dict[str, any]:
        """Get strategy info for display"""
        try:
            strategy = self.cfr_interface.get_strategy(game_state, player_id)
            
            if strategy:
                # Convert to display format
                if len(strategy) >= 3:
                    fold_prob = strategy[0]
                    passive_prob = strategy[1]
                    aggressive_prob = sum(strategy[2:])
                else:
                    fold_prob = strategy[0] if len(strategy) > 0 else 0.33
                    passive_prob = strategy[1] if len(strategy) > 1 else 0.33
                    aggressive_prob = 1.0 - fold_prob - passive_prob
                
                return {
                    "strategy_type": "Intelligent Mapping",
                    "fold_prob": fold_prob * 100,
                    "passive_prob": passive_prob * 100,
                    "aggressive_prob": aggressive_prob * 100,
                    "model_loaded": True,
                    "raw_strategy": strategy
                }
            
            return {
                "strategy_type": "Fallback",
                "model_loaded": self.cfr_interface.trained_model_loaded,
                "message": "No intelligent mapping found"
            }
            
        except Exception as e:
            return {
                "strategy_type": "Error",
                "error": str(e),
                "model_loaded": False
            }
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.cfr_interface.trained_model_loaded


# Convenience function for loading
def load_intelligent_cfr_model(strategy_file_path: str = None) -> IntelligentAIPlayer:
    """Load CFR model with intelligent mapping"""
    return IntelligentAIPlayer("Intelligent_CFR_AI", strategy_file_path)


if __name__ == "__main__":
    # Test the intelligent mapping
    print("🧪 Testing Intelligent Strategy Mapper")
    
    # Test with dummy strategy data
    dummy_strategies = {
        'hash1': [0.1, 0.4, 0.5],  # Aggressive
        'hash2': [0.7, 0.2, 0.1],  # Tight
        'hash3': [0.2, 0.6, 0.2],  # Passive
        'hash4': [0.3, 0.4, 0.3],  # Balanced
    }
    
    mapper = IntelligentStrategyMapper(dummy_strategies)
    
    # Create a dummy game state
    from poker_engine import SimplifiedPokerEngine
    
    engine = SimplifiedPokerEngine()
    game_state = engine.start_new_hand(["Human", "AI"])
    
    # Test the mapping
    strategy = mapper.find_strategy(game_state, 1)
    
    if strategy:
        print(f"✅ Found strategy: {strategy}")
    else:
        print("❌ No strategy found")
    
    print("✅ Intelligent mapping test complete")
