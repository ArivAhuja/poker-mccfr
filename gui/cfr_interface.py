"""
CFR Interface to load and use your trained MCCFR models
FIXED VERSION - Properly handles MD5-hashed information state keys like your MCCFR training
"""

import sys
import os
import pickle
import gzip
import hashlib
import random
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

from poker_engine import Action, SimplifiedPokerEngine, GameState


class AIPersonality(Enum):
    """Different AI playing styles"""
    TRAINED_CFR = "trained_cfr"
    TIGHT = "tight"
    AGGRESSIVE = "aggressive" 
    BALANCED = "balanced"
    RANDOM = "random"


class TrainedCFRInterface:
    """
    Interface to load and use your trained CFR models
    FIXED: Properly handles MD5-hashed strategy keys from MCCFR training
    """
    
    def __init__(self, strategy_file_path: str = None):
        self.strategy_data = {}
        self.trained_model_loaded = False
        self.strategy_file_path = strategy_file_path
        
        # Try to load your trained model
        self._load_trained_strategy()
        
        # Fallback strategies for when CFR model doesn't have coverage
        self.fallback_strategies = {
            "tight": {"fold_prob": 0.7, "call_prob": 0.25, "bet_prob": 0.05},
            "aggressive": {"fold_prob": 0.3, "call_prob": 0.3, "bet_prob": 0.4},
            "balanced": {"fold_prob": 0.4, "call_prob": 0.4, "bet_prob": 0.2}
        }
    
    def _load_trained_strategy(self):
        """Load your trained CFR strategy from .pkl.gz file"""
        strategy_files = [
            self.strategy_file_path,  # User-provided path
            "./gui_compatible_strategy.pkl.gz",  # NEW: Your extracted file
            "../mccfr/limit_holdem_strategy_parallel.pkl.gz",  # From gui/ to mccfr/
            "../../mccfr/limit_holdem_strategy_parallel.pkl.gz",  # Alternative path  
            "../mccfr/checkpoints/mccfr_checkpoint.pkl.gz",  # Fallback checkpoint
            "../../mccfr/checkpoints/mccfr_checkpoint.pkl.gz",  # Alternative checkpoint
        ]
        
        for file_path in strategy_files:
            if file_path and os.path.exists(file_path):
                try:
                    print(f"🔍 Loading trained CFR strategy from: {file_path}")
                    
                    # Try loading with different methods to handle pickle issues
                    data = self._safe_pickle_load(file_path)
                    if data is None:
                        continue
                    
                    print(f"📋 File contents keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                    
                    # Handle different file formats
                    if isinstance(data, dict) and 'strategy' in data:
                        # This is the format from your MCCFR training: {'strategy': {...}, 'iteration': ..., 'config': ...}
                        self.strategy_data = data['strategy']
                        print(f"✅ Loaded strategy dictionary with {len(self.strategy_data)} info states")
                        
                        if 'iteration' in data:
                            print(f"   Training iterations: {data['iteration']:,}")
                        if 'elapsed_time' in data:
                            print(f"   Training time: {data['elapsed_time']:.1f} seconds")
                            
                    elif isinstance(data, dict) and 'final_strategy' in data:
                        # Final strategy file format 
                        self.strategy_data = data['final_strategy']
                        print(f"✅ Loaded final strategy with {len(self.strategy_data)} info states")
                        
                        if 'iterations' in data:
                            print(f"   Training iterations: {data['iterations']:,}")
                        if 'training_time' in data:
                            print(f"   Training time: {data['training_time']:.1f} seconds")
                        if 'exploitability' in data:
                            print(f"   Final exploitability: {data['exploitability']:.6f}")
                            
                    elif isinstance(data, dict) and 'minimizer_state' in data:
                        # Checkpoint file format from your MCCFR training
                        print("🔄 Converting MCCFR checkpoint to usable strategies...")
                        minimizer_state = data['minimizer_state']
                        
                        if 'strategies' in minimizer_state:
                            strategy_sums = minimizer_state['strategies']
                            self.strategy_data = {}
                            
                            for state_hash, strategy_sum in strategy_sums.items():
                                total = sum(strategy_sum) if strategy_sum else 0
                                if total > 0:
                                    avg_strategy = [s / total for s in strategy_sum]
                                    # Only save non-uniform strategies (like your MCCFR code does)
                                    if not all(abs(s - avg_strategy[0]) < 1e-6 for s in avg_strategy):
                                        self.strategy_data[state_hash] = avg_strategy
                            
                            print(f"✅ Converted checkpoint to {len(self.strategy_data)} strategies")
                        
                        if 'iteration' in data:
                            print(f"   Training iterations: {data['iteration']:,}")
                    
                    elif isinstance(data, dict):
                        # Direct strategy dictionary format (assume it's already hashed keys)
                        print("🔄 Loading direct strategy dictionary...")
                        self.strategy_data = data
                        print(f"✅ Loaded {len(self.strategy_data)} strategies")
                    
                    else:
                        print(f"⚠️  Unknown file format in {file_path}: {type(data)}")
                        continue
                    
                    self.trained_model_loaded = True
                    print(f"🧠 CFR model successfully loaded!")
                    
                    # Show some example strategies to verify format
                    self._log_example_strategies()
                    break
                    
                except Exception as e:
                    print(f"❌ Error loading {file_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        if not self.trained_model_loaded:
            print("⚠️  No trained CFR model found. Using fallback strategies.")
            print("   Expected files:")
            for file_path in strategy_files:
                if file_path:
                    print(f"   - {file_path} {'✅' if os.path.exists(file_path) else '❌'}")
    
    def _safe_pickle_load(self, file_path: str):
        """Safely load pickle file with different methods to handle missing classes"""
        
        # Method 1: Try normal pickle loading
        try:
            with gzip.open(file_path, 'rb') as f:
                return pickle.load(f)
        except AttributeError as e:
            if "Config" in str(e):
                print(f"⚠️  Normal pickle loading failed due to missing Config class")
            else:
                print(f"⚠️  Normal pickle loading failed: {e}")
        except Exception as e:
            print(f"⚠️  Normal pickle loading failed: {e}")
        
        # Method 2: Try with custom unpickler that ignores missing classes
        try:
            print("🔄 Trying safe unpickler...")
            
            class SafeUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # If it's the problematic Config class, create a dummy one
                    if name == 'Config':
                        # Return a simple class that can hold any attributes
                        class DummyConfig:
                            def __init__(self, **kwargs):
                                for k, v in kwargs.items():
                                    setattr(self, k, v)
                            def __setstate__(self, state):
                                self.__dict__.update(state)
                        return DummyConfig
                    
                    # For other missing classes, try to load normally first
                    try:
                        return super().find_class(module, name)
                    except (ImportError, AttributeError):
                        # If class not found, create a dummy class
                        print(f"⚠️  Creating dummy class for {module}.{name}")
                        
                        class DummyClass:
                            def __init__(self, *args, **kwargs):
                                self.args = args
                                self.kwargs = kwargs
                            def __setstate__(self, state):
                                self.__dict__.update(state)
                                
                        return DummyClass
            
            with gzip.open(file_path, 'rb') as f:
                return SafeUnpickler(f).load()
                
        except Exception as e:
            print(f"⚠️  Safe unpickler failed: {e}")
            return None
    
    def _log_example_strategies(self):
        """Log some example strategies for debugging"""
        if not self.strategy_data:
            return
            
        print("\n📊 Example strategies from loaded model:")
        sample_size = min(5, len(self.strategy_data))
        sample_keys = list(self.strategy_data.keys())[:sample_size]
        
        for key in sample_keys:
            strategy = self.strategy_data[key]
            if isinstance(strategy, list):
                print(f"   {key}: {[f'{p:.3f}' for p in strategy]}")
            else:
                print(f"   {key}: {strategy} (type: {type(strategy)})")
        
        # Show key format information
        if sample_keys:
            example_key = sample_keys[0]
            print(f"\n🔑 Key format analysis:")
            print(f"   Example key: '{example_key}'")
            print(f"   Key length: {len(example_key)}")
            print(f"   Is hex (MD5-like): {all(c in '0123456789abcdef' for c in example_key)}")
            if len(example_key) == 32 and all(c in '0123456789abcdef' for c in example_key):
                print(f"   ✅ Keys appear to be MD5 hashes (32 hex chars)")
        print()
    
    def get_strategy_for_info_state(self, info_state: str) -> Optional[List[float]]:
        """
        Get strategy for a given information state string
        FIXED: Properly computes MD5 hash to match your MCCFR training
        """
        if not self.trained_model_loaded or not self.strategy_data:
            return None
        
        print(f"🔍 Looking up strategy for: '{info_state}'")
        
        # FIXED: The key step - compute MD5 hash exactly like your MCCFR training does
        info_state_hash = hashlib.md5(info_state.encode()).hexdigest()
        print(f"🔑 Computed MD5 hash: '{info_state_hash}'")
        
        # Try MD5 hash lookup first (this should work with your model)
        if info_state_hash in self.strategy_data:
            strategy = self.strategy_data[info_state_hash]
            print(f"✅ Hash match found: {strategy}")
            return strategy if isinstance(strategy, list) else None
        
        # Try direct lookup (unlikely to work but worth trying)
        if info_state in self.strategy_data:
            strategy = self.strategy_data[info_state]
            print(f"✅ Direct match found: {strategy}")
            return strategy if isinstance(strategy, list) else None
        
        # Debug: Show some example keys to help troubleshoot
        print(f"❌ No exact match found. Checking partial matches...")
        example_keys = list(self.strategy_data.keys())[:3]
        print(f"   Example keys in strategy data: {example_keys}")
        
        # Check if any keys contain the hash or info state as substring
        partial_matches = []
        for key in list(self.strategy_data.keys())[:20]:  # Check first 20 keys
            if info_state_hash in str(key) or str(key) in info_state_hash:
                partial_matches.append(key)
        
        if partial_matches:
            print(f"   Partial hash matches: {partial_matches[:3]}...")
        
        return None
    
    def get_ai_action(self, game_state: GameState, valid_actions: List[Action]) -> Action:
        """
        Get AI action based on trained CFR strategy or fallback
        """
        if not valid_actions:
            return None
        
        ai_player_id = 1  # Assume AI is player 1
        
        # Try to get action from trained CFR model
        if self.trained_model_loaded:
            cfr_action = self._get_cfr_action(game_state, ai_player_id, valid_actions)
            if cfr_action:
                return cfr_action
        
        # Fall back to simple strategy
        print("🔄 Using fallback strategy")
        return self._get_fallback_action(game_state, ai_player_id, valid_actions)
    
    def _get_cfr_action(self, game_state: GameState, player_id: int, valid_actions: List[Action]) -> Optional[Action]:
        """Get action using trained CFR model"""
        try:
            # Get info state from the poker engine
            from poker_engine import SimplifiedPokerEngine
            engine = SimplifiedPokerEngine()
            engine.game_state = game_state  # Set current state
            info_state = engine.get_information_set(player_id)
            
            print(f"🤖 AI analyzing info state: {info_state}")
            
            # Get strategy from trained model
            strategy = self.get_strategy_for_info_state(info_state)
            
            if strategy is None:
                print(f"❌ No CFR strategy found for info state")
                return None
            
            print(f"✅ CFR strategy found: {[f'{p:.3f}' for p in strategy]}")
            
            # Convert strategy probabilities to actions
            return self._strategy_to_action(strategy, valid_actions)
        
        except Exception as e:
            print(f"❌ Error in CFR action selection: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _strategy_to_action(self, strategy: List[float], valid_actions: List[Action]) -> Action:
        """
        Convert CFR strategy probabilities to poker actions
        Enhanced version with better action mapping
        """
        if len(strategy) < 2:
            print(f"⚠️  Invalid strategy length: {len(strategy)}, using random action")
            return random.choice(valid_actions)
        
        print(f"🎯 Converting strategy {[f'{p:.3f}' for p in strategy]} to action from {[a.value for a in valid_actions]}")
        
        # Your MCCFR model likely uses a standard CFR action ordering
        # Most common formats: [fold, call/check, bet/raise] or [fold, call, bet, raise]
        
        # Normalize strategy to ensure it sums to 1
        strategy = np.array(strategy)
        if strategy.sum() > 0:
            strategy = strategy / strategy.sum()
        else:
            strategy = np.ones(len(strategy)) / len(strategy)
        
        # Map strategy indices to actions based on valid actions
        action_mapping = self._create_action_mapping(valid_actions)
        
        if len(strategy) != len(action_mapping):
            print(f"⚠️  Strategy length ({len(strategy)}) != action mapping length ({len(action_mapping)})")
            # Try to adapt
            if len(strategy) > len(action_mapping):
                strategy = strategy[:len(action_mapping)]
            else:
                # Pad with uniform probability
                padding = np.ones(len(action_mapping) - len(strategy)) / len(action_mapping)
                strategy = np.concatenate([strategy, padding])
            
            # Re-normalize
            strategy = strategy / strategy.sum()
        
        print(f"📊 Action mapping: {[(action_mapping[i].value, f'{strategy[i]:.3f}') for i in range(len(action_mapping))]}")
        
        # Sample action based on strategy
        action_idx = np.random.choice(len(action_mapping), p=strategy)
        chosen_action = action_mapping[action_idx]
        
        print(f"🎲 Sampled action: {chosen_action.value} (prob={strategy[action_idx]:.3f})")
        return chosen_action
    
    def _create_action_mapping(self, valid_actions: List[Action]) -> List[Action]:
        """
        Create consistent action mapping for CFR strategy
        Maps strategy indices to actual actions in a consistent order
        """
        # Standard poker action order (matches most CFR implementations)
        action_order = [Action.FOLD, Action.CHECK, Action.CALL, Action.BET, Action.RAISE]
        
        # Create mapping of available actions in standard order
        action_mapping = []
        for action in action_order:
            if action in valid_actions:
                action_mapping.append(action)
        
        # Add any remaining actions not in standard order
        for action in valid_actions:
            if action not in action_mapping:
                action_mapping.append(action)
        
        return action_mapping
    
    def _get_fallback_action(self, game_state: GameState, player_id: int, valid_actions: List[Action]) -> Action:
        """Get action using simple fallback strategy"""
        if not valid_actions:
            return None
        
        # Use balanced strategy as fallback
        strategy = self.fallback_strategies["balanced"]
        
        # Simple hand strength evaluation
        hand_strength = self._evaluate_hand_strength(game_state, player_id)
        
        # Adjust probabilities based on hand strength
        if hand_strength > 0.7:  # Strong hand
            fold_prob = strategy["fold_prob"] * 0.3
            bet_prob = strategy["bet_prob"] * 1.5
        elif hand_strength < 0.3:  # Weak hand
            fold_prob = strategy["fold_prob"] * 1.5
            bet_prob = strategy["bet_prob"] * 0.3
        else:  # Medium hand
            fold_prob = strategy["fold_prob"]
            bet_prob = strategy["bet_prob"]
        
        call_prob = 1.0 - fold_prob - bet_prob
        
        # Choose action based on probabilities
        rand = random.random()
        
        if rand < fold_prob and Action.FOLD in valid_actions:
            return Action.FOLD
        elif rand < fold_prob + bet_prob:
            # Try aggressive actions
            if Action.BET in valid_actions:
                return Action.BET
            elif Action.RAISE in valid_actions:
                return Action.RAISE
            elif Action.CALL in valid_actions:
                return Action.CALL
            elif Action.CHECK in valid_actions:
                return Action.CHECK
        else:
            # Try passive actions
            if Action.CALL in valid_actions:
                return Action.CALL
            elif Action.CHECK in valid_actions:
                return Action.CHECK
            elif Action.BET in valid_actions:
                return Action.BET
        
        # Last resort
        return valid_actions[0]
    
    def _evaluate_hand_strength(self, game_state: GameState, player_id: int) -> float:
        """
        Simple hand strength evaluation (0.0 to 1.0)
        For simplified 6-card game
        """
        player = game_state.players[player_id]
        
        # Very simple evaluation based on highest card and pairs
        hole_ranks = [card.rank for card in player.hole_cards]
        community_ranks = [card.rank for card in game_state.community_cards]
        all_ranks = hole_ranks + community_ranks
        
        if not all_ranks:
            return 0.5
        
        highest_rank = max(all_ranks)
        
        # Check for pairs
        rank_counts = {}
        for rank in all_ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        has_pair = any(count >= 2 for count in rank_counts.values())
        
        # Basic strength calculation (ranks 2-7, so normalize to 0-1)
        strength = (highest_rank - 2) / 5.0  # 2-7 -> 0-1
        
        if has_pair:
            strength += 0.3  # Bonus for pair
        
        # Add some randomness for variety
        strength += random.uniform(-0.05, 0.05)
        
        return max(0.0, min(1.0, strength))
    
    def get_strategy_info(self, game_state: GameState, player_id: int) -> Dict[str, any]:
        """Get strategy information for display in GUI"""
        try:
            if self.trained_model_loaded:
                # Get info state and strategy from CFR model
                from poker_engine import SimplifiedPokerEngine
                engine = SimplifiedPokerEngine()
                engine.game_state = game_state
                info_state = engine.get_information_set(player_id)
                
                strategy = self.get_strategy_for_info_state(info_state)
                
                if strategy:
                    # Convert strategy to probabilities for display
                    # Adapt to different strategy lengths
                    if len(strategy) >= 3:
                        fold_prob = strategy[0] 
                        passive_prob = strategy[1] 
                        aggressive_prob = sum(strategy[2:])  # Sum remaining as aggressive
                    elif len(strategy) == 2:
                        fold_prob = strategy[0]
                        passive_prob = strategy[1] * 0.6  # Assume 60% passive
                        aggressive_prob = strategy[1] * 0.4  # Assume 40% aggressive
                    else:
                        fold_prob = 0.33
                        passive_prob = 0.33
                        aggressive_prob = 0.34
                    
                    return {
                        "strategy_type": "Trained CFR",
                        "info_state": info_state,
                        "fold_prob": fold_prob * 100,
                        "passive_prob": passive_prob * 100,
                        "aggressive_prob": aggressive_prob * 100,
                        "model_loaded": True,
                        "raw_strategy": strategy
                    }
            
            # Fallback to hand strength evaluation
            hand_strength = self._evaluate_hand_strength(game_state, player_id)
            return {
                "strategy_type": "Fallback",
                "hand_strength": hand_strength * 100,
                "model_loaded": self.trained_model_loaded
            }
            
        except Exception as e:
            print(f"Error getting strategy info: {e}")
            return {
                "strategy_type": "Error",
                "error": str(e),
                "model_loaded": self.trained_model_loaded
            }


class AIPlayer:
    """AI player that uses trained CFR interface"""
    
    def __init__(self, name: str = "AI", strategy_file_path: str = None):
        self.name = name
        self.cfr_interface = TrainedCFRInterface(strategy_file_path)
    
    def get_action(self, game_state: GameState, valid_actions: List[Action]) -> Action:
        """Get action for AI player"""
        return self.cfr_interface.get_ai_action(game_state, valid_actions)
    
    def get_strategy_info(self, game_state: GameState, player_id: int) -> Dict[str, any]:
        """Get strategy info for display"""
        return self.cfr_interface.get_strategy_info(game_state, player_id)
    
    def is_model_loaded(self) -> bool:
        """Check if trained model is loaded"""
        return self.cfr_interface.trained_model_loaded


# Convenience function to load CFR model
def load_cfr_model(strategy_file_path: str = None) -> AIPlayer:
    """
    Load your trained CFR model
    
    Args:
        strategy_file_path: Path to your .pkl.gz strategy file
        
    Returns:
        AIPlayer instance with loaded CFR model
    """
    return AIPlayer("CFR_AI", strategy_file_path)


if __name__ == "__main__":
    # Test the CFR interface
    print("🧪 Testing CFR Interface")
    
    # Test loading model
    ai_player = AIPlayer("TestAI")
    print(f"Model loaded: {ai_player.is_model_loaded()}")
    
    # Test with dummy game state
    from poker_engine import SimplifiedPokerEngine
    
    engine = SimplifiedPokerEngine()
    game_state = engine.start_new_hand(["Human", "AI"])
    
    valid_actions = engine.get_valid_actions()
    print(f"Valid actions: {[a.value for a in valid_actions]}")
    
    if valid_actions:
        ai_action = ai_player.get_action(game_state, valid_actions)
        print(f"AI chooses: {ai_action.value}")
        
        strategy_info = ai_player.get_strategy_info(game_state, 1)
        print(f"Strategy info: {strategy_info}")
    
    print("✅ CFR Interface test complete")
