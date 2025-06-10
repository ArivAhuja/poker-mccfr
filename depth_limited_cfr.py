from cfr import BaseCFR
import copy
import numpy as np

class DepthLimitedCFR:
    """
    Depth-Limited Search CFR Implementation for Imperfect Information Games
    
    This class implements depth-limited search techniques that allow real-time strategy
    computation during gameplay, rather than relying solely on pre-computed strategies.
    Based on the multi-valued states approach from Brown, Sandholm & Amos (2018).
    
    The key innovation is that at depth limits, opponents choose from multiple 
    continuation strategies, forcing our strategy to be robust to opponent adaptation.
    
    Arguments:
        base_cfr (BaseCFR): The underlying CFR implementation (e.g., KuhnVanillaCFR)
        depth_limit (int): Maximum tree depth before triggering real-time search
        blueprint_strategy (dict): Pre-trained coarse strategy for full game
        continuation_strategies (dict): k=4 different opponent strategies for depth limits
        search_cache (dict): Cache for repeated subgame computations
    """
    
    def __init__(self, base_cfr: BaseCFR, depth_limit: int):
            """
            Initialize the Depth-Limited CFR controller.
        
            :param base_cfr: Trained CFR instance (VanillaCFR or MonteCarloCFR)
            :param depth_limit: When to switch from blueprint to real-time search
            """
            self.base_cfr = base_cfr
            self.depth_limit = depth_limit
            self.blueprint_strategy = {}      # Pre-trained strategy for entire game
            self.continuation_strategies = {} # k=4 different opponent strategies  
            self.search_cache = {}           # Cache repeated subgame solutions
            self.num_players = 3 # track num of players for strategy combinations

    def generate_blueprint_strategy(self, iterations):
        """
        Generate the blueprint strategy using standard CFR training.
        
        The blueprint is a coarse-grained strategy for the entire game, computed
        offline using your existing CFR implementation. This serves as the baseline
        strategy that will be improved via real-time depth-limited search.
        
        Storage format: {info_set_string: [action_0_prob, action_1_prob, ...]}
        
        :param iterations: Number of CFR iterations to train the blueprint
        """
        # Train the underlying CFR to convergence
        self.base_cfr.train(iterations)

        # Extract average strategy from all discovered information sets
        for info_set in self.base_cfr.strategy_sum:
            # Get the converged average strategy for this information set
            strategy_array = self.base_cfr.get_average_strategy(info_set)
            # Store as list for easy JSON serialization and lookup
            self.blueprint_strategy[info_set] = strategy_array.tolist()

    def generate_continuation_strategies(self, biased_multiplier):
        """
        Generate k=4 different continuation strategies from the blueprint.
        
        These strategies represent different opponent "types" that could be chosen
        at depth limits during real-time search. By forcing opponents to choose
        between these strategies, our search finds robust solutions.
        
        Strategy types:
        - blueprint: Original unmodified strategy  
        - fold_heavy: Biased toward folding/passing (action 0)
        - call_heavy: Biased toward calling/checking (action 1)  
        - raise_heavy: Biased toward betting/raising (action 2+)
        
        Mathematical approach: Multiply target action probabilities by bias factor,
        then renormalize to maintain valid probability distribution.
        
        :param biased_multiplier: Factor to amplify target action probabilities (e.g., 5.0)
        """
        
        # Strategy 1: Exact copy of blueprint (baseline)
        self.continuation_strategies['blueprint'] = copy.deepcopy(self.blueprint_strategy)

        # Strategy 2: Fold-heavy (bias action 0 = fold/pass)
        self.continuation_strategies['fold_heavy'] = {}
        for info_set in self.blueprint_strategy:
            # Create modifiable copy of original probabilities
                original_probs = self.blueprint_strategy[info_set].copy()
            
                if len(original_probs) >= 1:
                    # Amplify fold/pass probability by bias factor
                    original_probs[0] *= biased_multiplier
                    # Renormalize to maintain probability constraints (sum = 1.0)
                    final_prob = np.array(original_probs) / sum(original_probs)
                    self.continuation_strategies['fold_heavy'][info_set] = final_prob.tolist()
                else:
                    self.continuation_strategies['fold_heavy'][info_set] = original_probs


        # Strategy 3: Call-heavy (bias action 1 = call/check)  
        self.continuation_strategies['call_heavy'] = {}
        for info_set in self.blueprint_strategy:
            original_probs = self.blueprint_strategy[info_set].copy()

            if len(original_probs) >= 2:
                    # Amplify call/check probability by bias factor
                    original_probs[1] *= biased_multiplier
                    # Renormalize probabilities
                    final_prob = np.array(original_probs) / sum(original_probs)
                    self.continuation_strategies['call_heavy'][info_set] = final_prob.tolist()

            else:
                    self.continuation_strategies['call_heavy'][info_set] = original_probs
            
        # Strategy 4: Raise-heavy (bias action 2 = raise/bet)
        # Now properly implemented for 3+ action games like Limit Hold'em
        self.continuation_strategies['raise_heavy'] = {}
        for info_set in self.blueprint_strategy:
            original_probs = self.blueprint_strategy[info_set].copy()
            
            # Check if this info set has raise action available
            if len(original_probs) >= 3:
                # Amplify raise/bet probability by bias factor
                original_probs[2] *= biased_multiplier
                # Renormalize probabilities
                final_prob = np.array(original_probs) / sum(original_probs)
                self.continuation_strategies['raise_heavy'][info_set] = final_prob.tolist()
            else:
                # For info sets with only fold/call, make it same as call_heavy
                if len(original_probs) >= 2:
                    original_probs[1] *= biased_multiplier
                    final_prob = np.array(original_probs) / sum(original_probs)
                    self.continuation_strategies['raise_heavy'][info_set] = final_prob.tolist()
                else:
                    self.continuation_strategies['raise_heavy'][info_set] = original_probs
        # Debug output to verify biasing worked correctly for 3-action game
        # Find a sample info set with 3 actions for testing
        sample_info_set = None
        for info_set, probs in self.blueprint_strategy.items():
            if len(probs) >= 3:
                sample_info_set = info_set
                break
        
        if sample_info_set:
            print(f"3-action info set '{sample_info_set}':")
            print(f"Blueprint: {self.blueprint_strategy[sample_info_set]}")
            print(f"Fold_heavy: {self.continuation_strategies['fold_heavy'][sample_info_set]}")
            print(f"Call_heavy: {self.continuation_strategies['call_heavy'][sample_info_set]}")
            print(f"Raise_heavy: {self.continuation_strategies['raise_heavy'][sample_info_set]}")
        else:
            # Fallback to any available info set
            first_info_set = list(self.blueprint_strategy.keys())[0]
            print(f"Sample info set '{first_info_set}':")
            print(f"Blueprint: {self.blueprint_strategy[first_info_set]}")
            print(f"Fold_heavy: {self.continuation_strategies['fold_heavy'][first_info_set]}")
