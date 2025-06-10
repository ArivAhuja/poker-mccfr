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
       - raise_heavy: Biased toward betting/raising (action 2+, or action 1 in Kuhn)
       
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
           # Amplify fold/pass probability by bias factor
           original_probs[0] *= biased_multiplier
           # Renormalize to maintain probability constraints (sum = 1.0)
           final_prob = np.array(original_probs) / sum(original_probs)
           self.continuation_strategies['fold_heavy'][info_set] = final_prob.tolist()

       # Strategy 3: Call-heavy (bias action 1 = call/check)  
       self.continuation_strategies['call_heavy'] = {}
       for info_set in self.blueprint_strategy:
           original_probs = self.blueprint_strategy[info_set].copy()
           # Amplify call/check probability by bias factor
           original_probs[1] *= biased_multiplier
           # Renormalize probabilities
           final_prob = np.array(original_probs) / sum(original_probs)
           self.continuation_strategies['call_heavy'][info_set] = final_prob.tolist()

       # Strategy 4: Raise-heavy (for Kuhn Poker, same as call_heavy since only 2 actions)
       # In games with 3+ actions, this would bias action 2 (raise/bet)
       self.continuation_strategies['raise_heavy'] = copy.deepcopy(self.continuation_strategies['call_heavy'])

       # Debug output to verify biasing worked correctly
       print("Blueprint action 0 prob:", self.blueprint_strategy['0'][0])
       print("Fold_heavy action 0 prob:", self.continuation_strategies['fold_heavy']['0'][0])
