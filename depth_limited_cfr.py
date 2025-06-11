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

        

    def should_trigger_search(self, state, current_depth, betting_round=None):
        """
        Determine whether to use blueprint strategy or trigger depth-limited search.
        
        Uses available state information to make intelligent decisions about when
        expensive real-time search will provide the most benefit.
        
        :param state: Current OpenSpiel game state
        :param current_depth: How deep we are in current decision tree  
        :param betting_round: Optional betting round info (if available)
        :return: Boolean - True if should use depth-limited search
        """
        
        # Priority 1: Never search terminal/chance nodes
        if state.is_terminal() or state.current_player() == -1:
            return False
        
        # Priority 2: Always search if too deep
        if current_depth >= self.depth_limit:
            return True

        # Priority 3: Always search if many actions available (complex decision)
        if hasattr(state, 'legal_actions'):
            num_legal_actions = len(state.legal_actions())
            if num_legal_actions >= 5:
                return True
        
        # Priority 4: Use history length as final criterion
        history_length = len(state.history()) if hasattr(state, 'history') else 0
        return history_length > 9 # search after preflop

    def create_subgame(self, root_state, depth_limit):
        """
        Create a subgame for real-time depth-limited solving.
        
        This method builds a game tree starting from root_state and stopping
        at specified boundaries. The leaf nodes become strategy selection points
        where opponents choose continuation strategies.
        
        :param root_state: Game state where subgame begins
        :param depth_limit: Maximum depth before creating leaf nodes
        :return: Dictionary containing subgame structure
        """
        
        subgame = {
            'root': root_state.clone(),
            'nodes': {},           # state_id -> state_info mapping
            'leaf_nodes': set(),   # Set of leaf node IDs
            'edges': {},           # state_id -> {action: next_state_id}
            'node_counter': 0      # To assign unique IDs
        }
        
        def get_state_id(state):
            """Generate unique ID for a game state based on history."""
            return str(state.history())
        
        def add_node_to_subgame(state, depth, is_leaf=False):
            """Add a state to the subgame structure."""
            state_id = get_state_id(state)
            
            if state_id not in subgame['nodes']:
                subgame['nodes'][state_id] = {
                    'state': state.clone(),
                    'depth': depth,
                    'player': state.current_player(),
                    'is_terminal': state.is_terminal(),
                    'legal_actions': state.legal_actions() if not state.is_terminal() else []
                }
                subgame['edges'][state_id] = {}
            
            if is_leaf:
                subgame['leaf_nodes'].add(state_id)
            
            return state_id
        
        def should_create_leaf(state, depth):
            """Determine if we should stop expanding and create a leaf node."""
            
            # Always stop at terminal states
            if state.is_terminal():
                return True
            
            # Stop if we've reached depth limit
            if depth >= depth_limit:
                return True
            
            # Stop if we've reached a reasonable subgame size (prevent explosion)
            if len(subgame['nodes']) >= 1000:  # Computational limit
                return True
            
            # For Hold'em: could add betting round completion logic here
            # Example: if end_of_betting_round(state):
            #     return True
            
            return False
        
        def build_subgame_recursive(current_state, current_depth):
            """Recursively build the subgame tree."""
            
            current_id = get_state_id(current_state)
            
            # Check if we should create a leaf node here
            if should_create_leaf(current_state, current_depth):
                add_node_to_subgame(current_state, current_depth, is_leaf=True)
                return current_id
            
            # Add current node to subgame
            add_node_to_subgame(current_state, current_depth, is_leaf=False)
            
            # Explore all legal actions from this state
            for action in current_state.legal_actions():
                # Create next state
                next_state = current_state.clone()
                next_state.apply_action(action)
                
                # Recursively build from next state
                next_id = build_subgame_recursive(next_state, current_depth + 1)
                
                # Record the edge (action leads to next state)
                subgame['edges'][current_id][action] = next_id
            
            return current_id
        
        # Build the subgame starting from root
        root_id = build_subgame_recursive(root_state, 0)
        subgame['root_id'] = root_id
        
        return subgame

        



