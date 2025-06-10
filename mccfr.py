import pyspiel
import numpy as np
from collections import defaultdict
import random

class MCCFR_P:
    def __init__(self, game, prune_threshold_iterations=20000, 
                 strategy_interval=10000, discount_interval=1000,
                 lcfr_threshold=40000, prune_probability=0.95,
                 regret_floor=-310000000):
        self.game = game
        self.num_players = game.num_players()
        self.prune_threshold = prune_threshold_iterations
        self.strategy_interval = strategy_interval
        self.discount_interval = discount_interval
        self.lcfr_threshold = lcfr_threshold
        self.prune_probability = prune_probability
        self.regret_floor = regret_floor
        
        # Storage for regrets and average strategy
        self.regrets = defaultdict(lambda: defaultdict(float))
        self.avg_strategy = defaultdict(lambda: defaultdict(float))
        
    def calculate_strategy(self, info_state, legal_actions):
        """Calculate current strategy based on regrets using regret matching"""
        regret_sum = 0.0
        strategy = {}
        
        # Sum positive regrets
        for action in legal_actions:
            regret_sum += max(0, self.regrets[info_state][action])
        
        # Calculate strategy
        for action in legal_actions:
            if regret_sum > 0:
                strategy[action] = max(0, self.regrets[info_state][action]) / regret_sum
            else:
                strategy[action] = 1.0 / len(legal_actions)
                
        return strategy
    
    def update_strategy(self, state, player):
        """Update average strategy for the given player"""
        if state.is_terminal():
            return
        
        if state.is_chance_node():
            # Sample from chance distribution
            outcomes = state.chance_outcomes()
            action, prob = zip(*outcomes)
            action = np.random.choice(action, p=prob)
            self.update_strategy(state.child(action), player)
            return
        
        current_player = state.current_player()
        
        if current_player == player:
            info_state = state.information_state_string(player)
            legal_actions = state.legal_actions()
            
            # Get current strategy
            strategy = self.calculate_strategy(info_state, legal_actions)
            
            # Sample action according to strategy
            actions = list(strategy.keys())
            probs = list(strategy.values())
            action = np.random.choice(actions, p=probs)
            
            # Update average strategy (only on first betting round in Pluribus)
            # For simplicity, we update always but you could add betting round check
            self.avg_strategy[info_state][action] += 1
            
            self.update_strategy(state.child(action), player)
        else:
            # Traverse all opponent actions
            for action in state.legal_actions():
                self.update_strategy(state.child(action), player)
    
    def traverse_mccfr(self, state, player):
        """Standard MCCFR traversal"""
        if state.is_terminal():
            return state.returns()[player]
        
        if state.is_chance_node():
            # Sample from chance distribution
            outcomes = state.chance_outcomes()
            action, prob = zip(*outcomes)
            action = np.random.choice(action, p=prob)
            return self.traverse_mccfr(state.child(action), player)
        
        current_player = state.current_player()
        info_state = state.information_state_string(current_player)
        legal_actions = state.legal_actions()
        
        if current_player == player:
            # Get strategy for this infoset
            strategy = self.calculate_strategy(info_state, legal_actions)
            
            # Initialize expected value
            expected_value = 0.0
            action_values = {}
            
            # Get value for each action
            for action in legal_actions:
                action_values[action] = self.traverse_mccfr(state.child(action), player)
                expected_value += strategy[action] * action_values[action]
            
            # Update regrets
            for action in legal_actions:
                regret = action_values[action] - expected_value
                self.regrets[info_state][action] += regret
                # Apply regret floor
                self.regrets[info_state][action] = max(self.regret_floor, 
                                                      self.regrets[info_state][action])
            
            return expected_value
        else:
            # Sample opponent action
            strategy = self.calculate_strategy(info_state, legal_actions)
            actions = list(strategy.keys())
            probs = list(strategy.values())
            action = np.random.choice(actions, p=probs)
            return self.traverse_mccfr(state.child(action), player)
    
    def traverse_mccfr_p(self, state, player):
        """MCCFR with pruning for very negative regrets"""
        if state.is_terminal():
            return state.returns()[player]
        
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action, prob = zip(*outcomes)
            action = np.random.choice(action, p=prob)
            return self.traverse_mccfr_p(state.child(action), player)
        
        current_player = state.current_player()
        info_state = state.information_state_string(current_player)
        legal_actions = state.legal_actions()
        
        if current_player == player:
            strategy = self.calculate_strategy(info_state, legal_actions)
            
            expected_value = 0.0
            action_values = {}
            explored = {}
            
            # Prune actions with very negative regret
            prune_threshold = -300000000  # From Pluribus paper
            
            for action in legal_actions:
                if self.regrets[info_state][action] > prune_threshold:
                    action_values[action] = self.traverse_mccfr_p(state.child(action), player)
                    explored[action] = True
                    expected_value += strategy[action] * action_values[action]
                else:
                    explored[action] = False
            
            # Update regrets only for explored actions
            for action in legal_actions:
                if explored.get(action, False):
                    regret = action_values[action] - expected_value
                    self.regrets[info_state][action] += regret
                    self.regrets[info_state][action] = max(self.regret_floor,
                                                          self.regrets[info_state][action])
            
            return expected_value
        else:
            # Sample opponent action
            strategy = self.calculate_strategy(info_state, legal_actions)
            actions = list(strategy.keys())
            probs = list(strategy.values())
            action = np.random.choice(actions, p=probs)
            return self.traverse_mccfr_p(state.child(action), player)
    
    def run(self, iterations):
        """Run MCCFR-P for specified iterations"""
        for t in range(iterations):
            if t % 100 == 0:
                print(f"Iteration {t}/{iterations}")
                
            # Update each player
            for player in range(self.num_players):
                # Update strategy periodically
                if t % self.strategy_interval == 0:
                    self.update_strategy(self.game.new_initial_state(), player)
                
                # Choose between pruning and regular MCCFR
                if t > self.prune_threshold:
                    if random.random() < self.prune_probability:
                        self.traverse_mccfr_p(self.game.new_initial_state(), player)
                    else:
                        self.traverse_mccfr(self.game.new_initial_state(), player)
                else:
                    self.traverse_mccfr(self.game.new_initial_state(), player)
            
            # Apply Linear CFR discounting
            if t < self.lcfr_threshold and t % self.discount_interval == 0:
                discount = t / self.discount_interval / (t / self.discount_interval + 1)
                
                # Discount all regrets and average strategies
                for info_state in list(self.regrets.keys()):
                    for action in self.regrets[info_state]:
                        self.regrets[info_state][action] *= discount
                        
                for info_state in list(self.avg_strategy.keys()):
                    for action in self.avg_strategy[info_state]:
                        self.avg_strategy[info_state][action] *= discount
    
    def get_final_strategy(self):
        """Get the final average strategy"""
        final_strategy = {}
        
        for info_state in self.avg_strategy:
            total = sum(self.avg_strategy[info_state].values())
            if total > 0:
                final_strategy[info_state] = {}
                for action, count in self.avg_strategy[info_state].items():
                    final_strategy[info_state][action] = count / total
                    
        return final_strategy


def create_pluribus_game():
    """Create 6-player no-limit Texas Hold'em game like Pluribus"""
    # Pluribus configuration: 6 players, $50/$100 blinds, $10,000 stacks
    game_string = (
        "universal_poker("
        "betting=nolimit,"              # No-limit betting
        "numPlayers=6,"                 # 6 players
        "numRounds=4,"                  # Preflop, flop, turn, river
        "blind=100 50 0 0 0 0,"         # BB=$100, SB=$50, others=$0
        "firstPlayer=2 0 0 0,"          # Player 3 first preflop, Player 1 first postflop
        "numSuits=4,"                   # 4 suits
        "numRanks=13,"                  # 13 ranks (2-A)
        "numHoleCards=2,"               # 2 hole cards per player
        "numBoardCards=0 3 1 1,"        # 0 preflop, 3 flop, 1 turn, 1 river
        "stack=10000,"                  # $10,000 starting stack (100 BB)
        "bettingAbstraction=fcpa"       # Full game, no abstraction
        ")"
    )
    return pyspiel.load_game(game_string)


def create_simple_test_game():
    """Create a simpler game for testing the algorithm"""
    # Option 1: Kuhn poker (very simple, good for testing)
    # return pyspiel.load_game("kuhn_poker")
    
    # Option 2: Leduc poker (more complex than Kuhn, but still tractable)
    # return pyspiel.load_game("leduc_poker")
    
    # Option 3: Simplified 3-player limit hold'em
    game_string = (
        "universal_poker("
        "betting=limit,"
        "numPlayers=3,"
        "numRounds=2,"                  # Just preflop and flop for simplicity
        "blind=2 1 0,"                  # BB=$2, SB=$1
        "firstPlayer=2 0,"              # Button first preflop, SB first postflop
        "numSuits=2,"                   # Reduced suits
        "numRanks=3,"                   # Reduced ranks (just 3 ranks)
        "numHoleCards=1,"               # 1 hole card
        "numBoardCards=0 1,"            # 0 preflop, 1 flop
        "stack=10,"                     # Small stacks
        "maxRaises=2"                   # Limit raises
        ")"
    )
    return pyspiel.load_game(game_string)


def main():
    # Choose which game to use
    use_full_game = False  # Set to True for Pluribus game, False for testing
    
    if use_full_game:
        print("Creating 6-player no-limit Texas Hold'em (Pluribus configuration)...")
        game = create_pluribus_game()
        
        # Pluribus training parameters (scaled down for demonstration)
        solver = MCCFR_P(
            game,
            prune_threshold_iterations=12000,    # Pluribus: ~200 minutes = ~12M iterations
            strategy_interval=10000,             # Pluribus: 10,000 iterations
            discount_interval=600000,            # Pluribus: 10 minutes = ~600k iterations
            lcfr_threshold=24000000,            # Pluribus: 400 minutes = ~24M iterations
            prune_probability=0.95,             # Pluribus: 95%
            regret_floor=-310000000             # Pluribus: -310M
        )
        iterations = 100  # Very small for demo (Pluribus used ~50M iterations)
    else:
        print("Creating simplified test game...")
        game = create_simple_test_game()
        
        # Smaller parameters for testing
        solver = MCCFR_P(
            game,
            prune_threshold_iterations=200,
            strategy_interval=100,
            discount_interval=100,
            lcfr_threshold=400,
            prune_probability=0.95,
            regret_floor=-310000
        )
        iterations = 1000
    
    # Print game info
    print(f"Game: {game}")
    print(f"Number of players: {game.num_players()}")
    print(f"Number of distinct actions: {game.num_distinct_actions()}")
    print(f"Max game length: {game.max_game_length()}")
    
    # Run MCCFR-P
    print(f"\nRunning MCCFR-P for {iterations} iterations...")
    solver.run(iterations)
    
    # Get final strategy
    strategy = solver.get_final_strategy()


if __name__ == "__main__":
    main()