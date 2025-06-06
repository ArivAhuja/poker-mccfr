import pyspiel
from open_spiel.python.algorithms import external_sampling_mccfr as es_mccfr
from open_spiel.python.algorithms import outcome_sampling_mccfr as os_mccfr
from open_spiel.python.algorithms import exploitability
import time
import numpy as np

def create_limit_holdem_game(simplified=True):
    """Create a limit hold'em game"""
    if simplified:
        # Simplified version for faster training
        params = {
            "betting": "limit",
            "numPlayers": 2,  # Integer, not string
            "numRounds": 4,
            "blind": "5 10",  # This stays as string
            "firstPlayer": "2 1 1 1",  # This stays as string
            "numSuits": 2,
            "numRanks": 6,
            "numHoleCards": 2,
            "numBoardCards": "0 3 1 1",  # This stays as string
            "raiseSize": "10 10 20 20",  # This stays as string
            "maxRaises": "3 3 3 3"  # This stays as string
        }
        print("Using simplified Hold'em (2 suits, 6 ranks) for faster training")
    else:
        # Full game
        params = {
            "betting": "limit",
            "numPlayers": 2,
            "numRounds": 4,
            "blind": "50 100",
            "firstPlayer": "2 1 1 1",
            "numSuits": 4,
            "numRanks": 13,
            "numHoleCards": 2,
            "numBoardCards": "0 3 1 1",
            "raiseSize": "100 100 200 200",
            "maxRaises": "3 3 3 3"
        }
        print("Using full Limit Hold'em")
    
    return pyspiel.load_game("universal_poker", params)


def train_with_external_sampling_mccfr(game, iterations):
    """Train using External Sampling MCCFR (built-in)"""
    print("\n" + "="*60)
    print("EXTERNAL SAMPLING MCCFR")
    print("="*60)
    
    # Create solver - fix the enum reference
    solver = es_mccfr.ExternalSamplingSolver(
        game,
        es_mccfr.AverageType.SIMPLE  # Fixed: use es_mccfr instead of external_sampling_mccfr
    )
    
    print(f"Training for {iterations} iterations...")
    start_time = time.time()
    
    # Training loop
    for i in range(iterations):
        solver.iteration()
        
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            conv = exploitability.nash_conv(game, solver.average_policy())
            print(f"Iteration {i+1:,}: "
                  f"Time: {elapsed:.1f}s, "
                  f"Nash-Conv: {conv:.6f}")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f} seconds")
    
    return solver


def train_with_outcome_sampling_mccfr(game, iterations):
    """Train using Outcome Sampling MCCFR (built-in)"""
    print("\n" + "="*60)
    print("OUTCOME SAMPLING MCCFR")
    print("="*60)
    
    # Create solver
    solver = os_mccfr.OutcomeSamplingSolver(game)
    
    print(f"Training for {iterations} iterations...")
    start_time = time.time()
    
    # Training loop
    for i in range(iterations):
        solver.iteration()
        
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            conv = exploitability.nash_conv(game, solver.average_policy())
            print(f"Iteration {i+1:,}: "
                  f"Time: {elapsed:.1f}s, "
                  f"Nash-Conv: {conv:.6f}")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f} seconds")
    
    return solver


def analyze_strategy(solver, game, num_samples=20):
    """Analyze the learned strategy"""
    print("\n" + "="*60)
    print("STRATEGY ANALYSIS")
    print("="*60)
    
    # Get the average policy
    average_policy = solver.average_policy()
    
    # Create a policy table for easier analysis
    policy_table = average_policy.to_dict()
    
    # Action names for universal_poker
    action_names = {0: 'Fold', 1: 'Call/Check', 2: 'Bet/Raise'}
    
    # Sample some interesting decision points
    print("\nSample strategies from different streets:")
    print("-" * 50)
    
    # Get a sample of info sets
    all_infosets = list(policy_table.keys())
    np.random.shuffle(all_infosets)
    
    samples_shown = 0
    for info_set in all_infosets[:100]:  # Look at first 100
        if samples_shown >= num_samples:
            break
            
        action_probs = policy_table[info_set]
        
        # Only show non-trivial decisions
        if len(action_probs) > 1 and max(action_probs.values()) < 0.95:
            print(f"\nInfo set: {info_set[:60]}...")
            
            # Sort actions by probability
            sorted_actions = sorted(action_probs.items(), 
                                  key=lambda x: x[1], reverse=True)
            
            for action, prob in sorted_actions:
                action_name = action_names.get(action, f'Action {action}')
                print(f"  {action_name}: {prob:.3f}")
            
            samples_shown += 1
    
    # Calculate exploitability
    print("\n" + "="*60)
    print("CONVERGENCE METRICS")
    print("="*60)
    
    conv = exploitability.nash_conv(game, average_policy)
    print(f"Nash Convergence: {conv:.6f}")
    print(f"Exploitability: {exploitability.exploitability(game, average_policy):.6f}")


def test_specific_hands(solver, game):
    """Test strategy with specific situations"""
    print("\n" + "="*60)
    print("TESTING SPECIFIC SITUATIONS")
    print("="*60)
    
    # Get the policy
    policy = solver.average_policy()
    
    # Create a game and play through it
    state = game.new_initial_state()
    
    print("\nPlaying a sample hand:")
    print("-" * 30)
    
    while not state.is_terminal():
        if state.is_chance_node():
            # Chance node - just sample
            outcomes = state.chance_outcomes()
            # Fix: outcomes is a list of tuples
            if outcomes:
                action = outcomes[0][0]  # Take first action
                state.apply_action(action)
        else:
            # Player decision
            player = state.current_player()
            info_state = state.information_state_string(player)
            legal_actions = state.legal_actions()
            
            # Get strategy from our policy
            action_probs = dict(policy.policy_for_key(info_state))
            
            # Show decision
            print(f"\nPlayer {player}")
            print(f"Info: {info_state[:60]}...")
            print("Strategy:")
            
            action_names = {0: 'Fold', 1: 'Call/Check', 2: 'Bet/Raise'}
            for action in legal_actions:
                prob = action_probs.get(action, 0.0)
                action_name = action_names.get(action, f'Action {action}')
                print(f"  {action_name}: {prob:.3f}")
            
            # Sample action according to strategy
            if action_probs:
                actions = list(action_probs.keys())
                probs = list(action_probs.values())
                action = np.random.choice(actions, p=probs)
            else:
                action = np.random.choice(legal_actions)
            
            state.apply_action(action)
    
    # Show outcome
    returns = state.returns()
    print(f"\nFinal pot distribution: P0: {returns[0]}, P1: {returns[1]}")


def main():
    print("BUILT-IN MCCFR FOR LIMIT HOLD'EM")
    print("="*60)
    
    # Create game
    simplified = True  # Set to False for full game
    game = create_limit_holdem_game(simplified=simplified)
    
    # Show game info
    print(f"\nGame info:")
    print(f"  Max game length: {game.max_game_length()}")
    print(f"  Number of players: {game.num_players()}")
    print(f"  Number of distinct actions: {game.num_distinct_actions()}")
    
    # Choose solver type
    use_external_sampling = True  # False for outcome sampling
    
    # Training parameters
    iterations = 10000  # Increase for better convergence
    
    # Train
    if use_external_sampling:
        solver = train_with_external_sampling_mccfr(game, iterations)
    else:
        solver = train_with_outcome_sampling_mccfr(game, iterations)
    
    # Analyze results
    analyze_strategy(solver, game)
    
    # Test specific hands
    test_specific_hands(solver, game)
    
    return solver, game


def compare_sampling_methods(game, iterations=5000):
    """Compare different MCCFR sampling methods"""
    print("\n" + "="*60)
    print("COMPARING MCCFR VARIANTS")
    print("="*60)
    
    # External Sampling
    print("\n1. External Sampling MCCFR:")
    start = time.time()
    es_solver = es_mccfr.ExternalSamplingSolver(game)
    for _ in range(iterations):
        es_solver.iteration()
    es_time = time.time() - start
    es_conv = exploitability.nash_conv(game, es_solver.average_policy())
    print(f"   Time: {es_time:.2f}s, Nash-Conv: {es_conv:.6f}")
    
    # Outcome Sampling
    print("\n2. Outcome Sampling MCCFR:")
    start = time.time()
    os_solver = os_mccfr.OutcomeSamplingSolver(game)
    for _ in range(iterations):
        os_solver.iteration()
    os_time = time.time() - start
    os_conv = exploitability.nash_conv(game, os_solver.average_policy())
    print(f"   Time: {os_time:.2f}s, Nash-Conv: {os_conv:.6f}")
    
    print("\nRecommendation:")
    if es_conv < os_conv:
        print("External Sampling converged better")
    else:
        print("Outcome Sampling converged better")


if __name__ == "__main__":
    solver, game = main()
    
    # Optionally compare methods
    # compare_sampling_methods(game, iterations=5000)
    
    print("\n" + "="*60)
    print("NOTES:")
    print("="*60)
    print("- External Sampling MCCFR typically converges faster")
    print("- Outcome Sampling MCCFR uses less memory")
    print("- For large games, consider:")
    print("  * Using abstraction (card bucketing)")
    print("  * Running for millions of iterations")
    print("  * Saving and loading policies")
    print("  * Using variance reduction techniques")