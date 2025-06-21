from cfr import KuhnVanillaCFR
from depth_limited_cfr import DepthLimitedCFR

'''
Tests current depth limited search functions from the DepthLimitedCfr class
'''


def test_depth_detection():
    """Test the should_trigger_search logic with various scenarios."""
    
    kuhn_cfr = KuhnVanillaCFR()
    dls = DepthLimitedCFR(kuhn_cfr, depth_limit=2)
    
    print("\n=== Testing Depth Detection ===")
    
    # Test 1: Initial state (should be False - chance node)
    state = kuhn_cfr.new_game()
    result1 = dls.should_trigger_search(state, current_depth=0)
    print(f"Test 1 - Initial state (chance node): {result1} (should be False)")
    print(f"  - Current player: {state.current_player()}")
    print(f"  - History length: {len(state.history())}")
    
    # Test 2: Early game state (should be False - short history)
    state.apply_action(0)  # Chance action (deal cards)
    result2 = dls.should_trigger_search(state, current_depth=0)
    print(f"Test 2 - After card dealing: {result2} (should be False)")
    print(f"  - Current player: {state.current_player()}")
    print(f"  - History length: {len(state.history())}")
    print(f"  - Legal actions: {state.legal_actions()}")
    
    # Test 3: After several actions (longer history)
    state.apply_action(1)  # Player 0 action
    state.apply_action(0)  # Player 1 action
    result3 = dls.should_trigger_search(state, current_depth=1)
    print(f"Test 3 - After 3 total actions: {result3}")
    print(f"  - Current player: {state.current_player()}")
    print(f"  - History length: {len(state.history())}")
    print(f"  - Legal actions: {state.legal_actions()}")
    
    # Test 4: Depth limit trigger (should be True)
    result4 = dls.should_trigger_search(state, current_depth=3)  # Over depth_limit=2
    print(f"Test 4 - Over depth limit (depth=3): {result4} (should be True)")
    
    # Test 5: Terminal state (should be False)
    # Let's create a terminal state
    terminal_state = kuhn_cfr.new_game()
    terminal_state.apply_action(0)  # Deal cards
    terminal_state.apply_action(1)  # P0 bets
    terminal_state.apply_action(0)  # P1 folds
    result5 = dls.should_trigger_search(terminal_state, current_depth=1)
    print(f"Test 5 - Terminal state: {result5} (should be False)")
    print(f"  - Is terminal: {terminal_state.is_terminal()}")
    
    # Test 6: Simulate many actions (long history)
    long_state = kuhn_cfr.new_game()
    # Simulate 12 actions to test history_length > 9
    action_sequence = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    for i, action in enumerate(action_sequence):
        if not long_state.is_terminal() and action in long_state.legal_actions():
            long_state.apply_action(action)
        else:
            break
    
    result6 = dls.should_trigger_search(long_state, current_depth=1)
    print(f"Test 6 - Long history (len={len(long_state.history())}): {result6}")
    print(f"  - Should be True if history > 9")
    
    print("\n=== Test Summary ===")
    print("✅ All tests completed! Check that results match expected behavior.")

def test_create_subgame():
    """Test the create_subgame functionality with various scenarios."""
    
    kuhn_cfr = KuhnVanillaCFR()
    dls = DepthLimitedCFR(kuhn_cfr, depth_limit=2)
    
    print("=== Testing Subgame Creation ===")
    
    # Test 1: Small subgame from game start
    print("\n--- Test 1: Subgame from initial state ---")
    initial_state = kuhn_cfr.new_game()
    
    # Move past chance node to get to actual decision
    if initial_state.current_player() == -1:
        initial_state.apply_action(0)  # Deal cards
    
    subgame1 = dls.create_subgame(initial_state, depth_limit=2)
    
    print(f"Root state history: {subgame1['root'].history()}")
    print(f"Total nodes in subgame: {len(subgame1['nodes'])}")
    print(f"Leaf nodes: {len(subgame1['leaf_nodes'])}")
    print(f"Root node info: {subgame1['nodes'][subgame1['root_id']]}")
    
    # Test 2: Subgame from mid-game
    print("\n--- Test 2: Subgame from mid-game state ---")
    mid_state = kuhn_cfr.new_game()
    mid_state.apply_action(0)  # Deal cards
    mid_state.apply_action(1)  # P0 bets
    
    subgame2 = dls.create_subgame(mid_state, depth_limit=3)
    
    print(f"Mid-game root history: {subgame2['root'].history()}")
    print(f"Total nodes: {len(subgame2['nodes'])}")
    print(f"Leaf nodes: {len(subgame2['leaf_nodes'])}")
    
    # Test 3: Explore subgame structure
    print("\n--- Test 3: Detailed subgame structure ---")
    print("All nodes in subgame:")
    for node_id, node_info in subgame1['nodes'].items():
        is_leaf = node_id in subgame1['leaf_nodes']
        print(f"  Node {node_id}: depth={node_info['depth']}, "
              f"player={node_info['player']}, "
              f"actions={node_info['legal_actions']}, "
              f"leaf={is_leaf}")
    
    print("\nEdge structure:")
    for state_id, actions in subgame1['edges'].items():
        for action, next_state in actions.items():
            print(f"  {state_id} --[action {action}]--> {next_state}")
    
    # Test 4: Verify leaf nodes
    print("\n--- Test 4: Leaf node analysis ---")
    for leaf_id in subgame1['leaf_nodes']:
        leaf_info = subgame1['nodes'][leaf_id]
        print(f"Leaf {leaf_id}: depth={leaf_info['depth']}, "
              f"terminal={leaf_info['is_terminal']}")
    
    print("\n=== Subgame Creation Tests Complete ===")

def test_blueprint_and_continuation():
    # Step 1: Create and test your implementation
    kuhn_cfr = KuhnVanillaCFR()
    dls = DepthLimitedCFR(kuhn_cfr, depth_limit=2)
    
    # Step 2: Test blueprint generation
    print("=== Testing Blueprint Generation ===")
    dls.generate_blueprint_strategy(1000)  # Train for 1000 iterations
    
    # Check if blueprint was created
    print(f"Number of info sets in blueprint: {len(dls.blueprint_strategy)}")
    print(f"Sample info sets: {list(dls.blueprint_strategy.keys())[:3]}")
    
    # Verify probabilities sum to 1
    for info_set, probs in list(dls.blueprint_strategy.items())[:3]:
        print(f"Info set '{info_set}': {probs} (sum = {sum(probs):.3f})")
    
    # Step 3: Test continuation strategies
    print("\n=== Testing Continuation Strategies ===")
    dls.generate_continuation_strategies(biased_multiplier=5.0)
    
    # Check all 4 strategies were created
    print(f"Strategy types: {list(dls.continuation_strategies.keys())}")
    
    # Compare strategies for same info set
    test_info_set = list(dls.blueprint_strategy.keys())[0]
    print(f"\nComparing strategies for info set '{test_info_set}':")
    for strategy_name, strategy_dict in dls.continuation_strategies.items():
        probs = strategy_dict[test_info_set]
        print(f"{strategy_name:12}: {[f'{p:.3f}' for p in probs]} (sum = {sum(probs):.3f})")



if __name__ == "__main__":
    test_create_subgame()
