import pyspiel

# Create 6-player no-limit Texas Hold'em with proper stack sizes
poker_game = pyspiel.load_game(
    "universal_poker(betting=nolimit,numPlayers=6,numRounds=4,"
    "blind=100 50 0 0 0 0,firstPlayer=2 2 2 2,numSuits=4,numRanks=13,"
    "numHoleCards=2,numBoardCards=0 3 1 1,"
    "stack=10000 10000 10000 10000 10000 10000)"
)

state = poker_game.new_initial_state()

print("Initial state:")
for p in range(6):
    print(f"Player {p}: {state.information_state_string(p)}")

# Deal some cards
print("\nDealing cards...")
for i in range(12):  # 2 cards per player
    if state.is_chance_node():
        outcomes = state.chance_outcomes()
        action = outcomes[0][0]  # Just take first available
        state.apply_action(action)
        print(f"Dealt card: {action}")

print("\nAfter dealing:")
for p in range(6):
    print(f"Player {p}: {state.information_state_string(p)}")

# Take some betting actions
print("\nTaking actions...")
print(f"Current player: {state.current_player()}")
print(f"Legal actions: {state.legal_actions()}")

# UTG folds
state.apply_action(0)
print("Action: 0 (fold)")

# Next player calls
state.apply_action(1)
print("Action: 1 (call)")

# Next player raises
if 2 in state.legal_actions():
    state.apply_action(2)
    print("Action: 2 (raise)")

print("\nAfter some betting:")
for p in range(6):
    print(f"Player {p}: {state.information_state_string(p)}")

print(f"\nCurrent player: {state.current_player()}")
print(f"Legal actions: {state.legal_actions()}")
print(f"Is terminal: {state.is_terminal()}")