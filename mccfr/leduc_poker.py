import pyspiel
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


# # Load Leduc Poker
# game = pyspiel.load_game("leduc_poker")
# state = game.new_initial_state()

# print("Initial state:", state)
# print("Is chance node?", state.is_chance_node())
# print("Current player:", state.current_player())


# while not state.is_terminal():
#   legal_actions = state.legal_actions()
#   print("Legal actions", legal_actions)
# #   Deal cards
#   if state.is_chance_node():
#     outcomes_with_probs = state.chance_outcomes()
#     print("Outcome with probs:", outcomes_with_probs)
#     action_list, prob_list = zip(*outcomes_with_probs)
#     action = np.random.choice(action_list, p=prob_list)
#     print("action:", action)
#     state.apply_action(action)
#     print("State:", str(state))
#   else:
#     action = legal_actions[0]
#     print("action", action)
#     state.apply_action(action)
#     print("State:", str(state))


# print("\nFinal state:", state)
# print("Returns:", state.returns())


class MCCFR:
    def __init__(self, game):
        self.game = game
        self.regrets = defaultdict(lambda: defaultdict(float))
        self.strategy = defaultdict(lambda: defaultdict(float))
        self.cumulative_strategy = defaultdict(lambda: defaultdict(float))
        self.avg_strategy = defaultdict(lambda: defaultdict(float))

    def get_info_set_key(self, state):
        if state.is_chance_node():
            return None
        return state.information_state_string(state.current_player())

    def regret_matching(self, info_set, legal_actions):
        regrets = self.regrets[info_set]
        total_pos = sum(max(0, regrets[a]) for a in legal_actions)
        strat = {}
        if total_pos > 0:
            for a in legal_actions:
                strat[a] = max(0, regrets[a]) / total_pos
        else:
            for a in legal_actions:
                strat[a] = 1.0 / len(legal_actions)
        self.strategy[info_set] = strat
        return strat

    def mccfr(self, state, pi, player, depth=0):
        if state.is_terminal():
            return state.returns()[player]

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = np.random.choice(actions, p=probs)
            return self.mccfr(state.child(action), pi, player)

        current_player = state.current_player()
        info_set = self.get_info_set_key(state)
        legal_actions = state.legal_actions()
        strat = self.regret_matching(info_set, legal_actions)

        for a in legal_actions:
            self.cumulative_strategy[info_set][a] += pi * strat[a]

        action_probs = np.array([strat[a] for a in legal_actions])
        action_probs /= np.sum(action_probs)
        action = np.random.choice(legal_actions, p=action_probs)
        child = state.child(action)
        # print("  " * depth + f"-> Player {current_player}, InfoSet: {info_set}, Action: {action}")

        util = self.mccfr(child, pi * strat[action], player, depth=depth+1)


        if current_player == player:
            # Counterfactual regret update (sampled)
            for a in legal_actions:
                alt_state = state.child(a)
                alt_util = self.mccfr(alt_state, pi * strat[a], player)
                regret = alt_util - util
                self.regrets[info_set][a] += regret

        return util

    def compute_avg_strategy(self):
        for info_set in self.cumulative_strategy:
            total = sum(self.cumulative_strategy[info_set].values())
            for a in self.cumulative_strategy[info_set]:
                if total > 0:
                    self.avg_strategy[info_set][a] = self.cumulative_strategy[info_set][a] / total
                else:
                    self.avg_strategy[info_set][a] = 1.0 / len(self.cumulative_strategy[info_set])

    def train(self, iterations=1000, log_every=100):
        exploitability_vals = []
        checkpoints = []

        for i in range(1, iterations + 1):
            for player in [0, 1]:
                state = self.game.new_initial_state()
                self.mccfr(state, 1.0, player)

            if i % log_every == 0:
                self.compute_avg_strategy()

                br0 = best_response_value(self.game.new_initial_state(), self.avg_strategy, 0)
                br1 = best_response_value(self.game.new_initial_state(), self.avg_strategy, 1)
                exploit = br0 + br1

                exploitability_vals.append(exploit)
                checkpoints.append(i)
                print(f"Iteration {i}: Exploitability = {exploit:.4f}")

        # Store for plotting
        self.exploitability_vals = exploitability_vals
        self.checkpoints = checkpoints

def best_response_value(state, policy_dict, br_player):
    if state.is_terminal():
        return state.returns()[br_player]

    if state.is_chance_node():
        value = 0.0
        for action, prob in state.chance_outcomes():
            next_state = state.child(action)
            value += prob * best_response_value(next_state, policy_dict, br_player)
        return value

    current_player = state.current_player()
    legal = state.legal_actions()

    if current_player == br_player:
        return max(best_response_value(state.child(a), policy_dict, br_player) for a in legal)
    else:
        info_state = state.information_state_string(current_player)
        strat = policy_dict.get(info_state, {})
        if not strat:
            strat = {a: 1 / len(legal) for a in legal}
        return sum(
            strat.get(a, 0.0) * best_response_value(state.child(a), policy_dict, br_player)
            for a in legal)

# Run on Leduc
game = pyspiel.load_game("leduc_poker")
mccfr = MCCFR(game)
mccfr.train(iterations=10000)

# Display learned strategy
for info_set, strat in mccfr.avg_strategy.items():
    total = sum(strat.values())
    norm_strat = {a: round(v / total, 2) if total > 0 else 1.0/len(strat) for a, v in strat.items()}
    print(f"{info_set}: {norm_strat}")



plt.plot(mccfr.checkpoints, mccfr.exploitability_vals)
plt.xlabel("Iterations")
plt.ylabel("Exploitability")
plt.title("MCCFR Exploitability Over Time")
plt.grid(True)
plt.show()
