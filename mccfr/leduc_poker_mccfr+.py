import pyspiel
import numpy as np
from collections import defaultdict

class CFRPlus:
    def __init__(self, game):
        self.game = game
        self.regret_sums = defaultdict(lambda: defaultdict(float))  # R+
        self.strategy_sums = defaultdict(lambda: defaultdict(float))
        self.current_strategy = defaultdict(lambda: defaultdict(float))
        self.iteration = 0
        self.exploitability_vals = []
        self.checkpoints = []

    def get_info_set_key(self, state):
        return state.information_state_string(state.current_player())

    def regret_matching_plus(self, info_set, legal_actions):
        regrets = self.regret_sums[info_set]
        positive_regrets = {a: max(0.0, regrets[a]) for a in legal_actions}
        total = sum(positive_regrets.values())

        if total > 0:
            strategy = {a: positive_regrets[a] / total for a in legal_actions}
        else:
            strategy = {a: 1.0 / len(legal_actions) for a in legal_actions}

        # Ensure probabilities sum to 1 (floating-point safety)
        prob_sum = sum(strategy.values())
        if prob_sum > 0:
            strategy = {a: p / prob_sum for a, p in strategy.items()}
        else:
            strategy = {a: 1.0 / len(legal_actions) for a in legal_actions}

        self.current_strategy[info_set] = strategy
        return strategy

    def cfr(self, state, reach_probs, player):
        if state.is_terminal():
            return state.returns()[player]

        if state.is_chance_node():
            return sum(
                prob * self.cfr(state.child(action), reach_probs, player)
                for action, prob in state.chance_outcomes()
            )

        current_player = state.current_player()
        info_set = self.get_info_set_key(state)
        legal_actions = state.legal_actions()
        strategy = self.regret_matching_plus(info_set, legal_actions)

        values = {}
        node_value = 0.0
        for a in legal_actions:
            action_prob = strategy[a]
            new_reach = reach_probs.copy()
            new_reach[current_player] *= action_prob
            v = self.cfr(state.child(a), new_reach, player)
            values[a] = v
            node_value += action_prob * v

        if current_player == player:
            for a in legal_actions:
                regret = values[a] - node_value
                self.regret_sums[info_set][a] += regret * reach_probs[1 - player]

        # Linear averaging (weight by iteration)
        for a in legal_actions:
            self.strategy_sums[info_set][a] += reach_probs[current_player] * strategy[a]

        return node_value

    def get_average_strategy(self):
        avg_strategy = {}
        for info_set in self.strategy_sums:
            total = sum(self.strategy_sums[info_set].values())
            if total > 0:
                avg_strategy[info_set] = {
                    a: self.strategy_sums[info_set][a] / total
                    for a in self.strategy_sums[info_set]
                }
            else:
                n = len(self.strategy_sums[info_set])
                avg_strategy[info_set] = {a: 1.0 / n for a in self.strategy_sums[info_set]}
        return avg_strategy

    def best_response_value(self, state, policy, br_player):
        if state.is_terminal():
            return state.returns()[br_player]
        if state.is_chance_node():
            return sum(
                prob * self.best_response_value(state.child(action), policy, br_player)
                for action, prob in state.chance_outcomes()
            )

        current_player = state.current_player()
        legal = state.legal_actions()
        if current_player == br_player:
            return max(
                self.best_response_value(state.child(a), policy, br_player)
                for a in legal
            )
        else:
            info_set = state.information_state_string(current_player)
            strat = policy.get(info_set)
            if strat is None or any(a not in strat for a in legal):
                strat = {a: 1.0 / len(legal) for a in legal}
            return sum(
                strat.get(a, 0.0) * self.best_response_value(state.child(a), policy, br_player)
                for a in legal
            )

    def train(self, iterations=10000, log_every=1000):
        for i in range(1, iterations + 1):
            self.iteration = i
            for player in [0, 1]:
                self.cfr(self.game.new_initial_state(), [1.0, 1.0], player)

            if i % log_every == 0:
                avg_policy = self.get_average_strategy()
                br0 = self.best_response_value(self.game.new_initial_state(), avg_policy, 0)
                br1 = self.best_response_value(self.game.new_initial_state(), avg_policy, 1)
                exploit = br0 + br1
                self.exploitability_vals.append(exploit)
                self.checkpoints.append(i)
                print(f"Iteration {i}: Exploitability = {exploit:.4f}")

        self.avg_strategy = self.get_average_strategy()

# Run CFR+
game = pyspiel.load_game("leduc_poker")
cfr_plus_solver = CFRPlus(game)
cfr_plus_solver.train(iterations=100000, log_every=100)

import matplotlib.pyplot as plt
plt.plot(cfr_plus_solver.checkpoints, cfr_plus_solver.exploitability_vals)
plt.xlabel("Iterations")
plt.ylabel("Exploitability")
plt.title("CFR+ Exploitability Over Time")
plt.grid(True)
plt.show()
