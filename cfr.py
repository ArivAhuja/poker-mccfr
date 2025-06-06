import pyspiel  # Keeping this import for game specific implementation
import numpy as np

"""
------------
    TODO
------------

For game specific CFR implementations simply need to adjust abstract classes/methods
utilize open_spiel's useful/available information set functions (strings mostly).

Game Specific Vanilla CFR to be implemented:
    -KuhnVanillaCFR
    -LeducHoldemVanillaCFR
    -TexasLimitVanillaCFR

Need to adjust main CFR classes, then reimplement for specific games

CFR algorithms to be implemented:
    -CFR+ (adjust update_regrets)
    -Monte Carlo CFR (adjust cfr, sampling based)

------------
    DONE
------------

Game Specific Vanilla CFR supported:
    -

CFR algorithms supported:
    -VanillaCFR (BaseCFR)
"""


class BaseCFR:
    """
    The abstract base class for CFR implementations.
    Is a Vanilla CFR implementation.

    Arguments:
        strategy_sum (dict): dict mapping info set -> action -> cumulative strategy
        regret_sum (dict): np array mapping info set -> action -> cumulative regret
        num_players (int): number of agents in the game
    """

    def __init__(self, num_players=2):
        self.strategy_sum = {}
        self.regret_sum = {}
        self.num_players = num_players

    # ------------------------------------------------
    # Abstract methods to be implemented in subclasses
    # ------------------------------------------------
    def get_num_actions(self, info_set: str) -> int:
        """Number of actions at an information set."""
        raise NotImplementedError("Implement get_num_actions in subclass")

    def is_terminal(self, history: str) -> bool:
        """Check if game state is terminal."""
        raise NotImplementedError("Implement is_terminal in subclass")

    def get_utility(self, history: str, player: int) -> float:
        """Get the terminal node utility."""
        raise NotImplementedError("Implement get_utility in subclass")

    def get_info_set(self, history: str, player: int) -> str:
        """Convert history to information set string."""
        raise NotImplementedError("Implement get_info_set in subclass")

    def get_player(self, history: str) -> int:
        """Get the current player to move (or chance node)."""
        raise NotImplementedError("Implement get_player in subclass")

    def get_chance_outcomes(self, history: str) -> list[tuple[int, float]]:
        """Get the list of (action (int index) : probability (float)) for a chance node."""
        raise NotImplementedError("Implement get_chance_outcomes in subclass")

    # -----------------------------------------------------------------
    # Abstract methods to manage the game history and general game tree
    # -----------------------------------------------------------------
    def get_next_history(self, history: str, action) -> str:
        """
        Apply a given action to the current history to generate a new history.

        :param history: history string
        :param action: action

        :returns: new history string
        """
        raise NotImplementedError("Implement get_next_history in subclass")

    def new_game(self) -> str:
        """
        Create a new initial game state.
        """
        raise NotImplementedError("Implement new_game in subclass")

    # -------------------------------------------------------
    # Regret matching algorithms, used in all implementations
    # -------------------------------------------------------

    # Might be beneficial to move to separate class.

    def regret_matching(self, regrets: np.ndarray) -> np.ndarray:
        """
        Convert the given regrets to probabilities via implementing positive regret matching.

        :param regrets: The given np.darray of regrets

        :return: The converted np.darray of regrets to probabilities.
        """
        # Calculate positive regret
        positive_regret = np.maximum(regrets, 0)

        if positive_regret.sum() == 0:
            return np.ones(len(positive_regret)) / len(positive_regret)  # Uniform probability
        else:
            return positive_regret / positive_regret.sum()

    def update_regrets(self, info_set, regrets: np.ndarray):
        """
        Add given regrets to the cumulative regret sum.

        :param info_set: The information set of the game
        :param regrets: The given np.darray of regrets
        """
        if info_set not in self.regret_sum:
            self.regret_sum[info_set] = np.zeros(len(regrets))
        # Always update the regret sum
        self.regret_sum[info_set] += regrets

    def update_strategy_sum(self, info_set, strategy: np.ndarray, reach_prob):
        """
        Track the average strategy, based on the current information set.

        :param info_set: The information set of the game
        :param strategy: The given np.darray of strategies
        :param reach_prob: The probability for the player to reach this node
        """
        if info_set not in self.strategy_sum:
            self.strategy_sum[info_set] = np.zeros(len(strategy))
        # Always update the strategy sum
        self.strategy_sum[info_set] += reach_prob * strategy

    # ----------------------------------------------------
    # Core CFR methods that act as the base implementation
    # ----------------------------------------------------
    def train(self, iterations=1):
        """
        The main training loop for CFR.

        :param iterations: the number of iterations
        """
        for i in range(iterations):
            for player in range(self.num_players):
                initial_reach = [1.0] * self.num_players
                self.cfr(self.new_game(), initial_reach, player)

    def cfr(self, history: str, reach_probs: list[float], cfr_player: int) -> float:
        """
        Recursive function for CFR traversal throughout the game tree.

        :param history: the history string
        :param reach_probs: the probability for the player to reach this node
        :param cfr_player: the player whose cfr is being calculated

        :return: the expected reward a player should receive after game termination
        """
        # Needs implementation

        # Check if terminal -> return utility
        if self.is_terminal(history):
            return self.get_utility(history, cfr_player)

        # Get whose turn it is, chance node = -1
        player_turn = self.get_player(history)

        # Check if chance node
        if player_turn == -1:
            expected_utility = 0.0
            for action, prob in self.get_chance_outcomes(history):
                new_history = self.get_next_history(history, action)
                # Reach probabilities stay constant for chance nodes
                expected_utility += prob * self.cfr(new_history, reach_probs, cfr_player)
            return expected_utility

        # Get our information set and strategy
        info_set = self.get_info_set(history, player_turn)
        strategy = self.get_strategy(info_set)

        # Get possible actions and initialize each actions expected utility
        num_actions = self.get_num_actions(info_set)
        action_expected_utility = np.zeros(num_actions)

        for action in range(num_actions):
            new_history = self.get_next_history(history, action)

            # Update new reach probabilities
            new_reach_prob = reach_probs.copy()
            new_reach_prob[player_turn] *= strategy[action]

            # Recursively call
            action_expected_utility[action] = self.cfr(new_history, new_reach_prob, cfr_player)

        # Compute the expected utility of the current node in the game tree
        current_node_expected_utility = np.dot(strategy, action_expected_utility)

        if cfr_player == player_turn:
            # Calculate regret based on action taken
            regrets = action_expected_utility - current_node_expected_utility
            # Update cumulative regret
            self.update_regrets(info_set, regrets)

            # Update strategy based on regrets
            self.update_strategy_sum(info_set, strategy, reach_probs[cfr_player])

        return current_node_expected_utility

    def get_strategy(self, info_set: str) -> np.ndarray:
        """
        Get the current strategy being used for a given info_set.

        If no strategy is found, the strategy is to randomly select an action.

        :param info_set: the information set of the game
        :return: the numpy array of current strategies being used
        """
        if info_set not in self.regret_sum:
            return np.ones(self.get_num_actions(info_set)) / self.get_num_actions(info_set)

        return self.regret_matching(self.regret_sum[info_set])

    def get_average_strategy(self, info_set: str) -> np.ndarray:
        """
        Get the average strategy being used for a given info_set.

        If no strategy is found, the strategy is to randomly select an action.

        :param info_set: the information set of the game
        :return: the numpy array of average strategy being used
        """
        if info_set not in self.strategy_sum:
            return np.ones(self.get_num_actions(info_set)) / self.get_num_actions(info_set)

        strategy = self.strategy_sum[info_set]
        if strategy.sum() <= 0:
            return np.ones(len(strategy)) / len(strategy)
        else:
            return strategy / strategy.sum()