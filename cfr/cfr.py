import pyspiel  # Keeping this import for game specific implementation
import numpy as np

"""
------------
    TODO
------------

For game specific CFR implementations simply need to adjust abstract classes/methods
utilize open_spiel's useful/available information set functions (strings mostly).

Game Specific Vanilla CFR to be implemented:
    -LeducHoldem <- Doable with vanilla but not ideal - decided not necessary for base cfr
    -TexasLimit <- Will need monte carlo - not feasible with vanilla/cfr+

------------
    DONE
------------

Game Specific Vanilla CFR supported:
    -KuhnVanillaCFR

Game Specific CFR+ supported:
    -KuhnCFRPlus

CFR algorithms supported:
    -VanillaCFR (BaseCFR)
    -CFR+ (CFRPlus)
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
        """Number of actions at a information set."""
        raise NotImplementedError("Implement get_num_actions in subclass")

    def is_terminal(self, state: pyspiel.State) -> bool:
        """Check if game state is terminal."""
        raise NotImplementedError("Implement is_terminal in subclass")

    def get_utility(self, state: pyspiel.State, player: int) -> float:
        """Get the terminal node utility."""
        raise NotImplementedError("Implement get_utility in subclass")

    def get_info_set(self, state: pyspiel.State, player: int) -> str:
        """Convert game state to information set string."""
        raise NotImplementedError("Implement get_info_set in subclass")

    def get_player(self, state: pyspiel.State) -> int:
        """Get the current player to move (or chance node)."""
        raise NotImplementedError("Implement get_player in subclass")

    def get_chance_outcomes(self, state: pyspiel.State) -> list[tuple[int, float]]:
        """Get the list of (action (int index) : probability (float)) for a chance node."""
        raise NotImplementedError("Implement get_chance_outcomes in subclass")

    def get_gui_profile(self):
        """Return the compatible GUI profile for the given implementation."""
        raise NotImplementedError("Implement get_gui_profile in subclass")

    # -----------------------------------------------------------------
    # Abstract methods to manage the game history and general game tree
    # -----------------------------------------------------------------
    def get_next_state(self, state: pyspiel.State, action) -> pyspiel.State:
        """
        Apply a given action to the current game state to generate a new game state.

        :param state: pyspiel.State game state.
        :param action: action

        :returns: new state
        """
        raise NotImplementedError("Implement get_next_history in subclass")

    def new_game(self) -> pyspiel.State:
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

    def cfr(self, state: pyspiel.State, reach_probs: list[float], cfr_player: int) -> float:
        """
        Recursive function for CFR traversal throughout the game tree.

        :param state: the game state
        :param reach_probs: the probability for the player to reach this node
        :param cfr_player: the player whose cfr is being calculated

        :return: the expected reward a player should receive after game termination
        """
        # Needs implementation

        # Check if terminal -> return utility
        if self.is_terminal(state):
            return self.get_utility(state, cfr_player)

        # Get whose turn it is, chance node = -1
        player_turn = self.get_player(state)

        # Check if chance node
        if player_turn == -1:
            expected_utility = 0.0
            for action, prob in self.get_chance_outcomes(state):
                new_state = self.get_next_state(state, action)
                # Reach probabilities stay constant for chance nodes
                expected_utility += prob * self.cfr(new_state, reach_probs, cfr_player)
            return expected_utility

        # Get our information set and strategy
        info_set = self.get_info_set(state, player_turn)
        strategy = self.get_strategy(info_set)

        # Get possible actions and initialize each actions expected utility
        num_actions = self.get_num_actions(info_set)
        action_expected_utility = np.zeros(num_actions)

        for action in range(num_actions):
            new_state = self.get_next_state(state, action)

            # Update new reach probabilities
            new_reach_prob = reach_probs.copy()
            new_reach_prob[player_turn] *= strategy[action]

            # Recursively call
            action_expected_utility[action] = self.cfr(new_state, new_reach_prob, cfr_player)

        # Compute the expected utility of the current node in the game tree
        current_node_expected_utility = np.dot(strategy, action_expected_utility)

        if cfr_player == player_turn:
            # Calculate regret based on action taken
            regrets = action_expected_utility - current_node_expected_utility

            # Calculate opponents reach prob
            opponent_reach = 1.0
            for player in range(self.num_players):
                if player != cfr_player:
                    opponent_reach *= reach_probs[player]

            # Weight our regrets based on reach
            weighted_regrets = opponent_reach * regrets

            # Update cumulative regret
            self.update_regrets(info_set, weighted_regrets)

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

    # --------------------------------------------------------------------
    # CFR information export methods, used for external applications (GUI)
    # --------------------------------------------------------------------
    def get_strategy_profile(self) -> dict[str, list[float]]:
        """
        Get the final strategy profile as a dictionary.

        :return: the final strategy mapping as a dictionary (list of actions probabilities)
        """
        profile = {}
        for info_set in self.strategy_sum:
            strategy = self.get_average_strategy(info_set)
            profile[info_set] = strategy.tolist()
        return profile

class CFRPlus(BaseCFR):
    """CFR+ implementation of Vanilla CFR."""

    def __init__(self, num_players=2):
        super(CFRPlus, self).__init__(num_players)

    def update_regrets(self, info_set, regrets: np.ndarray):
        """CFR+ modification to update_regrets, only record positive regrets.
        Add given regrets to the positive cumulative regret sum.

        :param info_set: The information set of the game
        :param regrets: The given np.darray of regrets
        """
        if info_set not in self.regret_sum:
            self.regret_sum[info_set] = np.zeros(len(regrets))
        # Always update the regret sum, only taking positive regrets
        self.regret_sum[info_set] = np.maximum(self.regret_sum[info_set] + regrets, 0)


class BaseKuhnCFR:
    """
    Base class for all Kuhn Poker CFR implementations.
    """
    def __init__(self):
        self.game = pyspiel.load_game("kuhn_poker")

    def get_num_actions(self, info_set: str) -> int:
        """Number of actions at an information set."""
        # Only two actions in Kuhn poker BET/PASS
        return 2

    def is_terminal(self, state: pyspiel.State) -> bool:
        """Check if game state is terminal."""
        return state.is_terminal()

    def get_utility(self, state: pyspiel.State, player: int) -> float:
        """Get the terminal node utility."""
        return state.player_return(player)

    def get_info_set(self, state: pyspiel.State, player: int) -> str:
        """Convert history to information set string."""
        return state.information_state_string(player)

    def get_player(self, state: pyspiel.State) -> int:
        """Get the current player to move (or chance node)."""
        return state.current_player()

    def get_chance_outcomes(self, state: pyspiel.State) -> list[tuple[int, float]]:
        """Get the list of (action (int index) : probability (float)) for a chance node."""
        return state.chance_outcomes()

    def get_next_state(self, state: pyspiel.State, action) -> pyspiel.State:
        """Get the next state based on the given action."""
        next_state = state.clone()
        next_state.apply_action(action)
        return next_state

    def new_game(self) -> pyspiel.State:
        """Create a new initial game state."""
        return self.game.new_initial_state()

    def get_gui_profile(self) -> dict[str, list[float]]:
        return self._kuhn_gui_profile()

    def _kuhn_gui_profile(self) -> dict[str, list[float]]:
        """
        Convert the open spiel Kuhn Poker information sets to GUI compatible formats.

        Ex.
        Open Spiel -> '0pb' VS KuhnGui -> 'Jpb'

        :return: a dictionary mapping information set strings to GUI compatible formats
        """
        profile = self.get_strategy_profile()
        kuhn_gui_profile = {}

        # Convert to KuhnGUI
        for info_set, strategy in profile.items():
            # Extract open spiel information
            raw_card = info_set[0]
            raw_actions = info_set[1:]

            # Player card mappings: J, Q, K
            card = {"0": "J", "1": "Q", "2": "K"}

            # Get the players card
            player_card = card[raw_card]

            # Convert p-> x since terminal states are not included in info sets
            gui_history = raw_actions.replace('p', 'x')

            gui_key = player_card + gui_history
            kuhn_gui_profile[gui_key] = strategy

        return kuhn_gui_profile

class KuhnVanillaCFR(BaseKuhnCFR, BaseCFR): 
    """Kuhn Poker with Vanilla CFR"""

    def __init__(self):
        BaseCFR.__init__(self, num_players=2)
        BaseKuhnCFR.__init__(self)

class KuhnCFRPlus(BaseKuhnCFR, CFRPlus):
    """Kuhn Poker with CFR+"""

    def __init__(self):
        CFRPlus.__init__(self, num_players=2)
        BaseKuhnCFR.__init__(self)