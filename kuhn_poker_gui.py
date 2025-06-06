# kuhn_poker_gui.py
# -------------------------------------------------------------
# Minimal, self‑contained implementation of:
#   • Kuhn Poker engine (2‑player, zero‑sum)
#   • Vanilla Counterfactual Regret Minimisation trainer
#   • Text GUI for human‑vs‑bot play
#   • Bot‑vs‑bot self‑play evaluator
# -------------------------------------------------------------
# Python ≥3.8, no external dependencies

import random
import os
import argparse
from typing import Dict, List, Tuple

# -------------------------------------------------------------
# Helper: clear console for fresh frame
# -------------------------------------------------------------

def clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")

# -------------------------------------------------------------
# Kuhn Poker rule utilities
# -------------------------------------------------------------
CARDS = ["J", "Q", "K"]
CARD_RANK = {"J": 0, "Q": 1, "K": 2}


def legal_actions(history: str) -> List[str]:
    """Return legal action strings given the betting history."""
    if history == "":
        return ["check", "bet"]
    if history == "c":
        return ["check", "bet"]
    if history == "b":
        return ["fold", "call"]
    if history == "cb":
        return ["fold", "call"]
    return []  # terminal


def is_terminal(history: str) -> bool:
    """True if history corresponds to an ended hand."""
    return history in {"cc", "bc", "bf", "cbf", "cbc"}


def pot_size(history: str) -> int:
    """Total chips in the pot at terminal (antes already in)."""
    if history == "cc":
        return 2
    if history in {"bf", "cbf"}:  # bet‑fold
        return 3
    if history in {"bc", "cbc"}:  # bet‑call
        return 4
    return 2  # during play


def terminal_utility(history: str, cards: Tuple[str, str]) -> int:
    """Return payoff for **player 0** at terminal state (zero‑sum)."""
    p0_card, p1_card = cards

    # Someone folded — bettor wins 1 chip from opponent
    if history in {"bf", "cbf"}:
        winner = 0 if history[0] == "b" else 1
        return 1 if winner == 0 else -1

    # Showdown (check‑check or bet‑call)
    winner = 0 if CARD_RANK[p0_card] > CARD_RANK[p1_card] else 1
    payoff = 1 if history == "cc" else 2  # win amount depends on pot 2 or 4
    return payoff if winner == 0 else -payoff

# -------------------------------------------------------------
# CFR data structure (information set)
# -------------------------------------------------------------
class Node:
    def __init__(self, num_actions: int):
        self.regret_sum = [0.0] * num_actions
        self.strategy_sum = [0.0] * num_actions

    def get_strategy(self, reach_weight: float) -> List[float]:
        strategy = [max(r, 0.0) for r in self.regret_sum]
        norm = sum(strategy)
        if norm == 0:
            strategy = [1.0 / len(strategy)] * len(strategy)
        else:
            strategy = [s / norm for s in strategy]
        # accumulate for average strategy
        self.strategy_sum = [s_sum + reach_weight * s for s_sum, s in zip(self.strategy_sum, strategy)]
        return strategy

    def average_strategy(self) -> List[float]:
        norm = sum(self.strategy_sum)
        if norm == 0:
            return [1.0 / len(self.strategy_sum)] * len(self.strategy_sum)
        return [s / norm for s in self.strategy_sum]

# -------------------------------------------------------------
# Vanilla CFR trainer for Kuhn Poker
# -------------------------------------------------------------
class KuhnCFRTrainer:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}

    def train(self, iterations: int = 200_000) -> float:
        """Run CFR for *iterations* deals; return average game value."""
        util_sum = 0.0
        for _ in range(iterations):
            deck = CARDS[:]
            random.shuffle(deck)
            util_sum += self._cfr(deck, "", 1.0, 1.0)
        return util_sum / iterations

    def _cfr(self, cards: List[str], history: str, p0: float, p1: float) -> float:
        if is_terminal(history):
            return terminal_utility(history, (cards[0], cards[1]))

        player = len(history) % 2  # 0 or 1 turn
        info_key = cards[player] + history
        actions = legal_actions(history)
        node = self.nodes.setdefault(info_key, Node(len(actions)))

        strategy = node.get_strategy(p0 if player == 0 else p1)
        util = []
        node_util = 0.0

        for a_prob, action in zip(strategy, actions):
            next_history = history + ("c" if action in {"check", "call"} else "b" if action == "bet" else "f")
            if player == 0:
                val = -self._cfr(cards, next_history, p0 * a_prob, p1)
            else:
                val = -self._cfr(cards, next_history, p0, p1 * a_prob)
            util.append(val)
            node_util += a_prob * val

        # regret update
        for i in range(len(actions)):
            regret = util[i] - node_util
            if player == 0:
                node.regret_sum[i] += p1 * regret
            else:
                node.regret_sum[i] += p0 * regret
        return node_util

    def profile(self) -> Dict[str, List[float]]:
        return {k: n.average_strategy() for k, n in self.nodes.items()}

# -------------------------------------------------------------
# Text GUI (human vs CFR bot)
# -------------------------------------------------------------
class KuhnTextGUI:
    def __init__(self, profile: Dict[str, List[float]]):
        self.profile = profile
        self.char_map = {"check": "c", "call": "c", "bet": "b", "fold": "f"}

    def _bot_move(self, card: str, history: str) -> str:
        actions = legal_actions(history)
        probs = self.profile.get(card + history, [1.0 / len(actions)] * len(actions))
        return random.choices(actions, probs)[0]

    def play(self, human_first: bool = True):
        while True:
            deck = CARDS[:]
            random.shuffle(deck)
            human_card, bot_card = (deck[0], deck[1]) if human_first else (deck[1], deck[0])
            history = ""
            while not is_terminal(history):
                clear()
                print("===== KUHN POKER (CFR bot) =====")
                print(f"Hist: {history or '-'}    Pot: {pot_size(history)}")
                print(f"Your card: {human_card}\n")
                player = len(history) % 2
                human_turn = (player == 0 and human_first) or (player == 1 and not human_first)
                if human_turn:
                    allowed = legal_actions(history)
                    move = ""
                    while move not in allowed:
                        move = input(f"Your move {allowed}: ").strip().lower()
                    history += self.char_map[move]
                else:
                    move = self._bot_move(bot_card, history)
                    print(f"Bot: {move}")
                    input("(Enter)")
                    history += self.char_map[move]
            # result
            clear()
            payoff = terminal_utility(history, (human_card, bot_card))
            print("===== RESULT =====")
            print(f"History: {history}")
            print(f"Your card {human_card} | Bot card {bot_card}")
            print("You win" if payoff > 0 else "Bot wins" if payoff < 0 else "Tie", abs(payoff))
            if input("Again? (y/n): ").lower() != "y":
                break

# -------------------------------------------------------------
# Self‑play evaluator
# -------------------------------------------------------------

def self_play(profile: Dict[str, List[float]], hands: int = 50_000) -> float:
    payoff_sum = 0
    char_map = {"check": "c", "call": "c", "bet": "b", "fold": "f"}
    for _ in range(hands):
        deck = CARDS[:]
        random.shuffle(deck)
        history = ""
        while not is_terminal(history):
            player = len(history) % 2
            card = deck[player]
            actions = legal_actions(history)
            probs = profile.get(card + history, [1.0 / len(actions)] * len(actions))
            move = random.choices(actions, probs)[0]
            history += char_map[move]
        payoff_sum += terminal_utility(history, (deck[0], deck[1]))
    return payoff_sum / hands

def watch_self_play(profile, hands=1):
    char_map = {"check": "c", "call": "c", "bet": "b", "fold": "f"}
    for h in range(hands):
        deck = CARDS[:]
        random.shuffle(deck)
        history = ""
        while not is_terminal(history):
            player = len(history) % 2
            card = deck[player]
            acts = legal_actions(history)
            probs = profile.get(card + history, [1/len(acts)]*len(acts))
            move = random.choices(acts, probs)[0]
            history += char_map[move]
            print(f"P{player} ({card}) -> {move}   hist:{history}")
        print("Payoff P0:", terminal_utility(history, (deck[0], deck[1])))
        print("-"*40)


# -------------------------------------------------------------
# CLI entry
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kuhn Poker CFR demo")
    parser.add_argument("--iters", type=int, default=200_000, help="training iterations")
    parser.add_argument("--selfplay", action="store_true", help="run bot‑vs‑bot simulation only")
    args = parser.parse_args()

    print(f"Training CFR for {args.iters:,} iterations…")
    trainer = KuhnCFRTrainer()
    ev = trainer.train(args.iters)
    profile = trainer.profile()
    watch_self_play(profile, hands=5)   
    print(f"Done. Average game value for player 0 (the trainer's perspective): {ev:+.5f}\n")

    if args.selfplay:
        mean = self_play(profile, 50_000)
        print(f"50 k self‑play hands ⇒ mean payoff: {mean:+.5f} (equilibrium ≈ -0.05556)")
    else:
        gui = KuhnTextGUI(profile)
        gui.play()


# in command line:

# AI vs AI: python kuhn_poker_gui.py --iters 100000 --selfplay
# Player vs AI: python kuhn_poker_gui.py --iters 100000 
 
# You can choose how many iterations you want.
