import numpy as np
import pyspiel
from cfr import KuhnVanillaCFR
from kuhn_poker_gui import KuhnTextGUI, self_play, watch_self_play

def main():

    # Train our CFR
    print("Training Vanilla CFR")
    cfr_trainer = KuhnVanillaCFR()
    cfr_trainer.train(iterations=10000)

    # Get GUI compatible profile
    gui_profile = cfr_trainer.get_gui_profile()

    # Showing sample games
    print("Sample games")
    watch_self_play(gui_profile, hands=3)

    # Evaluate
    mean_payoff = self_play(gui_profile, hands = 50000)
    print(f"Played 50000 hands, mean payoff: {mean_payoff}")
    print("Nash Equilibrium ~ -0.05556 (-1/18)")


if __name__ == "__main__":
    main()