import numpy as np
import pyspiel
from cfr import KuhnVanillaCFR
from kuhn_poker_gui import KuhnTextGUI, self_play, watch_self_play
from test_dls_func import test_depth_detection, test_blueprint_and_continuation,test_create_subgame

def main():
    print("=== Depth-Limited CFR Demonstration ===")

    # Train our CFR
    print("\n1. Training Vanilla CFR")
    cfr_trainer = KuhnVanillaCFR()
    cfr_trainer.train(iterations=10000)

    # Test depth-limited search components
    print("\n2. Testing Depth-Limited Search Components")
    
    # Test blueprint and continuation strategies
    test_blueprint_and_continuation()
    
    # Test depth detection
    test_depth_detection()
    
    # Test subgame creation
    test_create_subgame()

    # Original CFR evaluation
    print("\n3. Original CFR Performance")
    gui_profile = cfr_trainer.get_gui_profile()
    
    # Show sample games
    print("\nSample games:")
    watch_self_play(gui_profile, hands=3)

    # Evaluate performance
    mean_payoff = self_play(gui_profile, hands=50000)
    print(f"\nPlayed 50000 hands, mean payoff: {mean_payoff}")
    print("Nash Equilibrium ~ -0.05556 (-1/18)")
    
    print("\n=== Depth-Limited CFR Components Complete! ===")
   
if __name__ == "__main__":
    main()
