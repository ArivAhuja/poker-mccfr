import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from cfr.cfr import BaseCFR, KuhnVanillaCFR, KuhnCFRPlus

def compute_game_value(instance: BaseCFR) -> float:
    """
    Compute the game value for the given instance, showing the expected value
    a player should receive by playing the current strategy.
    Get the average over multiple iterations (default 100).

    For Kuhn Poker the best game value for player 0 is the nash equilibrium (aka -1/18).
    """
    state = instance.new_game() # Make new game
    reach_probs = [1.0] * instance.num_players # initialize reach probs
    return instance.cfr(state, reach_probs, 0) # Get game value for player 0


def run_convergence_analysis(iterations: int = 1000):
    """Run the convergence analysis for both CFR implementations"""

    # Initialize both algorithms (Vanilla and CFR+)
    vanilla = KuhnVanillaCFR()
    plus = KuhnCFRPlus()

    # Store necessary values
    vanilla_values = []
    plus_values = []

    # Nash equilibrium value for Kuhn poker (Kuhn 1951)
    nash = -1/18

    # Track values for specific iterations (can adjust recording frequency as necessary)
    track_iterations = list(range(0, iterations + 1, 10)) # Doing jumps of 10 for efficiency

    # Open spiel to our implementation card mappings
    card_mappings = {'0': 'J', '1': 'Q', '2': 'K'}

    # Train over range of iterations, checking after each iteration
    for i in range(iterations + 1):
        # train for 1 iteration after initial run
        if i > 0:
            vanilla.train(1)
            plus.train(1)

        # Check metrics
        if i in track_iterations:
            # Get game values
            vanilla_value = compute_game_value(vanilla)
            plus_value = compute_game_value(plus)

            # Add game value to overall list
            vanilla_values.append(vanilla_value)
            plus_values.append(plus_value)

    return track_iterations, vanilla_values, plus_values, nash

def plot_nash_convergence(iterations, vanilla_values, plus_values, nash_value):
    """Compare the convergence to the nash equilibrium of each CFR implementation via plotting."""

    plt.figure(figsize=(10, 6))

    plt.plot(iterations, vanilla_values, 'b-', label='Vanilla CFR', linewidth=2.5, alpha=0.8)
    plt.plot(iterations, plus_values, 'r-', label='CFR+', linewidth=2.5, alpha=0.8)
    plt.axhline(y=nash_value, color='g', linestyle='--',
                label=f'Nash Equilibrium ({nash_value:.4f})', linewidth=2)

    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Game Value (Player 1)', fontsize=14)
    plt.title('CFR Convergence Speed Comparison in Kuhn Poker', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('kuhn_convergence_speed.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_nash_distance(iterations, vanilla_values, plus_values, nash_value):
    """Plot the distance from nash equilibrium of each CFR implementation."""
    # Calculate distances
    vanilla_distances = [abs(val - nash_value) for val in vanilla_values]
    plus_distances = [abs(val - nash_value) for val in plus_values]

    plt.figure(figsize=(10, 6))

    plt.plot(iterations, vanilla_distances, 'b-', label='Vanilla CFR', linewidth=2.5, alpha=0.8)
    plt.plot(iterations, plus_distances, 'r-', label='CFR+', linewidth=2.5, alpha=0.8)

    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Distance from Nash Equilibrium', fontsize=14)
    plt.title('Distance from Nash Equilibrium Over Time', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig('kuhn_distance_from_nash.png', dpi=300, bbox_inches='tight')
    plt.show()

    return vanilla_distances, plus_distances

def plot_exploitability(iterations, vanilla_values, plus_values, nash_value):
    """Plot the exploitability of each CFR implementation.

    Can be approximated to be
    exploitability ~ |current_game_value - nash_value|
    """
    vanilla_exploitability = [abs(val - nash_value) for val in vanilla_values]


def find_convergence_iteration(iterations, values, nash_value, threshold):
    """Find the iteration when a given algorithm gets within a certain threshold of the Nash equilibrium."""
    for i, (iteration, value) in enumerate(zip(iterations, values)):
        if abs(value - nash_value) < threshold:
            return iteration
    return None # Didn't reach threshold
# Main execution
if __name__ == "__main__":
    # Run analysis
    iterations, vanilla_values, plus_values, nash_value = run_convergence_analysis(1000)

    # Plot convergence results
    plot_nash_convergence(iterations, vanilla_values, plus_values, nash_value)
    print(f"\nFinal convergence values:")
    print(
        f"Vanilla CFR: {vanilla_values[-1]:.6f}")
    print(f"CFR+: {plus_values[-1]:.6f}")
    print(f"Nash Equilibrium: {nash_value:.6f}")

    # Plot distance from Nash
    vanilla_distances, plus_distances = plot_nash_distance(iterations, vanilla_values, plus_values, nash_value)
    print(f"\nFinal distance from equilibrium:")
    print(
        f"Vanilla CFR: {vanilla_distances[-1]:.6f}")
    print(f"CFR+: {plus_distances[-1]:.6f}")

    # Find what threshold each actually reached
    for check_threshold in [0.05, 0.01, 0.005]:
        vanilla_iter = find_convergence_iteration(iterations, vanilla_values, nash_value,
                                                  check_threshold)
        plus_iter = find_convergence_iteration(iterations, plus_values, nash_value, check_threshold)
        print(f"\nThreshold {check_threshold}: Vanilla={vanilla_iter}, CFR+={plus_iter}")