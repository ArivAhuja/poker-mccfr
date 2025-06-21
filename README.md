# Poker MCCFR

This repository contains implementations and tools related to Monte Carlo Counterfactual Regret Minimization (MCCFR) for poker.

## Repository Structure Overview

*   [`abstraction/`](abstraction/): Contains initial ideas and scripts for abstracting No-Limit Hold'em (NLHE) poker. Please note that these are not yet finished or well-tested.
*   [`cfr/`](cfr/): Includes scripts for understanding and experimenting with Counterfactual Regret Minimization (CFR), as well as analysis and chart visualizations specifically for Kuhn Poker.
*   [`mccfr/`](mccfr/): Houses the core algorithm files for Monte Carlo Counterfactual Regret Minimization.
*   [`mccfr_c/`](mccfr_c/): Contains an incomplete and not well-tested C++ implementation of MCCFR.
*   [`gui/`](gui/): Contains files related to the graphical user interface for a simplified poker game.

## Getting Started

Follow these instructions to set up and run the MCCFR agent and the simplified poker GUI.

### 1. Clone the Repository

First, clone the GitHub repository to your local machine:

```bash
git clone https://github.com/your-repo-link/poker-mccfr.git
cd poker-mccfr
```

### 2. Install Requirements

It is highly recommended to use a virtual environment. Install the necessary Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The key dependencies include `absl-py`, `Flask`, `flask-cors`, `numpy`, `open_spiel`, `pandas`, `phevaluator`, `psutil`, `PyYAML`, `scikit-learn`, `scipy`, `tqdm`, and others.

### 3. Python Version

Ensure you are using Python version `3.9.6`. This version is specified in the `.python-version` file. You can check your current version with:

```bash
python --version
```

If you need to switch versions, consider using `pyenv` or `conda`.

### 4. Adjust Game Variant (If Necessary)

The MCCFR implementation can be adapted for different poker game variants. You may need to adjust the game parameters within the relevant Python files (e.g., in `mccfr/mccfr_p_parellel.py`) to match your desired game.

### 5. Run the MCCFR Algorithm

To run the parallel MCCFR algorithm, navigate to the `mccfr` directory and execute the script:

```bash
cd mccfr
python mccfr_p_parellel.py
```

**Note on Cluster Usage:** This code is designed to be runnable on a cluster, and you can use `sbatch` for job submission if you have access to a Slurm-managed cluster.

### 6. Stopping the Script and Strategy Blueprint

The MCCFR script will continue to run until you manually stop it. To stop the script, press `Ctrl+C` in your terminal. After stopping, a compressed strategy blueprint (e.g., `limit_holdem_strategy_parallel.pkl.gz` or a checkpoint file) will be created in the `mccfr/` directory. This file contains the learned strategy of your agent.

### 7. Launching the GUI and Playing Against Your Agent

The repository includes a simplified poker GUI that allows you to play against your trained agent.

**Simplified Game Features:**
*   6 ranks (2,3,4,5,6,7) × 2 suits = 12 cards total
*   2 rounds only: Preflop → Flop (no turn/river)
*   Fixed limit betting: $2-$4 structure
*   Max 2 raises per betting round
*   20 chip starting stacks
*   Heads-up play (You vs AI)

To launch the GUI, navigate to the `gui` directory and run the `run_server.py` script:

```bash
cd gui
python run_server.py
```

The script will automatically attempt to find and load your trained strategy file. If no compatible file is found, it may prompt you to extract one or use a fallback strategy. Once the server starts, open your web browser and navigate to:

```
http://localhost:5001
```

You can then play against your trained MCCFR agent.

### 8. Evaluate Best Response

To evaluate the best response to your trained agent's strategy, you can run the `nash_calculation.py` script:

```bash
cd mccfr
python nash_calculation.py
```
This script will help you assess the effectiveness of your agent's strategy by calculating the best response.
