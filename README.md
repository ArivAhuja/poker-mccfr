# Counterfactual Regret Minimization (CFR) Framework

This repository provides a modular framework for implementing **Counterfactual Regret Minimization (CFR)** algorithms in Python. The design is abstract and extensible, allowing support for a variety of games (e.g., Kuhn Poker, Leduc Hold'em, Texas Hold'em) using the [OpenSpiel](https://github.com/deepmind/open_spiel) library.

---

## 🧠 What is CFR?

**CFR** is an iterative self-play algorithm used to compute approximate Nash equilibria in **extensive-form games** (games with sequential moves, imperfect information, and chance events). It works by:

- Traversing the game tree recursively
- Tracking regrets and average strategies for each information set
- Using **regret matching** to update strategy profiles

---

## 📦 Structure

### `BaseCFR` Class

The abstract base class `BaseCFR` provides a **Vanilla CFR** implementation and contains:

- 🔧 **Core Data Structures**:

  - `regret_sum`: Regret values per information set and action
  - `strategy_sum`: Cumulative strategy used for averaging

- 🚀 **Key Methods**:

  - `cfr(...)`: Recursive traversal for computing regrets
  - `train(...)`: Repeated CFR iterations over players
  - `get_strategy(...)`: Current strategy based on regret matching
  - `get_average_strategy(...)`: Returns average strategy per info set

- 📜 **Abstract Methods**:
  These must be implemented in subclasses tailored to specific games:
  - `get_info_set`
  - `get_player`
  - `get_utility`
  - `get_num_actions`
  - `get_next_history`
  - `get_chance_outcomes`
  - `new_game`
  - `is_terminal`

---

## ✅ Supported Algorithms

Currently supported:

- [x] **Vanilla CFR**

Planned:

- [ ] **CFR+** (optimistic regret updates)
- [ ] **Monte Carlo CFR** (sampling-based CFR)

---

## 🕹️ Supported Games

To support a specific game, implement a subclass of `BaseCFR` and define the abstract methods using game-specific logic from `pyspiel`.

Example game classes to implement:

- `KuhnVanillaCFR`
- `LeducHoldemVanillaCFR`
- `TexasLimitVanillaCFR`

---

## 🔄 Usage

### Training

```python
cfr = YourGameCFR(num_players=2)
cfr.train(iterations=10000)
```
