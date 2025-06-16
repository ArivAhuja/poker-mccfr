# mccfr_p.py - MCCFR-P Implementation with Logging
"""
MCCFR-P (Monte Carlo Counterfactual Regret Minimization with Pruning) 
implementation for Limit Texas Hold'em using OpenSpiel universal_poker.

This implementation includes:
- Full MCCFR-P algorithm with pruning and Linear CFR discounting
- Sequential processing (to avoid multiprocessing pickling issues)
- Comprehensive logging and checkpointing

File Structure (all in the mccfr folder):
- logs/                         : Training logs and performance metrics
- checkpoints/                  : Periodic training state saves
- limit_holdem_strategy.pkl.gz : Final trained strategy

Note: Preflop uses direct indexing (169 buckets, index = cluster)

Requirements:
- numpy, pandas, tqdm, psutil
- open_spiel (install with: pip install open_spiel)
"""

# Check required imports first
import sys
try:
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    import psutil
except ImportError as e:
    print(f"Missing required module: {e}")
    print("Please install requirements:")
    print("  pip install numpy pandas tqdm psutil")
    sys.exit(1)

# Check OpenSpiel separately as it often has installation issues
try:
    import pyspiel
except ImportError:
    print("ERROR: OpenSpiel is not installed or not found!")
    print("\nTo install OpenSpiel:")
    print("  pip install open_spiel")
    print("\nFor more info: https://github.com/deepmind/open_spiel")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Failed to import OpenSpiel: {e}")
    sys.exit(1)

# Standard library imports
import os
import pickle
import gzip
import time
import signal
import re
import logging
import logging.handlers
import socket
import random
import sys
import contextlib
import argparse
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from multiprocessing import Pool, Manager
from dataclasses import dataclass

# Constants for Kevin Waugh indexer
SUITS = 4
RANKS = 13
MAX_GROUP_INDEX = 0x100000
MAX_ROUNDS = 8
CARDS = 52

# Custom filter to add hostname to all log records
class HostnameFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.hostname = socket.gethostname()
    
    def filter(self, record):
        record.hostname = self.hostname
        return True

def setup_logging(log_dir: str = "logs", clear_logs: bool = True):
    """Setup comprehensive logging system"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Define log file paths
    main_log_file = os.path.join(log_dir, 'mccfr_training.log')
    error_log_file = os.path.join(log_dir, 'mccfr_errors.log')
    perf_log_file = os.path.join(log_dir, 'mccfr_performance.log')
    
    # Clear existing logs if requested
    if clear_logs:
        for log_file in [main_log_file, error_log_file, perf_log_file]:
            if os.path.exists(log_file):
                # Truncate the file
                with open(log_file, 'w') as f:
                    f.write('')
                    
            # Also remove any rotated log files
            for i in range(1, 11):  # Remove up to 10 backup files
                rotated_file = f"{log_file}.{i}"
                if os.path.exists(rotated_file):
                    os.remove(rotated_file)
    
    # Create hostname filter
    hostname_filter = HostnameFilter()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(hostname)s - %(processName)s[%(process)d] - '
        '%(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Main log file handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        main_log_file,
        maxBytes=100*1024*1024,  # 100MB
        backupCount=10
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    file_handler.addFilter(hostname_filter)  # Add filter to handler
    
    # Error log file
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=50*1024*1024,  # 50MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    error_handler.addFilter(hostname_filter)  # Add filter to handler
    
    # Performance metrics log
    perf_handler = logging.handlers.RotatingFileHandler(
        perf_log_file,
        maxBytes=50*1024*1024,  # 50MB
        backupCount=5
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(detailed_formatter)
    perf_handler.addFilter(hostname_filter)  # Add filter to handler
    perf_handler.addFilter(lambda record: record.name == 'performance')
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # Create performance logger
    perf_logger = logging.getLogger('performance')
    perf_logger.handlers.clear()  # Clear any existing handlers
    perf_logger.addHandler(perf_handler)
    perf_logger.propagate = False
    
    if clear_logs:
        logging.info(f"Logs cleared - starting fresh logging session")
    
    logging.info(f"Logging initialized on {socket.gethostname()}")
    logging.info(f"Main log: {main_log_file}")
    logging.info(f"Error log: {error_log_file}")
    logging.info(f"Performance log: {perf_log_file}")


@dataclass
class Config:
    """Configuration for MCCFR-P solver"""
    # Thresholds for limit poker
    prune_threshold: float = -30000.0
    regret_floor: float = -50000.0
    prune_threshold_iterations: int = 10000
    
    # Intervals
    strategy_interval: int = 1000
    discount_interval: int = 10000
    lcfr_threshold: int = 40000
    
    # Probabilities
    prune_probability: float = 0.95
    
    # Parallelization (now just batch size for sequential processing)
    num_processes: int = 8  # Batch size for iterations
    
    # Game specific
    num_players: int = 2
    
    # Checkpointing
    checkpoint_interval: int = 10000
    checkpoint_dir: str = "checkpoints"
    
    # Logging
    log_interval: int = 1000
    detailed_log_interval: int = 10000


class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger('performance')
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.last_iteration = 0
        self.process = psutil.Process()
        
    def log_metrics(self, iteration: int, num_info_states: int, 
                   num_regrets: int, additional_metrics: dict = None):
        """Log performance metrics"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        interval = current_time - self.last_log_time
        iterations_in_interval = iteration - self.last_iteration
        
        # Calculate rates
        iter_per_sec = iterations_in_interval / interval if interval > 0 else 0
        total_iter_per_sec = iteration / elapsed if elapsed > 0 else 0
        
        # Memory usage
        memory_info = self.process.memory_info()
        memory_gb = memory_info.rss / (1024**3)
        
        # CPU usage
        cpu_percent = self.process.cpu_percent(interval=0.1)
        
        metrics = {
            'iteration': iteration,
            'elapsed_time': elapsed,
            'iter_per_sec': iter_per_sec,
            'total_iter_per_sec': total_iter_per_sec,
            'memory_gb': memory_gb,
            'cpu_percent': cpu_percent,
            'num_info_states': num_info_states,
            'num_regrets': num_regrets,
        }
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        # Log as structured data
        self.logger.info(f"METRICS: {metrics}")
        
        # Also log human-readable summary
        logging.info(
            f"Performance - Iter: {iteration:,} | "
            f"Speed: {iter_per_sec:.1f} iter/s | "
            f"Memory: {memory_gb:.1f} GB | "
            f"CPU: {cpu_percent:.1f}% | "
            f"Info States: {num_info_states:,}"
        )
        
        # Warn if memory usage is high
        available_memory = psutil.virtual_memory().available / (1024**3)
        if memory_gb > 10:
            logging.warning(f"High memory usage: {memory_gb:.1f} GB")
        if available_memory < 2:
            logging.warning(f"Low available memory: {available_memory:.1f} GB")
        
        self.last_log_time = current_time
        self.last_iteration = iteration

class RegretMinimizer:
    """Thread-safe regret and strategy storage"""
    
    def __init__(self, manager: Manager):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.regrets = manager.dict()
        self.avg_strategy_sum = manager.dict()
        self.regret_lock = manager.Lock()
        self.strategy_lock = manager.Lock()
        
        self.logger.info("RegretMinimizer initialized")
    
    def get_regrets(self, info_state: str) -> List[float]:
        """Get regrets for an info state"""
        return list(self.regrets.get(info_state, []))
    
    def update_regrets(self, info_state: str, action_regrets: List[float], 
                      regret_floor: float):
        """Update regrets with thread safety"""
        with self.regret_lock:
            current = self.regrets.get(info_state, [0.0] * len(action_regrets))
            # Handle case where stored regrets have different length
            if len(current) != len(action_regrets):
                # Reset to match current action count
                current = [0.0] * len(action_regrets)
            updated = [max(current[i] + action_regrets[i], regret_floor) 
                      for i in range(len(action_regrets))]
            self.regrets[info_state] = updated
    
    def update_strategy_sum(self, info_state: str, strategy: List[float]):
        """Update average strategy sum"""
        with self.strategy_lock:
            current = self.avg_strategy_sum.get(info_state, [0.0] * len(strategy))
            # Handle case where stored strategy has different length
            if len(current) != len(strategy):
                # Reset to match current action count
                current = [0.0] * len(strategy)
            updated = [current[i] + strategy[i] for i in range(len(strategy))]
            self.avg_strategy_sum[info_state] = updated
    
    def apply_discounting(self, discount: float):
        """Apply Linear CFR discounting"""
        self.logger.info(f"Applying discount factor: {discount:.4f}")
        
        with self.regret_lock:
            for key in list(self.regrets.keys()):
                self.regrets[key] = [r * discount for r in self.regrets[key]]
        
        with self.strategy_lock:
            for key in list(self.avg_strategy_sum.keys()):
                self.avg_strategy_sum[key] = [s * discount 
                                            for s in self.avg_strategy_sum[key]]


class MCCFRPSolver:
    """MCCFR-P Solver with no abstraction"""
    
    def __init__(self, game: pyspiel.Game,
                 config: Config = Config()):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing MCCFR-P Solver")
        
        self.game = game
        self.config = config
        
        # Initialize multiprocessing manager
        self.manager = Manager()
        self.minimizer = RegretMinimizer(self.manager)
        
        # Random seeds for each process
        self.seeds = [random.randint(0, 2**32-1) 
                     for _ in range(config.num_processes)]
        
        # Training statistics
        self.iteration = 0
        self.start_time = time.time()
        
        # Performance monitor
        self.perf_monitor = PerformanceMonitor()
        
        # Checkpointing
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(config.checkpoint_dir, 
                                          "mccfr_checkpoint.pkl.gz")
        
        # Signal handling for graceful shutdown
        self.should_stop = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info(f"Solver initialized with batch size {config.num_processes}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.warning("Shutdown signal received. Saving checkpoint...")
        self.should_stop = True
    
    def get_current_strategy(self, info_state: str, 
                           num_actions: int) -> np.ndarray:
        """Get current strategy using regret matching"""
        regrets = self.minimizer.get_regrets(info_state)
        
        if not regrets:
            return np.ones(num_actions) / num_actions
        
        # Handle case where stored regrets have different length
        if len(regrets) != num_actions:
            return np.ones(num_actions) / num_actions
        
        positive_regrets = np.maximum(regrets, 0)
        sum_positive = np.sum(positive_regrets)
        
        if sum_positive > 0:
            return positive_regrets / sum_positive
        else:
            return np.ones(num_actions) / num_actions
    
    def traverse(self, state: pyspiel.State, player: int, 
                rng: random.Random) -> float:
        """Standard MCCFR traversal"""
        if state.is_terminal():
            return state.returns()[player]
        
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            action = rng.choices(outcomes, weights=probs)[0]
            state.apply_action(action)
            return self.traverse(state, player, rng)
        
        current_player = state.current_player()
        info_state_str = state.information_state_string()
        legal_actions = state.legal_actions()
        
        if current_player == player:
            strategy = self.get_current_strategy(info_state_str, 
                                               len(legal_actions))
            
            action_values = []
            for i, action in enumerate(legal_actions):
                state_copy = state.clone()
                state_copy.apply_action(action)
                value = self.traverse(state_copy, player, rng)
                action_values.append(value)
            
            expected_value = np.dot(strategy, action_values)
            
            action_regrets = [av - expected_value for av in action_values]
            self.minimizer.update_regrets(info_state_str, action_regrets,
                                        self.config.regret_floor)
            
            return expected_value
        else:
            strategy = self.get_current_strategy(info_state_str, 
                                               len(legal_actions))
            action_idx = rng.choices(range(len(legal_actions)), 
                                   weights=strategy)[0]
            state.apply_action(legal_actions[action_idx])
            return self.traverse(state, player, rng)
    
    def traverse_with_pruning(self, state: pyspiel.State, player: int,
                            rng: random.Random) -> float:
        """MCCFR traversal with pruning"""
        if state.is_terminal():
            return state.returns()[player]
        
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            action = rng.choices(outcomes, weights=probs)[0]
            state.apply_action(action)
            return self.traverse_with_pruning(state, player, rng)
        
        current_player = state.current_player()
        info_state_str = state.information_state_string()
        legal_actions = state.legal_actions()
        
        if current_player == player:
            strategy = self.get_current_strategy(info_state_str, 
                                               len(legal_actions))
            
            current_regrets = self.minimizer.get_regrets(info_state_str)
            if not current_regrets:
                current_regrets = [0.0] * len(legal_actions)
            
            action_values = []
            explored = []
            expected_value = 0.0
            
            for i, action in enumerate(legal_actions):
                if current_regrets[i] > self.config.prune_threshold:
                    state_copy = state.clone()
                    state_copy.apply_action(action)
                    value = self.traverse_with_pruning(state_copy, player, rng)
                    action_values.append(value)
                    explored.append(True)
                    expected_value += strategy[i] * value
                else:
                    action_values.append(0.0)
                    explored.append(False)
            
            action_regrets = []
            for i in range(len(legal_actions)):
                if explored[i]:
                    action_regrets.append(action_values[i] - expected_value)
                else:
                    action_regrets.append(0.0)
            
            self.minimizer.update_regrets(info_state_str, action_regrets,
                                        self.config.regret_floor)
            
            return expected_value
        else:
            strategy = self.get_current_strategy(info_state_str, 
                                               len(legal_actions))
            action_idx = rng.choices(range(len(legal_actions)), 
                                   weights=strategy)[0]
            state.apply_action(legal_actions[action_idx])
            return self.traverse_with_pruning(state, player, rng)
    
    def update_strategy(self, state: pyspiel.State, player: int,
                       rng: random.Random):
        """Update average strategy"""
        if state.is_terminal():
            return
        
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            action = rng.choices(outcomes, weights=probs)[0]
            state.apply_action(action)
            self.update_strategy(state, player, rng)
            return
        
        current_player = state.current_player()
        
        if current_player == player:
            info_state_str = state.information_state_string()
            legal_actions = state.legal_actions()
            
            strategy = self.get_current_strategy(info_state_str, 
                                               len(legal_actions))
            
            # Update average strategy
            self.minimizer.update_strategy_sum(info_state_str, list(strategy))
            
            # Continue with sampled action
            action_idx = rng.choices(range(len(legal_actions)), 
                                   weights=strategy)[0]
            state.apply_action(legal_actions[action_idx])
            self.update_strategy(state, player, rng)
        else:
            for action in state.legal_actions():
                state_copy = state.clone()
                state_copy.apply_action(action)
                self.update_strategy(state_copy, player, rng)
    
    def train_iteration(self, args):
        """Single training iteration - runs in main process"""
        iteration, process_id = args
        rng = random.Random(self.seeds[process_id] + iteration)
        
        results = []
        
        # Update strategies periodically
        if iteration % self.config.strategy_interval == 0:
            for player in range(self.config.num_players):
                state = self.game.new_initial_state()
                self.update_strategy(state, player, rng)
        
        # Main MCCFR traversal
        for player in range(self.config.num_players):
            state = self.game.new_initial_state()
            
            if iteration > self.config.prune_threshold_iterations:
                if rng.random() < self.config.prune_probability:
                    value = self.traverse_with_pruning(state, player, rng)
                else:
                    value = self.traverse(state, player, rng)
            else:
                value = self.traverse(state, player, rng)
            
            results.append(value)
        
        return results
    
    def save_checkpoint(self):
        """Save training state"""
        self.logger.info(f"Saving checkpoint at iteration {self.iteration}")
        
        # Convert manager dicts to regular dicts for pickling
        checkpoint = {
            'iteration': self.iteration,
            'regrets': dict(self.minimizer.regrets),
            'avg_strategy_sum': dict(self.minimizer.avg_strategy_sum),
            'config': self.config,
            'elapsed_time': time.time() - self.start_time
        }
        
        # Log checkpoint statistics
        num_info_states = len(checkpoint['regrets'])
        num_strategies = len(checkpoint['avg_strategy_sum'])
        
        self.logger.info(
            f"Checkpoint stats - Info states: {num_info_states:,}, "
            f"Strategies: {num_strategies:,}"
        )
        
        try:
            with gzip.open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            file_size_mb = os.path.getsize(self.checkpoint_file) / (1024**2)
            self.logger.info(
                f"Checkpoint saved to {self.checkpoint_file} ({file_size_mb:.1f} MB)"
            )
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}", exc_info=True)
    
    def load_checkpoint(self):
        """Load training state"""
        if not os.path.exists(self.checkpoint_file):
            self.logger.info("No checkpoint found, starting fresh")
            return False
        
        self.logger.info(f"Loading checkpoint from {self.checkpoint_file}")
        
        try:
            with gzip.open(self.checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            
            self.iteration = checkpoint['iteration']
            
            # Restore data to manager dicts
            for key, value in checkpoint['regrets'].items():
                self.minimizer.regrets[key] = value
            
            for key, value in checkpoint['avg_strategy_sum'].items():
                self.minimizer.avg_strategy_sum[key] = value
            
            self.logger.info(
                f"Resumed from iteration {self.iteration:,} with "
                f"{len(checkpoint['regrets']):,} info states"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
            return False
    
    def train(self, max_iterations: Optional[int] = None):
        """Train using MCCFR-P"""
        self.logger.info(f"Starting MCCFR-P training (sequential mode)")
        
        # Try to load checkpoint
        self.load_checkpoint()
        
        pbar = tqdm(initial=self.iteration, desc="MCCFR-P Training")
        
        while not self.should_stop:
            if max_iterations and self.iteration >= max_iterations:
                self.logger.info(f"Reached max iterations: {max_iterations}")
                break
            
            # Run a batch of iterations sequentially
            # This avoids multiprocessing pickling issues
            batch_size = self.config.num_processes
            
            for i in range(batch_size):
                if self.should_stop or (max_iterations and self.iteration >= max_iterations):
                    break
                    
                # Run single iteration
                args = (self.iteration, i % self.config.num_processes)
                try:
                    self.train_iteration(args)
                except Exception as e:
                    self.logger.error(f"Error in training iteration: {e}", exc_info=True)
                    self.should_stop = True
                    break
                
                self.iteration += 1
            
            # Apply discounting
            if self.iteration < self.config.lcfr_threshold and \
               self.iteration % self.config.discount_interval == 0:
                discount = self.iteration / self.config.discount_interval / \
                         (self.iteration / self.config.discount_interval + 1)
                self.minimizer.apply_discounting(discount)
            
            # Regular logging
            if self.iteration % self.config.log_interval == 0:
                num_info_states = len(self.minimizer.regrets)
                num_strategies = len(self.minimizer.avg_strategy_sum)
                
                self.perf_monitor.log_metrics(
                    self.iteration,
                    num_info_states,
                    num_strategies,
                    {
                        'pruning_active': self.iteration > self.config.prune_threshold_iterations,
                        'discount_active': self.iteration < self.config.lcfr_threshold
                    }
                )
            
            # Detailed logging
            if self.iteration % self.config.detailed_log_interval == 0:
                self._log_detailed_stats()
            
            # Checkpoint
            if self.iteration % self.config.checkpoint_interval == 0:
                self.save_checkpoint()
            
            pbar.update(batch_size)
            
            # Update progress bar description
            elapsed = time.time() - self.start_time
            iter_per_sec = self.iteration / elapsed if elapsed > 0 else 0
            pbar.set_description(
                f"MCCFR-P Training (iter/s: {iter_per_sec:.1f})")
        
        pbar.close()
        
        # Final checkpoint
        self.save_checkpoint()
        self.logger.info(f"Training stopped at iteration {self.iteration:,}")
    
    def _log_detailed_stats(self):
        """Log detailed statistics"""
        self.logger.info("=== Detailed Statistics ===")
        
        # Sample some info states to log
        sample_size = min(10, len(self.minimizer.regrets))
        if sample_size > 0:
            sample_keys = random.sample(list(self.minimizer.regrets.keys()), sample_size)
            
            self.logger.info("Sample info states:")
            for key in sample_keys:
                regrets = self.minimizer.regrets[key]
                strategy_sum = self.minimizer.avg_strategy_sum.get(key, [])
                
                self.logger.info(f"  {key}:")
                self.logger.info(f"    Regrets: {regrets}")
                if strategy_sum:
                    total = sum(strategy_sum)
                    if total > 0:
                        avg_strategy = [s/total for s in strategy_sum]
                        self.logger.info(f"    Avg strategy: {avg_strategy}")
    
    def get_average_strategy(self, info_state: str, 
                           num_actions: int) -> np.ndarray:
        """Get average strategy for an info state"""
        strategy_sum = self.minimizer.avg_strategy_sum.get(info_state, 
                                                         [0.0] * num_actions)
        # Handle case where stored strategy has different length
        if len(strategy_sum) != num_actions:
            return np.ones(num_actions) / num_actions
            
        total = sum(strategy_sum)
        
        if total > 0:
            return np.array(strategy_sum) / total
        else:
            return np.ones(num_actions) / num_actions
    
    def save_strategy(self, path: str):
        """Save trained strategy to compressed file"""
        self.logger.info(f"Saving strategy to {path}")
        
        # Convert to regular dicts and compute average strategies
        strategy = {}
        for info_state, strategy_sum in self.minimizer.avg_strategy_sum.items():
            total = sum(strategy_sum)
            if total > 0:
                avg_strategy = [s / total for s in strategy_sum]
                # Only save non-uniform strategies to save space
                if not all(abs(s - avg_strategy[0]) < 1e-6 for s in avg_strategy):
                    strategy[info_state] = avg_strategy
        
        data = {
            'strategy': strategy,
            'iteration': self.iteration,
            'config': self.config,
            'elapsed_time': time.time() - self.start_time
        }
        
        try:
            with gzip.open(path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            file_size_mb = os.path.getsize(path) / (1024**2)
            self.logger.info(
                f"Strategy saved ({len(strategy):,} non-uniform info states, "
                f"{file_size_mb:.1f} MB)"
            )
        except Exception as e:
            self.logger.error(f"Failed to save strategy: {e}", exc_info=True)


def create_simple_limit_holdem_game():
    """Create limit hold'em game using OpenSpiel universal_poker"""
    logger = logging.getLogger("game_creation")
    
    logger.info("Creating limit hold'em game")
    
    CUSTOM_LIMIT_HOLDEM_ACPC_GAMEDEF = """\
GAMEDEF
limit
numPlayers = 2
numRounds = 2
blind = 1 2
raiseSize = 2 2
firstPlayer = 1
maxRaises = 2 2
numSuits = 2
numRanks = 6
numHoleCards = 2
numBoardCards = 0 3
stack = 20
END GAMEDEF
"""
    universal_poker = pyspiel.universal_poker
    game = universal_poker.load_universal_poker_from_acpc_gamedef(CUSTOM_LIMIT_HOLDEM_ACPC_GAMEDEF)
    
    # Log game information
    logger.info(f"Game created: {game}")
    logger.info(f"Number of players: {game.num_players()}")
    logger.info(f"Number of distinct actions: {game.num_distinct_actions()}")
    
    # Test initial state
    state = game.new_initial_state()
    logger.debug("Initial state created successfully")
    
    return game

def main():
    """Main training script"""
    
    # Setup logging first
    setup_logging()
    
    logger = logging.getLogger("main")
    logger.info(f"MCCFR-P Training for Limit Hold'em")
    
    config = Config(
        num_processes=mp.cpu_count(),  # Using num_processes for backward compatibility
        prune_threshold=-30000.0,
        strategy_interval=1000,
        discount_interval=10000,
        checkpoint_interval=10000,
        log_interval=1000,
        detailed_log_interval=10000,
        num_players=2
    )
    
    # System information
    logger.info("\nSystem Information:")
    logger.info(f"  Hostname: {socket.gethostname()}")
    logger.info(f"  CPU count: {mp.cpu_count()}")
    logger.info(f"  Total memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    logger.info(f"  Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    game = create_simple_limit_holdem_game()

    # Create solver
    logger.info("\nInitializing solver...")
    solver = MCCFRPSolver(game, config)
    
    # Train until stopped
    solver.train()
    
    # Save final strategy
    logger.info("\nSaving final strategy...")
    strategy_file = f"limit_holdem_strategy.pkl.gz"
    solver.save_strategy(strategy_file)
    
    # Final statistics
    elapsed = time.time() - solver.start_time
    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info(f"Total iterations: {solver.iteration:,}")
    logger.info(f"Total time: {elapsed:.1f} seconds ({elapsed/3600:.1f} hours)")
    logger.info(f"Average speed: {solver.iteration/elapsed:.1f} iterations/second")
    logger.info("=" * 80)
    
    # Log file locations
    logger.info("\nOutput files:")
    logger.info(f"  Strategy: {strategy_file}")
    logger.info(f"  Checkpoint: {os.path.join(config.checkpoint_dir, 'mccfr_checkpoint.pkl.gz')}")

if __name__ == "__main__":
    main()