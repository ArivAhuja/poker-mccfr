# mccfr_p_parallel.py - MCCFR-P Implementation with Proper Parallelization
"""
MCCFR-P (Monte Carlo Counterfactual Regret Minimization with Pruning) 
implementation for Simplified Limit Texas Hold'em using OpenSpiel universal_poker.

This implementation includes:
- Full MCCFR-P algorithm with pruning and Linear CFR discounting
- Proper multiprocessing using shared memory and worker pools
- Comprehensive logging and checkpointing
- Efficient batch updates to reduce lock contention

Key improvements for parallelization:
- Uses shared memory arrays for regrets and strategies
- Creates game instances in worker processes
- Batches updates to reduce synchronization overhead
- Proper worker initialization to avoid pickling issues
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
import contextlib
import argparse
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any
import multiprocessing as mp
from multiprocessing import Pool, Manager, RawArray, Lock
from dataclasses import dataclass
import ctypes
import json
import hashlib

# Constants
SUITS = 4
RANKS = 13
MAX_GROUP_INDEX = 0x100000
MAX_ROUNDS = 8
CARDS = 52

# Global variables for worker processes
worker_game = None
worker_config = None
worker_regret_data = None
worker_strategy_data = None
worker_info_state_map = None
worker_regret_lock = None
worker_strategy_lock = None
worker_update_queue = None

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
                with open(log_file, 'w') as f:
                    f.write('')
                    
            # Also remove any rotated log files
            for i in range(1, 11):
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
    file_handler.addFilter(hostname_filter)
    
    # Error log file
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=50*1024*1024,  # 50MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    error_handler.addFilter(hostname_filter)
    
    # Performance metrics log
    perf_handler = logging.handlers.RotatingFileHandler(
        perf_log_file,
        maxBytes=50*1024*1024,  # 50MB
        backupCount=5
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(detailed_formatter)
    perf_handler.addFilter(hostname_filter)
    perf_handler.addFilter(lambda record: record.name == 'performance')
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # Create performance logger
    perf_logger = logging.getLogger('performance')
    perf_logger.handlers.clear()
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
    
    # Parallelization
    num_processes: int = mp.cpu_count()
    batch_size: int = 100  # Iterations per batch
    update_batch_size: int = 1000  # Batch size for updates
    
    # Game specific
    num_players: int = 2
    max_actions: int = 10  # Maximum actions per info state
    
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


class SharedRegretMinimizer:
    """Shared memory regret and strategy storage for multiprocessing"""
    
    def __init__(self, max_info_states: int = 1000000, max_actions: int = 10):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_info_states = max_info_states
        self.max_actions = max_actions
        
        # Shared memory arrays
        self.regret_array = mp.RawArray(ctypes.c_double, max_info_states * max_actions)
        self.strategy_array = mp.RawArray(ctypes.c_double, max_info_states * max_actions)
        self.action_count_array = mp.RawArray(ctypes.c_int, max_info_states)
        
        # Info state mapping (hash -> index)
        self.manager = Manager()
        self.info_state_map = self.manager.dict()
        self.next_index = mp.Value('i', 0)
        
        # Locks
        self.regret_lock = mp.Lock()
        self.strategy_lock = mp.Lock()
        self.map_lock = mp.Lock()
        
        # Update queue for batched updates
        self.update_queue = self.manager.Queue()
        
        self.logger.info(f"SharedRegretMinimizer initialized with capacity for "
                        f"{max_info_states:,} info states")
    
    def get_or_create_index(self, info_state: str, num_actions: int) -> int:
        """Get index for info state, creating if necessary"""
        # Use hash for consistent mapping
        state_hash = hashlib.md5(info_state.encode()).hexdigest()
        
        with self.map_lock:
            if state_hash in self.info_state_map:
                return self.info_state_map[state_hash]
            
            if self.next_index.value >= self.max_info_states:
                raise RuntimeError("Maximum info states exceeded")
            
            idx = self.next_index.value
            self.info_state_map[state_hash] = idx
            self.next_index.value += 1
            
            # Initialize action count
            self.action_count_array[idx] = num_actions
            
            return idx
    
    def get_regrets_from_array(self, idx: int, num_actions: int) -> List[float]:
        """Get regrets from shared array"""
        start = idx * self.max_actions
        return [self.regret_array[start + i] for i in range(num_actions)]
    
    def get_strategy_from_array(self, idx: int, num_actions: int) -> List[float]:
        """Get strategy sum from shared array"""
        start = idx * self.max_actions
        return [self.strategy_array[start + i] for i in range(num_actions)]
    
    def queue_regret_update(self, info_state: str, action_regrets: List[float]):
        """Queue regret update for batch processing"""
        self.update_queue.put(('regret', info_state, action_regrets))
    
    def queue_strategy_update(self, info_state: str, strategy: List[float]):
        """Queue strategy update for batch processing"""
        self.update_queue.put(('strategy', info_state, strategy))
    
    def process_update_batch(self, regret_floor: float):
        """Process a batch of updates"""
        updates = []
        try:
            while len(updates) < 1000 and not self.update_queue.empty():
                updates.append(self.update_queue.get_nowait())
        except:
            pass
        
        if not updates:
            return
        
        # Group updates by type
        regret_updates = defaultdict(list)
        strategy_updates = defaultdict(list)
        
        for update_type, info_state, values in updates:
            if update_type == 'regret':
                regret_updates[info_state].append(values)
            else:
                strategy_updates[info_state].append(values)
        
        # Apply regret updates
        with self.regret_lock:
            for info_state, updates_list in regret_updates.items():
                num_actions = len(updates_list[0])
                idx = self.get_or_create_index(info_state, num_actions)
                start = idx * self.max_actions
                
                # Sum all updates for this info state
                total_update = [0.0] * num_actions
                for update in updates_list:
                    for i in range(num_actions):
                        total_update[i] += update[i]
                
                # Apply update with floor
                for i in range(num_actions):
                    self.regret_array[start + i] = max(
                        self.regret_array[start + i] + total_update[i],
                        regret_floor
                    )
        
        # Apply strategy updates
        with self.strategy_lock:
            for info_state, updates_list in strategy_updates.items():
                num_actions = len(updates_list[0])
                idx = self.get_or_create_index(info_state, num_actions)
                start = idx * self.max_actions
                
                # Sum all updates
                for update in updates_list:
                    for i in range(num_actions):
                        self.strategy_array[start + i] += update[i]
    
    def apply_discounting(self, discount: float):
        """Apply Linear CFR discounting"""
        self.logger.info(f"Applying discount factor: {discount:.4f}")
        
        with self.regret_lock:
            for i in range(self.next_index.value * self.max_actions):
                self.regret_array[i] *= discount
        
        with self.strategy_lock:
            for i in range(self.next_index.value * self.max_actions):
                self.strategy_array[i] *= discount
    
    def get_state_dict(self) -> Dict:
        """Get state for checkpointing"""
        with self.regret_lock, self.strategy_lock:
            # Convert shared arrays to regular dicts
            regrets = {}
            strategies = {}
            
            for state_hash, idx in self.info_state_map.items():
                num_actions = self.action_count_array[idx]
                if num_actions > 0:
                    regrets[state_hash] = self.get_regrets_from_array(idx, num_actions)
                    strategies[state_hash] = self.get_strategy_from_array(idx, num_actions)
            
            return {
                'regrets': regrets,
                'strategies': strategies,
                'info_state_map': dict(self.info_state_map),
                'next_index': self.next_index.value
            }
    
    def load_state_dict(self, state_dict: Dict):
        """Load state from checkpoint"""
        with self.regret_lock, self.strategy_lock:
            # Clear current state
            self.next_index.value = 0
            self.info_state_map.clear()
            
            # Load info state mapping
            for state_hash, idx in state_dict['info_state_map'].items():
                self.info_state_map[state_hash] = idx
            
            self.next_index.value = state_dict['next_index']
            
            # Load regrets and strategies
            for state_hash, regrets in state_dict['regrets'].items():
                if state_hash in self.info_state_map:
                    idx = self.info_state_map[state_hash]
                    start = idx * self.max_actions
                    self.action_count_array[idx] = len(regrets)
                    for i, r in enumerate(regrets):
                        self.regret_array[start + i] = r
            
            for state_hash, strategy in state_dict['strategies'].items():
                if state_hash in self.info_state_map:
                    idx = self.info_state_map[state_hash]
                    start = idx * self.max_actions
                    for i, s in enumerate(strategy):
                        self.strategy_array[start + i] = s


def create_simple_limit_holdem_game():
    """Create limit hold'em game using OpenSpiel universal_poker"""
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
    return game


def init_worker(game_string: str, config: Config, 
                regret_array, strategy_array, action_count_array,
                info_state_map, next_index,
                regret_lock, strategy_lock, map_lock, update_queue):
    """Initialize worker process with game and shared data"""
    global worker_game, worker_config
    global worker_regret_data, worker_strategy_data, worker_action_count
    global worker_info_state_map, worker_next_index
    global worker_regret_lock, worker_strategy_lock, worker_map_lock
    global worker_update_queue
    
    # Create game instance in worker
    worker_game = create_simple_limit_holdem_game()
    worker_config = config
    
    # Store references to shared data
    worker_regret_data = regret_array
    worker_strategy_data = strategy_array
    worker_action_count = action_count_array
    worker_info_state_map = info_state_map
    worker_next_index = next_index
    worker_regret_lock = regret_lock
    worker_strategy_lock = strategy_lock
    worker_map_lock = map_lock
    worker_update_queue = update_queue


def get_or_create_index_worker(info_state: str, num_actions: int) -> int:
    """Get index for info state in worker"""
    state_hash = hashlib.md5(info_state.encode()).hexdigest()
    
    if state_hash in worker_info_state_map:
        return worker_info_state_map[state_hash]
    
    # Need to create new index
    with worker_map_lock:
        # Double-check after acquiring lock
        if state_hash in worker_info_state_map:
            return worker_info_state_map[state_hash]
        
        if worker_next_index.value >= 1000000:  # Max info states
            raise RuntimeError("Maximum info states exceeded")
        
        idx = worker_next_index.value
        worker_info_state_map[state_hash] = idx
        worker_next_index.value += 1
        worker_action_count[idx] = num_actions
        
        return idx


def get_current_strategy_worker(info_state: str, num_actions: int) -> np.ndarray:
    """Get current strategy using regret matching in worker"""
    try:
        idx = get_or_create_index_worker(info_state, num_actions)
        start = idx * worker_config.max_actions
        
        regrets = [worker_regret_data[start + i] for i in range(num_actions)]
        positive_regrets = np.maximum(regrets, 0)
        sum_positive = np.sum(positive_regrets)
        
        if sum_positive > 0:
            return positive_regrets / sum_positive
        else:
            return np.ones(num_actions) / num_actions
    except:
        return np.ones(num_actions) / num_actions


def traverse_worker(state: pyspiel.State, player: int, rng: random.Random) -> float:
    """Standard MCCFR traversal in worker"""
    if state.is_terminal():
        return state.returns()[player]
    
    if state.is_chance_node():
        outcomes, probs = zip(*state.chance_outcomes())
        action = rng.choices(outcomes, weights=probs)[0]
        state.apply_action(action)
        return traverse_worker(state, player, rng)
    
    current_player = state.current_player()
    info_state_str = state.information_state_string()
    legal_actions = state.legal_actions()
    
    if current_player == player:
        strategy = get_current_strategy_worker(info_state_str, len(legal_actions))
        
        action_values = []
        for i, action in enumerate(legal_actions):
            state_copy = state.clone()
            state_copy.apply_action(action)
            value = traverse_worker(state_copy, player, rng)
            action_values.append(value)
        
        expected_value = np.dot(strategy, action_values)
        
        # Queue regret update
        action_regrets = [av - expected_value for av in action_values]
        worker_update_queue.put(('regret', info_state_str, action_regrets))
        
        return expected_value
    else:
        strategy = get_current_strategy_worker(info_state_str, len(legal_actions))
        action_idx = rng.choices(range(len(legal_actions)), weights=strategy)[0]
        state.apply_action(legal_actions[action_idx])
        return traverse_worker(state, player, rng)


def traverse_with_pruning_worker(state: pyspiel.State, player: int,
                                rng: random.Random) -> float:
    """MCCFR traversal with pruning in worker"""
    if state.is_terminal():
        return state.returns()[player]
    
    if state.is_chance_node():
        outcomes, probs = zip(*state.chance_outcomes())
        action = rng.choices(outcomes, weights=probs)[0]
        state.apply_action(action)
        return traverse_with_pruning_worker(state, player, rng)
    
    current_player = state.current_player()
    info_state_str = state.information_state_string()
    legal_actions = state.legal_actions()
    
    if current_player == player:
        strategy = get_current_strategy_worker(info_state_str, len(legal_actions))
        
        # Get current regrets for pruning check
        try:
            idx = get_or_create_index_worker(info_state_str, len(legal_actions))
            start = idx * worker_config.max_actions
            current_regrets = [worker_regret_data[start + i] for i in range(len(legal_actions))]
        except:
            current_regrets = [0.0] * len(legal_actions)
        
        action_values = []
        explored = []
        expected_value = 0.0
        
        for i, action in enumerate(legal_actions):
            if current_regrets[i] > worker_config.prune_threshold:
                state_copy = state.clone()
                state_copy.apply_action(action)
                value = traverse_with_pruning_worker(state_copy, player, rng)
                action_values.append(value)
                explored.append(True)
                expected_value += strategy[i] * value
            else:
                action_values.append(0.0)
                explored.append(False)
        
        # Queue regret update
        action_regrets = []
        for i in range(len(legal_actions)):
            if explored[i]:
                action_regrets.append(action_values[i] - expected_value)
            else:
                action_regrets.append(0.0)
        
        worker_update_queue.put(('regret', info_state_str, action_regrets))
        
        return expected_value
    else:
        strategy = get_current_strategy_worker(info_state_str, len(legal_actions))
        action_idx = rng.choices(range(len(legal_actions)), weights=strategy)[0]
        state.apply_action(legal_actions[action_idx])
        return traverse_with_pruning_worker(state, player, rng)


def update_strategy_worker(state: pyspiel.State, player: int, rng: random.Random):
    """Update average strategy in worker"""
    if state.is_terminal():
        return
    
    if state.is_chance_node():
        outcomes, probs = zip(*state.chance_outcomes())
        action = rng.choices(outcomes, weights=probs)[0]
        state.apply_action(action)
        update_strategy_worker(state, player, rng)
        return
    
    current_player = state.current_player()
    
    if current_player == player:
        info_state_str = state.information_state_string()
        legal_actions = state.legal_actions()
        
        strategy = get_current_strategy_worker(info_state_str, len(legal_actions))
        
        # Queue strategy update
        worker_update_queue.put(('strategy', info_state_str, list(strategy)))
        
        # Continue with sampled action
        action_idx = rng.choices(range(len(legal_actions)), weights=strategy)[0]
        state.apply_action(legal_actions[action_idx])
        update_strategy_worker(state, player, rng)
    else:
        for action in state.legal_actions():
            state_copy = state.clone()
            state_copy.apply_action(action)
            update_strategy_worker(state_copy, player, rng)


def train_iteration_worker(args):
    """Single training iteration in worker process"""
    iteration, process_id, seed = args
    rng = random.Random(seed)
    
    results = []
    
    # Update strategies periodically
    if iteration % worker_config.strategy_interval == 0:
        for player in range(worker_config.num_players):
            state = worker_game.new_initial_state()
            update_strategy_worker(state, player, rng)
    
    # Main MCCFR traversal
    for player in range(worker_config.num_players):
        state = worker_game.new_initial_state()
        
        if iteration > worker_config.prune_threshold_iterations:
            if rng.random() < worker_config.prune_probability:
                value = traverse_with_pruning_worker(state, player, rng)
            else:
                value = traverse_worker(state, player, rng)
        else:
            value = traverse_worker(state, player, rng)
        
        results.append(value)
    
    return results


class MCCFRPSolver:
    """MCCFR-P Solver with proper multiprocessing"""
    
    def __init__(self, game: pyspiel.Game, config: Config = Config()):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing MCCFR-P Solver with multiprocessing")
        
        self.game = game
        self.config = config
        
        # Initialize shared memory storage
        self.minimizer = SharedRegretMinimizer(
            max_info_states=1000000,
            max_actions=config.max_actions
        )
        
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
        
        # Initialize worker pool
        self.pool = None
        self._init_pool()
        
        self.logger.info(f"Solver initialized with {config.num_processes} processes")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.warning("Shutdown signal received. Saving checkpoint...")
        self.should_stop = True
    
    def _init_pool(self):
        """Initialize multiprocessing pool"""
        if self.pool:
            self.pool.close()
            self.pool.join()
        
        # Prepare initializer arguments
        init_args = (
            None,  # game_string (we'll create game in worker)
            self.config,
            self.minimizer.regret_array,
            self.minimizer.strategy_array,
            self.minimizer.action_count_array,
            self.minimizer.info_state_map,
            self.minimizer.next_index,
            self.minimizer.regret_lock,
            self.minimizer.strategy_lock,
            self.minimizer.map_lock,
            self.minimizer.update_queue
        )
        
        self.pool = Pool(
            processes=self.config.num_processes,
            initializer=init_worker,
            initargs=init_args
        )
    
    def save_checkpoint(self):
        """Save training state"""
        self.logger.info(f"Saving checkpoint at iteration {self.iteration}")
        
        checkpoint = {
            'iteration': self.iteration,
            'minimizer_state': self.minimizer.get_state_dict(),
            'config': self.config,
            'elapsed_time': time.time() - self.start_time
        }
        
        # Log checkpoint statistics
        num_info_states = self.minimizer.next_index.value
        
        self.logger.info(f"Checkpoint stats - Info states: {num_info_states:,}")
        
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
            self.minimizer.load_state_dict(checkpoint['minimizer_state'])
            
            self.logger.info(
                f"Resumed from iteration {self.iteration:,} with "
                f"{self.minimizer.next_index.value:,} info states"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
            return False
    
    def train(self, max_iterations: Optional[int] = None):
        """Train using MCCFR-P with multiprocessing"""
        self.logger.info(f"Starting MCCFR-P training with {self.config.num_processes} processes")
        
        # Try to load checkpoint
        self.load_checkpoint()
        
        pbar = tqdm(initial=self.iteration, desc="MCCFR-P Training")
        
        while not self.should_stop:
            if max_iterations and self.iteration >= max_iterations:
                self.logger.info(f"Reached max iterations: {max_iterations}")
                break
            
            # Prepare batch of iterations
            batch_args = []
            batch_size = self.config.batch_size
            
            for i in range(batch_size):
                if self.should_stop or (max_iterations and self.iteration + i >= max_iterations):
                    break
                
                args = (
                    self.iteration + i,
                    i % self.config.num_processes,
                    random.randint(0, 2**32-1)
                )
                batch_args.append(args)
            
            if not batch_args:
                break
            
            try:
                # Run batch in parallel
                self.pool.map(train_iteration_worker, batch_args)
                
                # Process update queue
                self.minimizer.process_update_batch(self.config.regret_floor)
                
                self.iteration += len(batch_args)
                
            except Exception as e:
                self.logger.error(f"Error in training batch: {e}", exc_info=True)
                self.should_stop = True
                break
            
            # Apply discounting
            if self.iteration < self.config.lcfr_threshold and \
               self.iteration % self.config.discount_interval == 0:
                discount = self.iteration / self.config.discount_interval / \
                         (self.iteration / self.config.discount_interval + 1)
                self.minimizer.apply_discounting(discount)
            
            # Regular logging
            if self.iteration % self.config.log_interval == 0:
                num_info_states = self.minimizer.next_index.value
                
                self.perf_monitor.log_metrics(
                    self.iteration,
                    num_info_states,
                    num_info_states,
                    {
                        'pruning_active': self.iteration > self.config.prune_threshold_iterations,
                        'discount_active': self.iteration < self.config.lcfr_threshold,
                        'processes': self.config.num_processes
                    }
                )
            
            # Detailed logging
            if self.iteration % self.config.detailed_log_interval == 0:
                self._log_detailed_stats()
            
            # Checkpoint
            if self.iteration % self.config.checkpoint_interval == 0:
                self.save_checkpoint()
            
            pbar.update(len(batch_args))
            
            # Update progress bar description
            elapsed = time.time() - self.start_time
            iter_per_sec = self.iteration / elapsed if elapsed > 0 else 0
            pbar.set_description(
                f"MCCFR-P Training (iter/s: {iter_per_sec:.1f})")
        
        pbar.close()
        
        # Cleanup pool
        if self.pool:
            self.pool.close()
            self.pool.join()
        
        # Final checkpoint
        self.save_checkpoint()
        self.logger.info(f"Training stopped at iteration {self.iteration:,}")
    
    def _log_detailed_stats(self):
        """Log detailed statistics"""
        self.logger.info("=== Detailed Statistics ===")
        
        # Sample some info states to log
        sample_size = min(10, self.minimizer.next_index.value)
        if sample_size > 0:
            sample_indices = random.sample(range(self.minimizer.next_index.value), sample_size)
            
            self.logger.info("Sample info states:")
            for idx in sample_indices:
                num_actions = self.minimizer.action_count_array[idx]
                if num_actions > 0:
                    regrets = self.minimizer.get_regrets_from_array(idx, num_actions)
                    strategy_sum = self.minimizer.get_strategy_from_array(idx, num_actions)
                    
                    self.logger.info(f"  Info state index {idx}:")
                    self.logger.info(f"    Regrets: {regrets}")
                    total = sum(strategy_sum)
                    if total > 0:
                        avg_strategy = [s/total for s in strategy_sum]
                        self.logger.info(f"    Avg strategy: {avg_strategy}")
    
    def get_average_strategy(self, info_state: str, num_actions: int) -> np.ndarray:
        """Get average strategy for an info state"""
        try:
            state_hash = hashlib.md5(info_state.encode()).hexdigest()
            if state_hash not in self.minimizer.info_state_map:
                return np.ones(num_actions) / num_actions
            
            idx = self.minimizer.info_state_map[state_hash]
            strategy_sum = self.minimizer.get_strategy_from_array(idx, num_actions)
            total = sum(strategy_sum)
            
            if total > 0:
                return np.array(strategy_sum) / total
            else:
                return np.ones(num_actions) / num_actions
        except:
            return np.ones(num_actions) / num_actions
    
    def save_strategy(self, path: str):
        """Save trained strategy to compressed file"""
        self.logger.info(f"Saving strategy to {path}")
        
        # Get final strategies
        strategy = {}
        state_dict = self.minimizer.get_state_dict()
        
        # Reverse map from hash to actual info state
        # Since we only have hashes, we'll save the strategy by hash
        for state_hash, strategy_sum in state_dict['strategies'].items():
            total = sum(strategy_sum)
            if total > 0:
                avg_strategy = [s / total for s in strategy_sum]
                # Only save non-uniform strategies
                if not all(abs(s - avg_strategy[0]) < 1e-6 for s in avg_strategy):
                    strategy[state_hash] = avg_strategy
        
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


def main():
    """Main training script"""
    
    # Command line arguments
    parser = argparse.ArgumentParser(description='MCCFR-P Training')
    parser.add_argument('--processes', type=int, default=mp.cpu_count(),
                       help='Number of processes to use')
    parser.add_argument('--iterations', type=int, default=None,
                       help='Maximum iterations (default: run until stopped)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Iterations per batch')
    parser.add_argument('--clear-logs', action='store_true',
                       help='Clear existing logs before starting')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(clear_logs=args.clear_logs)
    
    logger = logging.getLogger("main")
    logger.info(f"MCCFR-P Training for Limit Hold'em (Parallel Version)")
    
    config = Config(
        num_processes=args.processes,
        batch_size=args.batch_size,
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
    logger.info(f"  Using {config.num_processes} processes")
    logger.info(f"  Total memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    logger.info(f"  Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    game = create_simple_limit_holdem_game()
    
    # Create solver
    logger.info("\nInitializing solver...")
    solver = MCCFRPSolver(game, config)
    
    # Train
    solver.train(max_iterations=args.iterations)
    
    # Save final strategy
    logger.info("\nSaving final strategy...")
    strategy_file = f"limit_holdem_strategy_parallel.pkl.gz"
    solver.save_strategy(strategy_file)
    
    # Final statistics
    elapsed = time.time() - solver.start_time
    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info(f"Total iterations: {solver.iteration:,}")
    logger.info(f"Total time: {elapsed:.1f} seconds ({elapsed/3600:.1f} hours)")
    logger.info(f"Average speed: {solver.iteration/elapsed:.1f} iterations/second")
    logger.info(f"Used {config.num_processes} processes")
    logger.info("=" * 80)
    
    # Log file locations
    logger.info("\nOutput files:")
    logger.info(f"  Strategy: {strategy_file}")
    logger.info(f"  Checkpoint: {os.path.join(config.checkpoint_dir, 'mccfr_checkpoint.pkl.gz')}")


if __name__ == "__main__":
    main()