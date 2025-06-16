// mccfr_p.h
#ifndef MCCFR_P_H
#define MCCFR_P_H

#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <mutex>
#include <random>
#include <fstream>
#include <sstream>
#include <omp.h>

#include "spiel.h"
#include "games/universal_poker/universal_poker.h"
#include "algorithms/cfr.h"

namespace mccfr_p {

// Abstraction manager for loading and managing card abstractions
class AbstractionManager {
public:
    AbstractionManager(const std::string& preflop_path,
                      const std::string& flop_path,
                      const std::string& turn_path,
                      const std::string& river_path);
    
    int GetCluster(int round, uint64_t hand_index) const;
    
private:
    std::vector<std::vector<int>> clusters_; // [round][hand_index] -> cluster
    
    std::vector<int> LoadCSV(const std::string& path);
};

// Abstract information state that includes clustering
class AbstractInfoState {
public:
    AbstractInfoState(const open_spiel::State& state, 
                     const AbstractionManager& abstraction);
    
    std::string ToString() const;
    bool operator==(const AbstractInfoState& other) const;
    
    struct Hash {
        std::size_t operator()(const AbstractInfoState& s) const {
            return std::hash<std::string>{}(s.ToString());
        }
    };
    
private:
    std::string base_info_state_;
    int cluster_;
    int round_;
    int player_;
};

// MCCFR-P Solver
class MCCFRPSolver {
public:
    struct Config {
        // Thresholds (smaller than Pluribus for limit poker)
        double prune_threshold = -30000.0;  // vs -300M in Pluribus
        double regret_floor = -50000.0;     // vs -310M in Pluribus
        int prune_threshold_iterations = 1000; // vs 20k in Pluribus
        
        // Intervals
        int strategy_interval = 100;
        int discount_interval = 100;
        int lcfr_threshold = 2000;
        
        // Probabilities
        double prune_probability = 0.95;
        
        // Parallelization
        int num_threads = 8;
    };
    
    MCCFRPSolver(std::shared_ptr<const open_spiel::Game> game,
                 std::unique_ptr<AbstractionManager> abstraction,
                 const Config& config = Config());
    
    void Train(int iterations);
    
    // Get average strategy for a given information state
    std::vector<double> GetAverageStrategy(const AbstractInfoState& info_state) const;
    
private:
    using RegretTable = std::unordered_map<AbstractInfoState, 
                                          std::vector<double>, 
                                          AbstractInfoState::Hash>;
    using StrategyTable = std::unordered_map<AbstractInfoState,
                                            std::vector<double>,
                                            AbstractInfoState::Hash>;
    
    std::shared_ptr<const open_spiel::Game> game_;
    std::unique_ptr<AbstractionManager> abstraction_;
    Config config_;
    
    // Thread-safe storage
    std::vector<std::mutex> regret_mutexes_;  // One per bucket
    std::vector<std::mutex> strategy_mutexes_;
    
    RegretTable regrets_;
    StrategyTable avg_strategy_;
    StrategyTable avg_strategy_sum_;
    
    // Random number generators per thread
    std::vector<std::mt19937> rngs_;
    
    // Core MCCFR functions
    double Traverse(std::unique_ptr<open_spiel::State> state, 
                   int traversing_player,
                   int thread_id);
    
    double TraverseWithPruning(std::unique_ptr<open_spiel::State> state,
                              int traversing_player,
                              int thread_id);
    
    void UpdateStrategy(std::unique_ptr<open_spiel::State> state,
                       int updating_player,
                       int thread_id);
    
    std::vector<double> GetCurrentStrategy(const AbstractInfoState& info_state);
    
    // Utility functions
    int GetMutexIndex(const AbstractInfoState& info_state) const;
    void ApplyDiscounting(double discount_factor);
    bool ShouldPrune(double regret, int iteration) const;
};

// Parallel training coordinator
class ParallelMCCFRPTrainer {
public:
    ParallelMCCFRPTrainer(const std::string& game_string,
                         const std::string& preflop_path,
                         const std::string& flop_path,
                         const std::string& turn_path,
                         const std::string& river_path,
                         const MCCFRPSolver::Config& config);
    
    void Train(int iterations);
    void SaveStrategy(const std::string& path) const;
    void LoadStrategy(const std::string& path);
    
private:
    std::unique_ptr<MCCFRPSolver> solver_;
    std::shared_ptr<const open_spiel::Game> game_;
};

} // namespace mccfr_p

#endif // MCCFR_P_H