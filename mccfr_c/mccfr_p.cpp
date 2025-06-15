// mccfr_p.cpp
#include "mccfr_p.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace mccfr_p {

// AbstractionManager implementation
AbstractionManager::AbstractionManager(const std::string& preflop_path,
                                     const std::string& flop_path,
                                     const std::string& turn_path,
                                     const std::string& river_path) {
    clusters_.resize(4);
    clusters_[0] = LoadCSV(preflop_path);
    clusters_[1] = LoadCSV(flop_path);
    clusters_[2] = LoadCSV(turn_path);
    clusters_[3] = LoadCSV(river_path);
}

std::vector<int> AbstractionManager::LoadCSV(const std::string& path) {
    std::vector<int> clusters;
    std::ifstream file(path);
    std::string line;
    
    while (std::getline(file, line)) {
        clusters.push_back(std::stoi(line));
    }
    
    return clusters;
}

int AbstractionManager::GetCluster(int round, uint64_t hand_index) const {
    if (round >= clusters_.size() || hand_index >= clusters_[round].size()) {
        return 0; // Default cluster
    }
    return clusters_[round][hand_index];
}

// AbstractInfoState implementation
AbstractInfoState::AbstractInfoState(const open_spiel::State& state,
                                   const AbstractionManager& abstraction) {
    player_ = state.CurrentPlayer();
    round_ = state.GetGame()->NumPlayers(); // Placeholder - get actual round
    
    // Get base information state
    base_info_state_ = state.InformationStateString(player_);
    
    // TODO: Extract hand index from state and get cluster
    // This requires parsing the state to get the actual cards
    // and computing the hand index using the indexer
    uint64_t hand_index = 0; // Placeholder
    cluster_ = abstraction.GetCluster(round_, hand_index);
}

std::string AbstractInfoState::ToString() const {
    return base_info_state_ + "_c" + std::to_string(cluster_);
}

bool AbstractInfoState::operator==(const AbstractInfoState& other) const {
    return ToString() == other.ToString();
}

// MCCFRPSolver implementation
MCCFRPSolver::MCCFRPSolver(std::shared_ptr<const open_spiel::Game> game,
                          std::unique_ptr<AbstractionManager> abstraction,
                          const Config& config)
    : game_(game), 
      abstraction_(std::move(abstraction)), 
      config_(config),
      regret_mutexes_(1024),  // Hash buckets for thread safety
      strategy_mutexes_(1024) {
    
    // Initialize RNGs for each thread
    rngs_.reserve(config.num_threads);
    for (int i = 0; i < config.num_threads; ++i) {
        rngs_.emplace_back(std::random_device{}() + i);
    }
    
    omp_set_num_threads(config.num_threads);
}

void MCCFRPSolver::Train(int iterations) {
    for (int t = 0; t < iterations; ++t) {
        if (t % 100 == 0) {
            std::cout << "Iteration " << t << "/" << iterations << std::endl;
        }
        
        // Update strategies periodically
        if (t % config_.strategy_interval == 0) {
            #pragma omp parallel for
            for (int player = 0; player < game_->NumPlayers(); ++player) {
                int thread_id = omp_get_thread_num();
                auto state = game_->NewInitialState();
                UpdateStrategy(std::move(state), player, thread_id);
            }
        }
        
        // Main MCCFR traversal
        #pragma omp parallel for
        for (int player = 0; player < game_->NumPlayers(); ++player) {
            int thread_id = omp_get_thread_num();
            auto state = game_->NewInitialState();
            
            // Choose between pruning and regular MCCFR
            if (t > config_.prune_threshold_iterations) {
                std::uniform_real_distribution<> dis(0.0, 1.0);
                if (dis(rngs_[thread_id]) < config_.prune_probability) {
                    TraverseWithPruning(std::move(state), player, thread_id);
                } else {
                    Traverse(std::move(state), player, thread_id);
                }
            } else {
                Traverse(std::move(state), player, thread_id);
            }
        }
        
        // Apply Linear CFR discounting
        if (t < config_.lcfr_threshold && t % config_.discount_interval == 0) {
            double discount = static_cast<double>(t / config_.discount_interval) /
                            (t / config_.discount_interval + 1);
            ApplyDiscounting(discount);
        }
    }
}

double MCCFRPSolver::Traverse(std::unique_ptr<open_spiel::State> state,
                             int traversing_player,
                             int thread_id) {
    if (state->IsTerminal()) {
        return state->PlayerReturn(traversing_player);
    }
    
    if (state->IsChanceNode()) {
        // Sample chance outcome
        auto outcomes = state->ChanceOutcomes();
        std::discrete_distribution<> dis(outcomes.second.begin(), outcomes.second.end());
        int action = outcomes.first[dis(rngs_[thread_id])];
        state->ApplyAction(action);
        return Traverse(std::move(state), traversing_player, thread_id);
    }
    
    int current_player = state->CurrentPlayer();
    AbstractInfoState info_state(*state, *abstraction_);
    auto legal_actions = state->LegalActions();
    
    if (current_player == traversing_player) {
        // Get current strategy
        auto strategy = GetCurrentStrategy(info_state);
        
        // Get value for each action
        std::vector<double> action_values(legal_actions.size());
        double expected_value = 0.0;
        
        for (size_t i = 0; i < legal_actions.size(); ++i) {
            auto state_copy = state->Clone();
            state_copy->ApplyAction(legal_actions[i]);
            action_values[i] = Traverse(std::move(state_copy), 
                                      traversing_player, thread_id);
            expected_value += strategy[i] * action_values[i];
        }
        
        // Update regrets with thread safety
        int mutex_idx = GetMutexIndex(info_state);
        {
            std::lock_guard<std::mutex> lock(regret_mutexes_[mutex_idx]);
            auto& regrets = regrets_[info_state];
            if (regrets.empty()) {
                regrets.resize(legal_actions.size(), 0.0);
            }
            
            for (size_t i = 0; i < legal_actions.size(); ++i) {
                double regret = action_values[i] - expected_value;
                regrets[i] = std::max(regrets[i] + regret, config_.regret_floor);
            }
        }
        
        return expected_value;
    } else {
        // Sample opponent action
        auto strategy = GetCurrentStrategy(info_state);
        std::discrete_distribution<> dis(strategy.begin(), strategy.end());
        int action_idx = dis(rngs_[thread_id]);
        state->ApplyAction(legal_actions[action_idx]);
        return Traverse(std::move(state), traversing_player, thread_id);
    }
}

double MCCFRPSolver::TraverseWithPruning(std::unique_ptr<open_spiel::State> state,
                                        int traversing_player,
                                        int thread_id) {
    if (state->IsTerminal()) {
        return state->PlayerReturn(traversing_player);
    }
    
    if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        std::discrete_distribution<> dis(outcomes.second.begin(), outcomes.second.end());
        int action = outcomes.first[dis(rngs_[thread_id])];
        state->ApplyAction(action);
        return TraverseWithPruning(std::move(state), traversing_player, thread_id);
    }
    
    int current_player = state->CurrentPlayer();
    AbstractInfoState info_state(*state, *abstraction_);
    auto legal_actions = state->LegalActions();
    
    if (current_player == traversing_player) {
        auto strategy = GetCurrentStrategy(info_state);
        
        std::vector<double> action_values(legal_actions.size(), 0.0);
        std::vector<bool> explored(legal_actions.size(), false);
        double expected_value = 0.0;
        
        // Check regrets for pruning
        int mutex_idx = GetMutexIndex(info_state);
        std::vector<double> current_regrets;
        {
            std::lock_guard<std::mutex> lock(regret_mutexes_[mutex_idx]);
            if (regrets_.find(info_state) != regrets_.end()) {
                current_regrets = regrets_[info_state];
            } else {
                current_regrets.resize(legal_actions.size(), 0.0);
            }
        }
        
        // Traverse actions with pruning
        for (size_t i = 0; i < legal_actions.size(); ++i) {
            if (current_regrets.empty() || 
                current_regrets[i] > config_.prune_threshold) {
                auto state_copy = state->Clone();
                state_copy->ApplyAction(legal_actions[i]);
                action_values[i] = TraverseWithPruning(std::move(state_copy),
                                                      traversing_player, thread_id);
                explored[i] = true;
                expected_value += strategy[i] * action_values[i];
            }
        }
        
        // Update regrets only for explored actions
        {
            std::lock_guard<std::mutex> lock(regret_mutexes_[mutex_idx]);
            auto& regrets = regrets_[info_state];
            if (regrets.empty()) {
                regrets.resize(legal_actions.size(), 0.0);
            }
            
            for (size_t i = 0; i < legal_actions.size(); ++i) {
                if (explored[i]) {
                    double regret = action_values[i] - expected_value;
                    regrets[i] = std::max(regrets[i] + regret, config_.regret_floor);
                }
            }
        }
        
        return expected_value;
    } else {
        auto strategy = GetCurrentStrategy(info_state);
        std::discrete_distribution<> dis(strategy.begin(), strategy.end());
        int action_idx = dis(rngs_[thread_id]);
        state->ApplyAction(legal_actions[action_idx]);
        return TraverseWithPruning(std::move(state), traversing_player, thread_id);
    }
}

void MCCFRPSolver::UpdateStrategy(std::unique_ptr<open_spiel::State> state,
                                 int updating_player,
                                 int thread_id) {
    if (state->IsTerminal()) {
        return;
    }
    
    if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        std::discrete_distribution<> dis(outcomes.second.begin(), outcomes.second.end());
        int action = outcomes.first[dis(rngs_[thread_id])];
        state->ApplyAction(action);
        UpdateStrategy(std::move(state), updating_player, thread_id);
        return;
    }
    
    int current_player = state->CurrentPlayer();
    
    if (current_player == updating_player) {
        AbstractInfoState info_state(*state, *abstraction_);
        auto legal_actions = state->LegalActions();
        auto strategy = GetCurrentStrategy(info_state);
        
        // Sample action according to strategy
        std::discrete_distribution<> dis(strategy.begin(), strategy.end());
        int action_idx = dis(rngs_[thread_id]);
        
        // Update average strategy
        int mutex_idx = GetMutexIndex(info_state);
        {
            std::lock_guard<std::mutex> lock(strategy_mutexes_[mutex_idx]);
            auto& avg_sum = avg_strategy_sum_[info_state];
            if (avg_sum.empty()) {
                avg_sum.resize(legal_actions.size(), 0.0);
            }
            avg_sum[action_idx] += 1.0;
        }
        
        state->ApplyAction(legal_actions[action_idx]);
        UpdateStrategy(std::move(state), updating_player, thread_id);
    } else {
        // Traverse all opponent actions
        auto legal_actions = state->LegalActions();
        for (int action : legal_actions) {
            auto state_copy = state->Clone();
            state_copy->ApplyAction(action);
            UpdateStrategy(std::move(state_copy), updating_player, thread_id);
        }
    }
}

std::vector<double> MCCFRPSolver::GetCurrentStrategy(const AbstractInfoState& info_state) {
    int mutex_idx = GetMutexIndex(info_state);
    std::lock_guard<std::mutex> lock(regret_mutexes_[mutex_idx]);
    
    auto it = regrets_.find(info_state);
    if (it == regrets_.end()) {
        // Uniform strategy if no regrets yet
        auto state = game_->NewInitialState();
        // TODO: Navigate to this info state to get legal actions count
        int num_actions = 2; // Placeholder
        return std::vector<double>(num_actions, 1.0 / num_actions);
    }
    
    const auto& regrets = it->second;
    std::vector<double> strategy(regrets.size());
    
    // Regret matching
    double sum = 0.0;
    for (double r : regrets) {
        sum += std::max(0.0, r);
    }
    
    if (sum > 0) {
        for (size_t i = 0; i < regrets.size(); ++i) {
            strategy[i] = std::max(0.0, regrets[i]) / sum;
        }
    } else {
        // Uniform strategy
        std::fill(strategy.begin(), strategy.end(), 1.0 / strategy.size());
    }
    
    return strategy;
}

std::vector<double> MCCFRPSolver::GetAverageStrategy(const AbstractInfoState& info_state) const {
    auto it = avg_strategy_sum_.find(info_state);
    if (it == avg_strategy_sum_.end()) {
        // Return uniform strategy
        auto state = game_->NewInitialState();
        // TODO: Navigate to this info state to get legal actions count
        int num_actions = 2; // Placeholder
        return std::vector<double>(num_actions, 1.0 / num_actions);
    }
    
    const auto& sum = it->second;
    double total = std::accumulate(sum.begin(), sum.end(), 0.0);
    
    std::vector<double> avg_strategy(sum.size());
    if (total > 0) {
        for (size_t i = 0; i < sum.size(); ++i) {
            avg_strategy[i] = sum[i] / total;
        }
    } else {
        std::fill(avg_strategy.begin(), avg_strategy.end(), 1.0 / avg_strategy.size());
    }
    
    return avg_strategy;
}

int MCCFRPSolver::GetMutexIndex(const AbstractInfoState& info_state) const {
    return AbstractInfoState::Hash{}(info_state) % regret_mutexes_.size();
}

void MCCFRPSolver::ApplyDiscounting(double discount_factor) {
    // Apply discount to all regrets
    #pragma omp parallel for
    for (size_t i = 0; i < regret_mutexes_.size(); ++i) {
        std::lock_guard<std::mutex> lock(regret_mutexes_[i]);
        for (auto& [info_state, regrets] : regrets_) {
            if (GetMutexIndex(info_state) == i) {
                for (double& r : regrets) {
                    r *= discount_factor;
                }
            }
        }
    }
    
    // Apply discount to average strategies
    #pragma omp parallel for
    for (size_t i = 0; i < strategy_mutexes_.size(); ++i) {
        std::lock_guard<std::mutex> lock(strategy_mutexes_[i]);
        for (auto& [info_state, strategy_sum] : avg_strategy_sum_) {
            if (GetMutexIndex(info_state) == i) {
                for (double& s : strategy_sum) {
                    s *= discount_factor;
                }
            }
        }
    }
}

// ParallelMCCFRPTrainer implementation
ParallelMCCFRPTrainer::ParallelMCCFRPTrainer(const std::string& game_string,
                                           const std::string& preflop_path,
                                           const std::string& flop_path,
                                           const std::string& turn_path,
                                           const std::string& river_path,
                                           const MCCFRPSolver::Config& config) {
    game_ = open_spiel::LoadGame(game_string);
    auto abstraction = std::make_unique<AbstractionManager>(
        preflop_path, flop_path, turn_path, river_path);
    solver_ = std::make_unique<MCCFRPSolver>(game_, std::move(abstraction), config);
}

void ParallelMCCFRPTrainer::Train(int iterations) {
    solver_->Train(iterations);
}

void ParallelMCCFRPTrainer::SaveStrategy(const std::string& path) const {
    // TODO: Implement strategy serialization
    std::ofstream file(path, std::ios::binary);
    // Serialize solver_->avg_strategy_sum_
}

void ParallelMCCFRPTrainer::LoadStrategy(const std::string& path) {
    // TODO: Implement strategy deserialization
    std::ifstream file(path, std::ios::binary);
    // Deserialize into solver_->avg_strategy_sum_
}

} // namespace mccfr_p