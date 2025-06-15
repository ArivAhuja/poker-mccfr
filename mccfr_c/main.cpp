// main.cpp - Example usage
#include "mccfr_p.h"
#include <iostream>

int main(int argc, char** argv) {
    // Configuration
    mccfr_p::MCCFRPSolver::Config config;
    config.num_threads = 32;  // Adjust based on supercomputer
    config.prune_threshold = -10000.0;  // Smaller for limit poker
    config.strategy_interval = 100;
    config.discount_interval = 100;
    
    // Game string for limit hold'em heads-up
    std::string game_string = "universal_poker("
        "betting=limit,"
        "numPlayers=2,"
        "numRounds=4,"
        "blind=10 5,"
        "firstPlayer=2 1 1 1,"
        "numSuits=4,"
        "numRanks=13,"
        "numHoleCards=2,"
        "numBoardCards=0 3 1 1,"
        "stack=1000,"
        "bettingAbstraction=fcpa)";
    
    // Create trainer
    mccfr_p::ParallelMCCFRPTrainer trainer(
        game_string,
        "abstraction_c/data/preflop_cluster.csv",
        "abstraction_c/data/flop_kmeans.csv", 
        "abstraction_c/data/turn_kmeans.csv",
        "abstraction_c/data/river_kmeans.csv",
        config
    );
    
    // Train
    int iterations = 1000000;
    std::cout << "Starting MCCFR-P training with " << config.num_threads 
              << " threads for " << iterations << " iterations..." << std::endl;
    
    trainer.Train(iterations);
    
    // Save strategy
    trainer.SaveStrategy("limit_holdem_strategy.bin");
    
    return 0;
}