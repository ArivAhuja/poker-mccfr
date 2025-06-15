#ifndef POKER_HAND_CLUSTERING_TURN_CLUSTER_DISTANCE_H_
#define POKER_HAND_CLUSTERING_TURN_CLUSTER_DISTANCE_H_

#include <vector>
#include <cstdint>

namespace poker {
std::vector<std::vector<double>> calc_turn_cluster_distance(  // Add "calc_"
    const std::vector<std::vector<size_t>>& turn_histograms,
    const std::vector<size_t>& turn_clustering);  // Change uint8_t to size_t
}

#endif  // POKER_HAND_CLUSTERING_TURN_CLUSTER_DISTANCE_H_