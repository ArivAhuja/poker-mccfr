#ifndef POKER_HAND_CLUSTERING_RIVER_CLUSTERING_H_
#define POKER_HAND_CLUSTERING_RIVER_CLUSTERING_H_

#include <vector>
#include <cstdint>

namespace poker {
std::vector<uint16_t> calculate_river_clustering(
    const std::vector<uint8_t>& preflop_clustering,
    size_t num_river_clusters = 200);
}

#endif  // POKER_HAND_CLUSTERING_RIVER_CLUSTERING_H_