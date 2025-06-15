#ifndef POKER_HAND_CLUSTERING_PREFLOP_CLUSTERING_H_
#define POKER_HAND_CLUSTERING_PREFLOP_CLUSTERING_H_

#include <vector>
#include <cstdint>

namespace poker {
std::vector<uint8_t> calculate_preflop_clustering(size_t num_clusters = 8);
}

#endif  // POKER_HAND_CLUSTERING_PREFLOP_CLUSTERING_H_