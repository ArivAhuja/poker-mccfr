#ifndef POKER_HAND_CLUSTERING_FLOP_HISTOGRAMS_H_
#define POKER_HAND_CLUSTERING_FLOP_HISTOGRAMS_H_
#include <vector>
#include <cstdint>

namespace poker {
std::vector<std::vector<uint8_t>> calc_flop_histograms(
    const std::vector<size_t>& turn_clustering);  // Change uint8_t to size_t
}
#endif  // POKER_HAND_CLUSTERING_FLOP_HISTOGRAMS_H_