#include "river_clustering.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>
#include "common.h"
#include "equity.h"
#include "poker_hand.h"

namespace poker {

// Compute 8-dimensional feature vector for river hand
std::vector<double> compute_river_features(
    const uint8_t cards[7],
    const std::vector<uint8_t>& preflop_clustering,
    const std::vector<int32_t>& equity_vec) {
  
  const size_t num_clusters = 8;
  std::vector<double> features(num_clusters, 0.0);
  std::vector<size_t> cluster_counts(num_clusters, 0);
  
  hand_indexer_t preflop_indexer, river_indexer;
  hand_indexer_init(1, (const uint8_t[]){2}, &preflop_indexer);
  hand_indexer_init(2, (const uint8_t[]){2, 5}, &river_indexer);
  
  uint8_t hero_cards[2] = {cards[0], cards[1]};
  uint8_t board[5] = {cards[2], cards[3], cards[4], cards[5], cards[6]};
  
  // Get hero's river hand strength
  auto hero_idx = hand_index_last(&river_indexer, cards);
  int32_t hero_equity = equity_vec[hero_idx];
  
  // Check against all possible opponent hands
  for (uint8_t opp_c1 = 0; opp_c1 < 52; opp_c1++) {
    // Skip if card is already used
    bool used = false;
    for (int i = 0; i < 7; i++) {
      if (cards[i] == opp_c1) {
        used = true;
        break;
      }
    }
    if (used) continue;
    
    for (uint8_t opp_c2 = opp_c1 + 1; opp_c2 < 52; opp_c2++) {
      // Skip if card is already used
      bool used2 = false;
      for (int i = 0; i < 7; i++) {
        if (cards[i] == opp_c2) {
          used2 = true;
          break;
        }
      }
      if (used2) continue;
      
      // Get opponent's preflop cluster
      uint8_t opp_preflop[2] = {opp_c1, opp_c2};
      auto opp_preflop_idx = hand_index_last(&preflop_indexer, opp_preflop);
      uint8_t opp_cluster = preflop_clustering[opp_preflop_idx];
      
      // Get opponent's river hand
      uint8_t opp_river[7] = {opp_c1, opp_c2, board[0], board[1], board[2], board[3], board[4]};
      auto opp_idx = hand_index_last(&river_indexer, opp_river);
      int32_t opp_equity = equity_vec[opp_idx];
      
      // Update features
      cluster_counts[opp_cluster]++;
      if (hero_equity > opp_equity) {
        features[opp_cluster] += 1.0;
      } else if (hero_equity == opp_equity) {
        features[opp_cluster] += 0.5;
      }
    }
  }
  
  // Normalize by cluster counts
  for (size_t i = 0; i < num_clusters; i++) {
    if (cluster_counts[i] > 0) {
      features[i] /= cluster_counts[i];
    } else {
      features[i] = 0.5;  // Default to 50% if no hands from this cluster
    }
  }
  
  hand_indexer_free(&preflop_indexer);
  hand_indexer_free(&river_indexer);
  
  return features;
}

std::vector<uint16_t> calculate_river_clustering(
    const std::vector<uint8_t>& preflop_clustering,
    size_t num_river_clusters) {
  
  // Get precomputed equity values
  std::vector<int32_t> equity_vec = calculate_equity();
  
  hand_indexer_t river_indexer;
  hand_indexer_init(2, (const uint8_t[]){2, 5}, &river_indexer);
  const size_t river_size = hand_indexer_size(&river_indexer, 1);
  
  // Compute feature vectors for all river hands
  std::vector<std::vector<double>> feature_vectors;
  std::vector<size_t> hand_indices;
  
  for (size_t idx = 0; idx < river_size; idx++) {
    uint8_t cards[7];
    hand_unindex(&river_indexer, 1, idx, cards);
    
    auto features = compute_river_features(cards, preflop_clustering, equity_vec);
    feature_vectors.push_back(features);
    hand_indices.push_back(idx);
  }
  
  const size_t num_hands = feature_vectors.size();
  const size_t feature_dim = 8;
  
  // Initialize clustering with k-means++
  std::vector<uint16_t> clustering(river_size);
  std::vector<std::vector<double>> centers(num_river_clusters, std::vector<double>(feature_dim));
  std::mt19937 mt{std::random_device{}()};
  
  // Choose first center randomly
  size_t first_idx = std::uniform_int_distribution<size_t>(0, num_hands - 1)(mt);
  centers[0] = feature_vectors[first_idx];
  
  // k-means++ initialization
  for (size_t k = 1; k < num_river_clusters; k++) {
    std::vector<double> min_distances(num_hands, std::numeric_limits<double>::max());
    
    // Compute distances to nearest center
    for (size_t i = 0; i < num_hands; i++) {
      for (size_t c = 0; c < k; c++) {
        double dist = 0.0;
        for (size_t d = 0; d < feature_dim; d++) {
          double diff = feature_vectors[i][d] - centers[c][d];
          dist += diff * diff;
        }
        min_distances[i] = std::min(min_distances[i], dist);
      }
    }
    
    // Choose next center with probability proportional to squared distance
    std::vector<double> cumulative(num_hands);
    cumulative[0] = min_distances[0] * min_distances[0];
    for (size_t i = 1; i < num_hands; i++) {
      cumulative[i] = cumulative[i-1] + min_distances[i] * min_distances[i];
    }
    
    double r = std::uniform_real_distribution<>(0, cumulative.back())(mt);
    size_t next_idx = std::lower_bound(cumulative.begin(), cumulative.end(), r) - cumulative.begin();
    centers[k] = feature_vectors[next_idx];
  }
  
  // Run k-means iterations
  const size_t max_iterations = 100;
  for (size_t iter = 0; iter < max_iterations; iter++) {
    // Assignment step
    size_t changes = 0;
    
    for (size_t i = 0; i < num_hands; i++) {
      double min_dist = std::numeric_limits<double>::max();
      uint16_t best_cluster = 0;
      
      for (size_t k = 0; k < num_river_clusters; k++) {
        double dist = 0.0;
        for (size_t d = 0; d < feature_dim; d++) {
          double diff = feature_vectors[i][d] - centers[k][d];
          dist += diff * diff;
        }
        
        if (dist < min_dist) {
          min_dist = dist;
          best_cluster = k;
        }
      }
      
      size_t river_idx = hand_indices[i];
      if (clustering[river_idx] != best_cluster) {
        clustering[river_idx] = best_cluster;
        changes++;
      }
    }
    
    if (changes == 0) break;
    
    // Update step - compute new centers
    for (size_t k = 0; k < num_river_clusters; k++) {
      std::fill(centers[k].begin(), centers[k].end(), 0.0);
      size_t count = 0;
      
      for (size_t i = 0; i < num_hands; i++) {
        size_t river_idx = hand_indices[i];
        if (clustering[river_idx] == k) {
          for (size_t d = 0; d < feature_dim; d++) {
            centers[k][d] += feature_vectors[i][d];
          }
          count++;
        }
      }
      
      if (count > 0) {
        for (size_t d = 0; d < feature_dim; d++) {
          centers[k][d] /= count;
        }
      }
    }
  }
  
  hand_indexer_free(&river_indexer);
  return clustering;
}

}  // namespace poker