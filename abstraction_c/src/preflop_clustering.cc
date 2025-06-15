#include "preflop_clustering.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>
#include <map>
#include "common.h"
#include "equity.h"
#include "mean_histogram.h"

namespace poker {

// Compute equity for a preflop hand against all other preflop hands
std::vector<double> compute_preflop_equity_vector(uint8_t card1, uint8_t card2,
                                                 const std::vector<int32_t>& equity_vec) {
  hand_indexer_t preflop_indexer, river_indexer;
  hand_indexer_init(1, (const uint8_t[]){2}, &preflop_indexer);
  hand_indexer_init(2, (const uint8_t[]){2, 5}, &river_indexer);
  
  std::vector<double> equity_against_hands(169, 0.0);
  std::vector<size_t> matchup_counts(169, 0);
  
  uint8_t hero_cards[2] = {card1, card2};
  
  // Iterate through all possible opponent preflop hands
  for (uint8_t opp_card1 = 0; opp_card1 < 52; opp_card1++) {
    if (opp_card1 == card1 || opp_card1 == card2) continue;
    
    for (uint8_t opp_card2 = opp_card1 + 1; opp_card2 < 52; opp_card2++) {
      if (opp_card2 == card1 || opp_card2 == card2) continue;
      
      uint8_t opp_cards[2] = {opp_card1, opp_card2};
      
      // Get canonical index for opponent's preflop hand
      auto opp_preflop_idx = hand_index_last(&preflop_indexer, opp_cards);
      
      // Sum equities for all possible board runouts
      double total_equity = 0.0;
      size_t board_count = 0;
      
      // Iterate through all possible 5-card boards
      for (uint8_t b1 = 0; b1 < 52; b1++) {
        if (b1 == card1 || b1 == card2 || b1 == opp_card1 || b1 == opp_card2) continue;
        
        for (uint8_t b2 = b1 + 1; b2 < 52; b2++) {
          if (b2 == card1 || b2 == card2 || b2 == opp_card1 || b2 == opp_card2) continue;
          
          for (uint8_t b3 = b2 + 1; b3 < 52; b3++) {
            if (b3 == card1 || b3 == card2 || b3 == opp_card1 || b3 == opp_card2) continue;
            
            for (uint8_t b4 = b3 + 1; b4 < 52; b4++) {
              if (b4 == card1 || b4 == card2 || b4 == opp_card1 || b4 == opp_card2) continue;
              
              for (uint8_t b5 = b4 + 1; b5 < 52; b5++) {
                if (b5 == card1 || b5 == card2 || b5 == opp_card1 || b5 == opp_card2) continue;
                
                uint8_t hero_river[7] = {card1, card2, b1, b2, b3, b4, b5};
                uint8_t opp_river[7] = {opp_card1, opp_card2, b1, b2, b3, b4, b5};
                
                auto hero_idx = hand_index_last(&river_indexer, hero_river);
                auto opp_idx = hand_index_last(&river_indexer, opp_river);
                
                int32_t hero_equity = equity_vec[hero_idx];
                int32_t opp_equity = equity_vec[opp_idx];
                
                if (hero_equity > opp_equity) {
                  total_equity += 1.0;
                } else if (hero_equity == opp_equity) {
                  total_equity += 0.5;
                }
                board_count++;
              }
            }
          }
        }
      }
      
      equity_against_hands[opp_preflop_idx] += total_equity / board_count;
      matchup_counts[opp_preflop_idx]++;
    }
  }
  
  // Average the equities
  for (size_t i = 0; i < 169; i++) {
    if (matchup_counts[i] > 0) {
      equity_against_hands[i] /= matchup_counts[i];
    }
  }
  
  hand_indexer_free(&preflop_indexer);
  hand_indexer_free(&river_indexer);
  
  return equity_against_hands;
}

std::vector<uint8_t> calculate_preflop_clustering(size_t num_clusters) {
  // Get precomputed equity values
  std::vector<int32_t> equity_vec = calculate_equity();
  
  hand_indexer_t preflop_indexer;
  hand_indexer_init(1, (const uint8_t[]){2}, &preflop_indexer);
  
  const size_t num_hands = 169;  // Number of canonical preflop hands
  
  // Compute equity vectors for all canonical preflop hands
  std::vector<std::vector<double>> equity_vectors(num_hands);
  
  for (size_t i = 0; i < num_hands; i++) {
    uint8_t cards[2];
    hand_unindex(&preflop_indexer, 0, i, cards);
    equity_vectors[i] = compute_preflop_equity_vector(cards[0], cards[1], equity_vec);
  }
  
  // Initialize clustering with k-means++
  std::vector<uint8_t> clustering(num_hands);
  std::vector<size_t> centers;
  std::mt19937 mt{std::random_device{}()};
  
  // Choose first center randomly
  centers.push_back(std::uniform_int_distribution<size_t>(0, num_hands - 1)(mt));
  
  for (size_t k = 1; k < num_clusters; k++) {
    std::vector<double> min_distances(num_hands, std::numeric_limits<double>::max());
    
    // Compute distances to nearest center
    for (size_t i = 0; i < num_hands; i++) {
      for (size_t c : centers) {
        double dist = 0.0;
        for (size_t j = 0; j < num_hands; j++) {
          double diff = equity_vectors[i][j] - equity_vectors[c][j];
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
    size_t next_center = std::lower_bound(cumulative.begin(), cumulative.end(), r) - cumulative.begin();
    centers.push_back(next_center);
  }
  
  // Run k-means iterations
  const size_t max_iterations = 100;
  for (size_t iter = 0; iter < max_iterations; iter++) {
    // Assignment step
    size_t changes = 0;
    for (size_t i = 0; i < num_hands; i++) {
      double min_dist = std::numeric_limits<double>::max();
      uint8_t best_cluster = 0;
      
      for (size_t k = 0; k < num_clusters; k++) {
        double dist = 0.0;
        for (size_t j = 0; j < num_hands; j++) {
          double diff = equity_vectors[i][j] - equity_vectors[centers[k]][j];
          dist += diff * diff;
        }
        
        if (dist < min_dist) {
          min_dist = dist;
          best_cluster = k;
        }
      }
      
      if (clustering[i] != best_cluster) {
        clustering[i] = best_cluster;
        changes++;
      }
    }
    
    if (changes == 0) break;
    
    // Update step - find new centers
    for (size_t k = 0; k < num_clusters; k++) {
      std::vector<double> centroid(num_hands, 0.0);
      size_t count = 0;
      
      for (size_t i = 0; i < num_hands; i++) {
        if (clustering[i] == k) {
          for (size_t j = 0; j < num_hands; j++) {
            centroid[j] += equity_vectors[i][j];
          }
          count++;
        }
      }
      
      if (count > 0) {
        for (size_t j = 0; j < num_hands; j++) {
          centroid[j] /= count;
        }
        
        // Find hand closest to centroid
        double min_dist = std::numeric_limits<double>::max();
        size_t best_hand = centers[k];
        
        for (size_t i = 0; i < num_hands; i++) {
          double dist = 0.0;
          for (size_t j = 0; j < num_hands; j++) {
            double diff = equity_vectors[i][j] - centroid[j];
            dist += diff * diff;
          }
          
          if (dist < min_dist) {
            min_dist = dist;
            best_hand = i;
          }
        }
        
        centers[k] = best_hand;
      }
    }
  }
  
  hand_indexer_free(&preflop_indexer);
  return clustering;
}

}  // namespace poker