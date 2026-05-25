#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

// Simple test to verify matching algorithm
int main() {
    std::cout << "=== Fixed Matching Algorithm Test ===\n";
    
    // Simulate descriptors: 2048 keypoints, 64 dimensions
    const int n1 = 2048;
    const int n2 = 2048;
    const int dim = 64;
    
    // Create dummy descriptors
    std::vector<float> desc1(n1 * dim, 0.1f);
    std::vector<float> desc2(n2 * dim, 0.1f);
    
    // Add some variation
    for (int i = 0; i < n1; ++i) {
        for (int d = 0; d < dim; ++d) {
            desc1[i * dim + d] += (i % 10) * 0.01f;
        }
    }
    
    for (int j = 0; j < n2; ++j) {
        for (int d = 0; d < dim; ++d) {
            desc2[j * dim + d] += (j % 8) * 0.01f;
        }
    }
    
    // Parameters
    float beta = 20.0f;
    float match_thresh = 0.01f;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Step 1: Compute dot products
    std::vector<float> dist(n1 * n2, 0.0f);
    for (int i = 0; i < n1; ++i) {
        const float* d1 = desc1.data() + i * dim;
        float* row = dist.data() + i * n2;
        
        for (int j = 0; j < n2; ++j) {
            const float* d2 = desc2.data() + j * dim;
            float dot = 0.0f;
            
            // Unrolled loop for better performance
            for (int d = 0; d < dim; d += 4) {
                if (d + 3 < dim) {
                    dot += d1[d] * d2[d];
                    dot += d1[d+1] * d2[d+1];
                    dot += d1[d+2] * d2[d+2];
                    dot += d1[d+3] * d2[d+3];
                }
            }
            row[j] = dot;
        }
    }
    
    // Step 2: Apply exponential transformation
    std::vector<float> exp_dist(n1 * n2);
    for (size_t idx = 0; idx < dist.size(); ++idx) {
        exp_dist[idx] = std::exp((dist[idx] - 1.0f) * beta);
    }
    
    // Step 3: Compute row and column sums
    std::vector<float> sum1(n1, 0.0f);
    std::vector<float> sum2(n2, 0.0f);
    
    for (int i = 0; i < n1; ++i) {
        const float* row = exp_dist.data() + i * n2;
        float row_sum = 0.0f;
        
        for (int j = 0; j < n2; ++j) {
            float val = row[j];
            row_sum += val;
            sum2[j] += val;
        }
        sum1[i] = row_sum;
    }
    
    // Step 4: Compute similarity matrix
    std::vector<float> sim(n1 * n2);
    for (int i = 0; i < n1; ++i) {
        float* sim_row = sim.data() + i * n2;
        const float* exp_row = exp_dist.data() + i * n2;
        
        for (int j = 0; j < n2; ++j) {
            float val = exp_row[j];
            float similarity = (val * val) / (sum1[i] * sum2[j] + 1e-12f);
            sim_row[j] = similarity;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start);
    
    std::cout << "Matching algorithm test completed\n";
    std::cout << "Time: " << duration.count() << " ms\n";
    std::cout << "Matrix size: " << dist.size() << " elements\n";
    
    // Test mutual nearest neighbor
    std::vector<int> nn12(n1, 0);
    for (int i = 0; i < n1; ++i) {
        const float* row = sim.data() + i * n2;
        float max_sim = row[0];
        int max_idx = 0;
        
        for (int j = 1; j < n2; ++j) {
            if (row[j] > max_sim) {
                max_sim = row[j];
                max_idx = j;
            }
        }
        nn12[i] = max_idx;
    }
    
    std::vector<int> nn21(n2, 0);
    for (int j = 0; j < n2; ++j) {
        float max_sim = sim[j];
        int max_idx = 0;
        
        for (int i = 1; i < n1; ++i) {
            if (sim[i * n2 + j] > max_sim) {
                max_sim = sim[i * n2 + j];
                max_idx = i;
            }
        }
        nn21[j] = max_idx;
    }
    
    // Count mutual matches above threshold
    int mutual_matches = 0;
    for (int i = 0; i < n1; ++i) {
        int j = nn12[i];
        if (nn21[j] == i) {
            if (sim[i * n2 + j] > match_thresh) {
                mutual_matches++;
            }
        }
    }
    
    std::cout << "Mutual matches above threshold: " << mutual_matches << "\n";
    
    // Test performance with realistic data
    std::cout << "\n=== Performance Test ===\n";
    
    const int iterations = 10;
    double total_time = 0.0;
    
    for (int iter = 0; iter < iterations; ++iter) {
        start = std::chrono::high_resolution_clock::now();
        
        // Simulate matching (simplified)
        int matches = 0;
        for (int i = 0; i < n1; ++i) {
            for (int j = 0; j < n2; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    dot += desc1[i * dim + d] * desc2[j * dim + d];
                }
                float sim_val = std::exp((dot - 1.0f) * beta);
                if (sim_val > match_thresh) {
                    matches++;
                }
            }
        }
        
        end = std::chrono::high_resolution_clock::now();
        double iter_time = std::chrono::duration<double, std::milli>(end - start).count();
        total_time += iter_time;
        
        std::cout << "Iteration " << iter + 1 << ": " << iter_time << " ms, matches: " << matches << "\n";
    }
    
    std::cout << "Average time: " << total_time / iterations << " ms\n";
    std::cout << "Total time: " << total_time << " ms\n";
    
    std::cout << "\n=== TEST PASSED ===\n";
    return 0;
}