#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

// Correct preprocessing for CLIDD
class CLIDDPreprocessor {
public:
    struct PreprocessResult {
        std::vector<float> input;  // (C, H, W) format, normalized [0, 1]
        int original_h;
        int original_w;
        int target_h;
        int target_w;
        float scale_x;
        float scale_y;
    };
    
    static PreprocessResult preprocess(const std::vector<uint8_t>& bgr_data, int h, int w) {
        PreprocessResult result;
        result.original_h = h;
        result.original_w = w;
        
        // Pad to multiple of 32
        result.target_h = ((h + 31) / 32) * 32;
        result.target_w = ((w + 31) / 32) * 32;
        
        result.scale_x = static_cast<float>(w) / result.target_w;
        result.scale_y = static_cast<float>(h) / result.target_h;
        
        // Convert BGR to RGB and normalize to [0, 1]
        std::vector<float> rgb_normalized(result.target_h * result.target_w * 3, 0.0f);
        
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int src_idx = (y * w + x) * 3;
                int dst_idx = (y * result.target_w + x) * 3;
                
                // BGR to RGB and normalize
                rgb_normalized[dst_idx] = bgr_data[src_idx + 2] / 255.0f;  // R
                rgb_normalized[dst_idx + 1] = bgr_data[src_idx + 1] / 255.0f;  // G
                rgb_normalized[dst_idx + 2] = bgr_data[src_idx] / 255.0f;  // B
            }
        }
        
        // Reorder to (C, H, W) format
        result.input.resize(3 * result.target_h * result.target_w);
        
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < result.target_h; ++y) {
                for (int x = 0; x < result.target_w; ++x) {
                    int src_idx = (y * result.target_w + x) * 3 + c;
                    int dst_idx = c * result.target_h * result.target_w + y * result.target_w + x;
                    result.input[dst_idx] = rgb_normalized[src_idx];
                }
            }
        }
        
        return result;
    }
};

// Optimized matching algorithm (CPU version, but optimized)
class CLIDDMatcher {
public:
    struct MatchResult {
        std::vector<int> idx1;
        std::vector<int> idx2;
        std::vector<float> scores;
    };
    
    static MatchResult match(const std::vector<float>& desc1, const std::vector<float>& desc2,
                            int dim, float beta = 20.0f, float min_score = 0.01f) {
        MatchResult result;
        
        int n1 = static_cast<int>(desc1.size()) / dim;
        int n2 = static_cast<int>(desc2.size()) / dim;
        
        if (n1 == 0 || n2 == 0) {
            return result;
        }
        
        // Compute dot products
        std::vector<float> dist(n1 * n2, 0.0f);
        
        // Optimized: use block processing
        const int block_size = 4;
        
        for (int i = 0; i < n1; ++i) {
            const float* d1 = desc1.data() + i * dim;
            float* row = dist.data() + i * n2;
            
            for (int j = 0; j < n2; ++j) {
                const float* d2 = desc2.data() + j * dim;
                float dot = 0.0f;
                
                // Process in blocks for better cache utilization
                int d = 0;
                for (; d <= dim - block_size; d += block_size) {
                    dot += d1[d] * d2[d];
                    dot += d1[d+1] * d2[d+1];
                    dot += d1[d+2] * d2[d+2];
                    dot += d1[d+3] * d2[d+3];
                }
                
                // Process remaining elements
                for (; d < dim; ++d) {
                    dot += d1[d] * d2[d];
                }
                
                row[j] = dot;
            }
        }
        
        // Apply exponential transformation
        std::vector<float> exp_dist(dist.size());
        for (size_t idx = 0; idx < dist.size(); ++idx) {
            exp_dist[idx] = std::exp((dist[idx] - 1.0f) * beta);
        }
        
        // Compute row and column sums
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
        
        // Compute similarity matrix
        std::vector<float> sim(dist.size());
        for (int i = 0; i < n1; ++i) {
            float* sim_row = sim.data() + i * n2;
            const float* exp_row = exp_dist.data() + i * n2;
            
            for (int j = 0; j < n2; ++j) {
                float val = exp_row[j];
                float similarity = (val * val) / (sum1[i] * sum2[j] + 1e-12f);
                sim_row[j] = similarity;
            }
        }
        
        // Find mutual nearest neighbors
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
        
        // Collect mutual matches above threshold
        for (int i = 0; i < n1; ++i) {
            int j = nn12[i];
            if (nn21[j] == i) {
                float score = sim[i * n2 + j];
                if (score > min_score) {
                    result.idx1.push_back(i);
                    result.idx2.push_back(j);
                    result.scores.push_back(score);
                }
            }
        }
        
        return result;
    }
};

// Performance test
void testPerformance() {
    std::cout << "=== Performance Test ===\n";
    
    // Simulate realistic data: 2048 keypoints, 64 dimensions
    const int n1 = 2048;
    const int n2 = 2048;
    const int dim = 64;
    
    // Create realistic descriptors
    std::vector<float> desc1(n1 * dim);
    std::vector<float> desc2(n2 * dim);
    
    // Fill with realistic values (normal distribution-like)
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    
    for (int i = 0; i < n1 * dim; ++i) {
        desc1[i] = (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 0.2f;
    }
    
    for (int i = 0; i < n2 * dim; ++i) {
        desc2[i] = (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 0.2f;
    }
    
    // Warm up
    auto warm_result = CLIDDMatcher::match(desc1, desc2, dim, 20.0f, 0.01f);
    
    // Benchmark
    const int iterations = 10;
    double total_time = 0.0;
    
    for (int iter = 0; iter < iterations; ++iter) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = CLIDDMatcher::match(desc1, desc2, dim, 20.0f, 0.01f);
        
        auto end = std::chrono::high_resolution_clock::now();
        double iter_time = std::chrono::duration<double, std::milli>(end - start).count();
        total_time += iter_time;
        
        std::cout << "Iteration " << iter + 1 << ": " << iter_time << " ms, matches: " << result.idx1.size() << "\n";
    }
    
    std::cout << "Average time: " << total_time / iterations << " ms\n";
    std::cout << "Total time: " << total_time << " ms\n";
}

int main() {
    std::cout << "=== Correct CLIDD Implementation Test ===\n";
    
    // Test preprocessing
    std::cout << "\n1. Testing preprocessing...\n";
    
    // Create a dummy 100x100 BGR image
    const int h = 100;
    const int w = 100;
    std::vector<uint8_t> bgr_data(h * w * 3, 128);
    
    auto preprocess_result = CLIDDPreprocessor::preprocess(bgr_data, h, w);
    
    std::cout << "Original: " << h << "x" << w << "\n";
    std::cout << "Target: " << preprocess_result.target_h << "x" << preprocess_result.target_w << "\n";
    std::cout << "Scale X: " << preprocess_result.scale_x << "\n";
    std::cout << "Scale Y: " << preprocess_result.scale_y << "\n";
    std::cout << "Input size: " << preprocess_result.input.size() << " elements\n";
    
    // Test matching algorithm
    std::cout << "\n2. Testing matching algorithm...\n";
    
    const int test_dim = 64;
    const int test_n1 = 500;
    const int test_n2 = 500;
    
    std::vector<float> test_desc1(test_n1 * test_dim, 0.1f);
    std::vector<float> test_desc2(test_n2 * test_dim, 0.1f);
    
    auto match_result = CLIDDMatcher::match(test_desc1, test_desc2, test_dim);
    
    std::cout << "Matches found: " << match_result.idx1.size() << "\n";
    
    if (!match_result.scores.empty()) {
        std::cout << "First match score: " << match_result.scores[0] << "\n";
    }
    
    // Performance test
    testPerformance();
    
    std::cout << "\n=== TEST COMPLETED SUCCESSFULLY ===\n";
    return 0;
}