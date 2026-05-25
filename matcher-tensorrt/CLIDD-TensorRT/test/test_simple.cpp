#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

// Simple test to verify OpenCV works
int main() {
    std::cout << "=== Simple OpenCV Test ===\n";
    
    // Test image loading
    cv::Mat img = cv::imread("assets/ref.png");
    if (img.empty()) {
        std::cerr << "ERROR: Failed to load image\n";
        return 1;
    }
    
    std::cout << "Image loaded: " << img.cols << "x" << img.rows 
              << ", channels: " << img.channels() << "\n";
    
    // Test BGR to RGB conversion
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    
    std::cout << "BGR->RGB conversion successful\n";
    
    // Test normalization
    cv::Mat normalized;
    rgb.convertTo(normalized, CV_32F, 1.0 / 255.0);
    
    std::cout << "Normalization successful\n";
    std::cout << "Min value: " << cv::min(normalized) << "\n";
    std::cout << "Max value: " << cv::max(normalized) << "\n";
    
    // Test padding to multiple of 32
    int targetH = ((img.rows + 31) / 32) * 32;
    int targetW = ((img.cols + 31) / 32) * 32;
    
    cv::Mat padded;
    cv::copyMakeBorder(normalized, padded, 
                       0, targetH - img.rows,
                       0, targetW - img.cols,
                       cv::BORDER_CONSTANT, cv::Scalar(0));
    
    std::cout << "Padding successful: " << padded.cols << "x" << padded.rows << "\n";
    
    // Reorder to (C, H, W) format
    std::vector<cv::Mat> channels;
    cv::split(padded, channels);
    
    std::vector<float> input(3 * targetH * targetW);
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < targetH; ++h) {
            for (int w = 0; w < targetW; ++w) {
                input[c * targetH * targetW + h * targetW + w] = 
                    channels[c].at<float>(h, w);
            }
        }
    }
    
    std::cout << "Input reordering successful\n";
    std::cout << "Input size: " << input.size() << " elements\n";
    
    // Test matching algorithm (simplified)
    int n1 = 2048, n2 = 2048, dim = 64;
    std::vector<float> desc1(n1 * dim, 0.1f);
    std::vector<float> desc2(n2 * dim, 0.1f);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Simple dot product
    std::vector<float> dist(n1 * n2);
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < dim; ++d) {
                dot += desc1[i * dim + d] * desc2[j * dim + d];
            }
            dist[i * n2 + j] = dot;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start);
    
    std::cout << "Matching test: " << duration.count() << " ms\n";
    std::cout << "Dist matrix size: " << dist.size() << " elements\n";
    
    std::cout << "\n=== TEST PASSED ===\n";
    return 0;
}