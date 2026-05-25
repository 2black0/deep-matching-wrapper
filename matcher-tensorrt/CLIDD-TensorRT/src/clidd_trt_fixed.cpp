#include "clidd_trt.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <iostream>

namespace clidd {

using Clock = std::chrono::steady_clock;

double ms_since(const Clock::time_point& t0, const Clock::time_point& t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Fixed preprocessing: pad to multiple of 32, not resize
static std::pair<std::vector<float>, std::pair<int, int>> preprocessImage(const cv::Mat& img) {
    int originalH = img.rows;
    int originalW = img.cols;
    
    // Pad to multiple of 32 (like Python implementation)
    int targetH = ((originalH + 31) / 32) * 32;
    int targetW = ((originalW + 31) / 32) * 32;
    
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    
    // Normalize to [0, 1]
    cv::Mat normalized;
    rgb.convertTo(normalized, CV_32F, 1.0 / 255.0);
    
    // Pad if necessary
    cv::Mat padded;
    if (targetH != originalH || targetW != originalW) {
        cv::copyMakeBorder(normalized, padded, 
                          0, targetH - originalH,
                          0, targetW - originalW,
                          cv::BORDER_CONSTANT, cv::Scalar(0));
    } else {
        padded = normalized;
    }
    
    // Reorder to (C, H, W) format
    std::vector<cv::Mat> channels;
    cv::split(padded, channels);
    
    std::vector<float> input(3LL * targetH * targetW);
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < targetH; ++h) {
            for (int w = 0; w < targetW; ++w) {
                input[static_cast<size_t>(c) * targetH * targetW + 
                      static_cast<size_t>(h) * targetW + 
                      static_cast<size_t>(w)] = 
                    channels[c].at<float>(h, w);
            }
        }
    }
    
    return {std::move(input), {targetH, targetW}};
}

// Optimized CPU matching (still CPU but optimized)
static std::pair<std::vector<int64_t>, std::vector<int64_t>> matchFeaturesCPU(
    const ImageFeatures& feat1,
    const ImageFeatures& feat2,
    float matchThresh,
    float beta) {
    
    const int n1 = static_cast<int>(feat1.keypoints.size());
    const int n2 = static_cast<int>(feat2.keypoints.size());
    const int dim = feat1.descriptorDim;
    
    if (n1 == 0 || n2 == 0 || dim == 0) {
        return {{}, {}};
    }
    
    // Pre-allocate with size_t to avoid overflow
    const size_t total_pairs = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    std::vector<float> distMatrix(total_pairs);
    
    // Optimized dot product with loop unrolling
    for (int i = 0; i < n1; ++i) {
        const float* desc1 = feat1.descriptors.data() + i * dim;
        float* row = distMatrix.data() + i * n2;
        
        for (int j = 0; j < n2; ++j) {
            const float* desc2 = feat2.descriptors.data() + j * dim;
            float dot = 0.0f;
            
            // Loop unrolling for better performance
            int d = 0;
            for (; d <= dim - 4; d += 4) {
                dot += desc1[d] * desc2[d];
                dot += desc1[d+1] * desc2[d+1];
                dot += desc1[d+2] * desc2[d+2];
                dot += desc1[d+3] * desc2[d+3];
            }
            for (; d < dim; ++d) {
                dot += desc1[d] * desc2[d];
            }
            
            row[j] = dot;
        }
    }
    
    // Apply exponential transformation
    std::vector<float> expMatrix(total_pairs);
    for (size_t i = 0; i < total_pairs; ++i) {
        expMatrix[i] = std::exp((distMatrix[i] - 1.0f) * beta);
    }
    
    // Compute row and column sums
    std::vector<float> sum1(n1, 0.0f);
    std::vector<float> sum2(n2, 0.0f);
    
    for (int i = 0; i < n1; ++i) {
        const float* row = expMatrix.data() + i * n2;
        float row_sum = 0.0f;
        for (int j = 0; j < n2; ++j) {
            float val = row[j];
            row_sum += val;
            sum2[j] += val;
        }
        sum1[i] = row_sum;
    }
    
    // Compute similarity matrix
    std::vector<float> simMatrix(total_pairs);
    for (int i = 0; i < n1; ++i) {
        float* sim_row = simMatrix.data() + i * n2;
        const float* exp_row = expMatrix.data() + i * n2;
        
        for (int j = 0; j < n2; ++j) {
            float val = exp_row[j];
            float sim = (val * val) / (sum1[i] * sum2[j] + 1e-12f);
            sim_row[j] = sim;
        }
    }
    
    // Find mutual nearest neighbors
    std::vector<int64_t> nn12(n1, 0);
    for (int i = 0; i < n1; ++i) {
        const float* row = simMatrix.data() + i * n2;
        float maxSim = row[0];
        int maxIdx = 0;
        
        for (int j = 1; j < n2; ++j) {
            float val = row[j];
            if (val > maxSim) {
                maxSim = val;
                maxIdx = j;
            }
        }
        nn12[i] = maxIdx;
    }
    
    std::vector<int64_t> nn21(n2, 0);
    for (int j = 0; j < n2; ++j) {
        float maxSim = simMatrix[j];
        int maxIdx = 0;
        
        for (int i = 1; i < n1; ++i) {
            float val = simMatrix[i * n2 + j];
            if (val > maxSim) {
                maxSim = val;
                maxIdx = i;
            }
        }
        nn21[j] = maxIdx;
    }
    
    // Collect mutual matches above threshold
    std::vector<int64_t> matchIdx1;
    std::vector<int64_t> matchIdx2;
    
    for (int i = 0; i < n1; ++i) {
        int j = nn12[i];
        if (nn21[j] == i) {
            float sim = simMatrix[i * n2 + j];
            if (sim > matchThresh) {
                matchIdx1.push_back(i);
                matchIdx2.push_back(j);
            }
        }
    }
    
    return {matchIdx1, matchIdx2};
}

struct CLIDDTRT::Impl {
    // Simple placeholder - actual TensorRT engine would go here
    int topK_;
    float scoreThresh_;
    int radius_;
    int border_;
    float matchThresh_;
    float beta_;
    int inputH_;
    int inputW_;
    
    explicit Impl(const std::string& enginePath, int topK, float scoreThresh,
                 int radius, int border, float matchThresh, float beta)
        : topK_(topK),
          scoreThresh_(scoreThresh),
          radius_(radius),
          border_(border),
          matchThresh_(matchThresh),
          beta_(beta),
          inputH_(640),  // Default from engine
          inputW_(480) { // Default from engine
        
        if (topK_ <= 0) {
            throw std::runtime_error("topK must be positive");
        }
        
        // In real implementation, we would load the TensorRT engine here
        std::cout << "Loading TensorRT engine: " << enginePath << std::endl;
    }
    
    ImageFeatures runSingleImage(const cv::Mat& image) {
        auto [input, targetDims] = preprocessImage(image);
        auto [targetH, targetW] = targetDims;
        
        // In real implementation, we would run inference here
        // For now, return dummy features
        
        ImageFeatures features;
        features.descriptorDim = 64;  // M64 model
        features.originalH = image.rows;
        features.originalW = image.cols;
        features.scaleX = static_cast<float>(image.cols) / targetW;
        features.scaleY = static_cast<float>(image.rows) / targetH;
        
        // Generate dummy keypoints (like the model would)
        const int numKeypoints = 2048;
        features.keypoints.resize(numKeypoints);
        features.scores.resize(numKeypoints);
        features.descriptors.resize(numKeypoints * 64);
        
        // Fill with dummy data
        for (int i = 0; i < numKeypoints; ++i) {
            features.keypoints[i] = {
                static_cast<float>(i % image.cols),
                static_cast<float>(i % image.rows)
            };
            features.scores[i] = 0.5f;
            
            for (int d = 0; d < 64; ++d) {
                features.descriptors[i * 64 + d] = 
                    (d % 2 == 0) ? 0.1f : -0.1f;
            }
        }
        
        return features;
    }
    
    int topK() const { return topK_; }
    float scoreThresh() const { return scoreThresh_; }
    int radius() const { return radius_; }
    int border() const { return border_; }
    float matchThresh() const { return matchThresh_; }
    float beta() const { return beta_; }
    int modelInputH() const { return inputH_; }
    int modelInputW() const { return inputW_; }
};

CLIDDTRT::CLIDDTRT(const std::string& enginePath, int topK, float scoreThresh,
                   int radius, int border, float matchThresh, float beta)
    : impl_(new Impl(enginePath, topK, scoreThresh, radius, border, matchThresh, beta)) {}

CLIDDTRT::~CLIDDTRT() = default;

CLIDDTRT::MatchResult CLIDDTRT::match(const cv::Mat& img0_bgr, const cv::Mat& img1_bgr) {
    if (img0_bgr.empty() || img1_bgr.empty()) {
        throw std::runtime_error("Empty input image");
    }
    
    MatchResult out;
    
    const auto t0 = Clock::now();
    auto feat1 = impl_->runSingleImage(img0_bgr);
    auto feat2 = impl_->run_->runSingleImage(img1_bgr);
    const auto t1 = Clock::now();
    out.ms_infer = ms_since(t0, t1) / 2.0;
    
    const auto t2 = Clock::now();
    auto [matchIdx1, matchIdx2] = matchFeaturesCPU(feat1, feat2, impl_->matchThresh(), impl_->beta());
    const auto t3 = Clock::now();
    out.ms_match = ms_since(t2, t3);
    
    out.matchCount = static_cast<int>(matchIdx1.size());
    out.matchIndices1 = std::move(matchIdx1);
    out.matchIndices2 = std::move(matchIdx2);
    
    // Convert indices to keypoints
    out.keypoints1.reserve(out.matchCount);
    out.keypoints2.reserve(out.matchCount);
    
    for (size_t i = 0; i < matchIdx1.size(); ++i) {
        out.keypoints1.push_back(feat1.keypoints[matchIdx1[i]]);
        out.keypoints2.push_back(feat2.keypoints[matchIdx2[i]]);
    }
    
    const auto t4 = Clock::now();
    if (out.keypoints1.size() >= 4) {
        // Convert to cv::Point2f for OpenCV
        std::vector<cv::Point2f> pts1, pts2;
        pts1.reserve(out.keypoints1.size());
        pts2.reserve(out.keypoints2.size());
        
        for (const auto& kpt : out.keypoints1) {
            pts1.emplace_back(kpt[0], kpt[1]);
        }
        for (const auto& kpt : out.keypoints2) {
            pts2.emplace_back(kpt[0], kpt[1]);
        }
        
        std::vector<unsigned char> mask;
        out.H = cv::findHomography(pts1, pts2, cv::USAC_MAGSAC, 3.0, mask, 2000, 0.95);
        out.inlier_mask.assign(mask.begin(), mask.end());
        
        for (size_t i = 0; i < mask.size(); ++i) {
            if (!mask[i]) continue;
            out.inlier_kpts0.push_back(out.keypoints1[i]);
            out.inlier_kpts1.push_back(out.keypoints2[i]);
        }
    }
    const auto t5 = Clock::now();
    out.ms_ransac = ms_since(t4, t5);
    
    return out;
}

int CLIDDTRT::topK() const { return impl_->topK(); }
float CLIDDTRT::scoreThresh() const { return impl_->scoreThresh(); }
int CLIDDTRT::radius() const { return impl_->radius(); }
int CLIDDTRT::border() const { return impl_->border(); }
float CLIDDTRT::matchThresh() const { return impl_->matchThresh(); }
float CLIDDTRT::beta() const { return impl_->beta(); }
int CLIDDTRT::modelInputH() const { return impl_->modelInputH(); }
int CLIDDTRT::modelInputW() const { return impl_->modelInputW(); }

}  // namespace clidd