#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>

namespace dmw {
namespace liftfeat_onnx {

struct LiftFeatConfig {
    std::string device = "cuda";  // "cpu" or "cuda"
    std::string dtype = "fp32";   // "fp32" or "fp16"
    int width = 640;
    int height = 480;
    int top_k = 4096;
    float detect_threshold = 0.005f;
    float min_cossim = -1.0f;
    std::string weights_path = "";  // Optional, auto-detected if empty
};

struct MatchResult {
    std::vector<cv::Point2f> mkpts0;
    std::vector<cv::Point2f> mkpts1;
    std::vector<cv::Point2f> kpts0;
    std::vector<cv::Point2f> kpts1;
    std::vector<std::vector<float>> desc0;
    std::vector<std::vector<float>> desc1;
    
    double ms_preprocess = 0.0;
    double ms_infer = 0.0;
    double ms_postprocess = 0.0;
    double ms_match = 0.0;
    double ms_total = 0.0;
};

class LiftFeatOnnxMatcher {
public:
    explicit LiftFeatOnnxMatcher(const LiftFeatConfig& config);
    ~LiftFeatOnnxMatcher();
    
    MatchResult match(const cv::Mat& img0, const cv::Mat& img1);
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace liftfeat_onnx
} // namespace dmw
