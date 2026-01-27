#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace dmw::xfeat {

enum class XFeatMode {
  XFEAT,        // Sparse features + Mutual Nearest Neighbors
  XFEAT_STAR    // Semi-dense features + Refinement
};

struct XFeatConfig {
  XFeatMode mode = XFeatMode::XFEAT;
  std::string device = "cpu";  // cpu | cuda
  std::string dtype = "fp32";   // TorchScript export is fp32-only for now
  
  int top_k = 4096;
  float detection_threshold = 0.05f;  // For sparse detection
  float fine_conf = 0.25f;            // For xfeat-star refinement
  
  // Matching parameters (for XFEAT mode with MNN)
  float min_cossim = -1.0f;  // Minimum cosine similarity for matches (-1 disables threshold)
};

struct MatchResult {
  cv::Mat H;  // 3x3 CV_64F homography matrix
  
  // All detected keypoints and descriptors
  std::vector<cv::Point2f> all_kpts0;
  std::vector<cv::Point2f> all_kpts1;
  cv::Mat all_desc0;  // (N0,64) CV_32F
  cv::Mat all_desc1;  // (N1,64) CV_32F
  
  // Matched keypoints (after matching)
  std::vector<cv::Point2f> matched_kpts0;
  std::vector<cv::Point2f> matched_kpts1;
  
  // Inlier keypoints (after RANSAC)
  std::vector<cv::Point2f> inlier_kpts0;
  std::vector<cv::Point2f> inlier_kpts1;
  std::vector<uint8_t> inlier_mask;
  
  // Timing information
  double ms_preprocess = 0.0;
  double ms_infer = 0.0;
  double ms_match = 0.0;
  double ms_ransac = 0.0;
};

class XFeatTorchMatcher {
 public:
  explicit XFeatTorchMatcher(const XFeatConfig& cfg);
  ~XFeatTorchMatcher();
  
  MatchResult match(const cv::Mat& img0_bgr, const cv::Mat& img1_bgr);
  
 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace dmw::xfeat
