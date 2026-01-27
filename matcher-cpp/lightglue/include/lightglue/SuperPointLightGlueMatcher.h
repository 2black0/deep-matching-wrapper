#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace dmw::lightglue {

struct SuperPointLightGlueConfig {
  std::string device = "cpu";  // cpu | cuda
  std::string dtype = "fp32";  // TorchScript export is fp32-only for now
  
  // SuperPoint parameters
  int max_num_keypoints = 2048;
  float detection_threshold = 0.005f;  // Keypoint detection threshold
  int nms_radius = 4;                   // NMS radius in pixels
  int remove_borders = 4;               // Remove keypoints within this distance from borders
  
  // LightGlue parameters
  float match_threshold = 0.1f;  // Matching threshold (probability after exp; 0.1 = default)
  
  // Model paths (auto-selected based on device if empty)
  std::string superpoint_weights = "";
  std::string lightglue_weights = "";
};

struct MatchResult {
  cv::Mat H;  // 3x3 CV_64F homography matrix
  
  // All detected keypoints and descriptors
  std::vector<cv::Point2f> all_kpts0;
  std::vector<cv::Point2f> all_kpts1;
  cv::Mat all_desc0;  // (N0, 256) CV_32F
  cv::Mat all_desc1;  // (N1, 256) CV_32F
  
  // Matched keypoints (after LightGlue matching)
  std::vector<cv::Point2f> matched_kpts0;
  std::vector<cv::Point2f> matched_kpts1;
  
  // Inlier keypoints (after RANSAC)
  std::vector<cv::Point2f> inlier_kpts0;
  std::vector<cv::Point2f> inlier_kpts1;
  std::vector<uint8_t> inlier_mask;
  
  // Timing information
  double ms_superpoint0 = 0.0;
  double ms_superpoint1 = 0.0;
  double ms_lightglue = 0.0;
  double ms_ransac = 0.0;
};

class SuperPointLightGlueMatcher {
 public:
  explicit SuperPointLightGlueMatcher(const SuperPointLightGlueConfig& cfg);
  ~SuperPointLightGlueMatcher();
  
  MatchResult match(const cv::Mat& img0_bgr, const cv::Mat& img1_bgr);
  
 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace dmw::lightglue
