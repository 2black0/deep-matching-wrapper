#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace dmw::liftfeat {

struct LiftFeatConfig {
  std::string model_name = "liftfeat";
  std::string device = "cpu";  // cpu | cuda
  std::string dtype = "fp32";  // TorchScript export is fp32-only for now.

  int top_k = 4096;
  float min_cossim = -1.0f;
};

struct MatchResult {
  cv::Mat H;  // 3x3 CV_64F
  std::vector<cv::Point2f> all_kpts0;
  std::vector<cv::Point2f> all_kpts1;
  cv::Mat all_desc0;  // (N0,64) CV_32F
  cv::Mat all_desc1;  // (N1,64) CV_32F
  std::vector<cv::Point2f> matched_kpts0;
  std::vector<cv::Point2f> matched_kpts1;
  std::vector<cv::Point2f> inlier_kpts0;
  std::vector<cv::Point2f> inlier_kpts1;
  std::vector<uint8_t> inlier_mask;

  double ms_preprocess = 0.0;
  double ms_infer = 0.0;
  double ms_match = 0.0;
  double ms_ransac = 0.0;
};

class LiftFeatTorchMatcher {
 public:
  explicit LiftFeatTorchMatcher(const LiftFeatConfig& cfg);
  MatchResult match(const cv::Mat& img0_bgr, const cv::Mat& img1_bgr);

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace dmw::liftfeat
