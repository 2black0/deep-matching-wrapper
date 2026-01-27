#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace dmw::edm {

struct EdmConfig {
  std::string device = "cpu";  // cpu | cuda
  std::string dtype = "fp32";  // fp32 (TorchScript export is fp32-only for now)

  // Exported TorchScript is fixed-shape by design (like clidd/liftfeat exports).
  int input_w = 640;
  int input_h = 480;
  int topk = 1680;
  int local_resolution = 8;

  // Selection thresholds (mirrors EDM config intent).
  float mconf_thr = 0.2f;
  int border_rm_coarse = 2;  // multiplied by local_resolution
  bool sigma_selection = true;
  float sigma_thr = 0.0f;

  std::string weights_path = "matcher-cpp/edm/weights/edm_fp32_w640_h480_topk1680.pt";
};

struct MatchResult {
  cv::Mat H;  // 3x3 CV_64F
  std::vector<cv::Point2f> matched_kpts0;
  std::vector<cv::Point2f> matched_kpts1;
  std::vector<cv::Point2f> inlier_kpts0;
  std::vector<cv::Point2f> inlier_kpts1;
  std::vector<uint8_t> inlier_mask;

  double ms_preprocess = 0.0;
  double ms_infer = 0.0;
  double ms_postprocess = 0.0;
  double ms_ransac = 0.0;
};

class EdmTorchMatcher {
 public:
  explicit EdmTorchMatcher(const EdmConfig& cfg);
  MatchResult match(const cv::Mat& img0_bgr, const cv::Mat& img1_bgr);

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace dmw::edm
