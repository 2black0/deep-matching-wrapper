#include "lightglue/SuperPointLightGlueMatcher.h"

#include <torch/script.h>
#include <torch/torch.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace dmw::lightglue {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(const Clock::time_point& t0, const Clock::time_point& t1) {
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

cv::Mat preprocess_bgr(const cv::Mat& bgr) {
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  return rgb;
}

// Simple NMS implementation
torch::Tensor simple_nms(torch::Tensor scores, int nms_radius) {
  if (nms_radius <= 0) return scores;
  
  int64_t kernel_size = nms_radius * 2 + 1;
  auto max_pool = torch::nn::functional::max_pool2d(
    scores,
    torch::nn::functional::MaxPool2dFuncOptions(kernel_size)
      .stride(1)
      .padding(nms_radius)
  );
  
  auto zeros = torch::zeros_like(scores);
  auto max_mask = scores == max_pool;
  
  // Two iterations of suppression
  for (int i = 0; i < 2; ++i) {
    auto supp_mask = torch::nn::functional::max_pool2d(
      max_mask.to(torch::kFloat32),
      torch::nn::functional::MaxPool2dFuncOptions(kernel_size)
        .stride(1)
        .padding(nms_radius)
    ) > 0;
    
    auto supp_scores = torch::where(supp_mask, zeros, scores);
    auto supp_max_pool = torch::nn::functional::max_pool2d(
      supp_scores,
      torch::nn::functional::MaxPool2dFuncOptions(kernel_size)
        .stride(1)
        .padding(nms_radius)
    );
    
    auto new_max_mask = supp_scores == supp_max_pool;
    max_mask = max_mask | (new_max_mask & (~supp_mask));
  }
  
  return torch::where(max_mask, scores, zeros);
}

// Sample descriptors at keypoint locations using bilinear interpolation
torch::Tensor sample_descriptors(
    torch::Tensor keypoints,  // (N, 2)
    torch::Tensor descriptors,  // (1, 256, H/8, W/8)
    int s = 8
) {
  auto opts = keypoints.options();
  int64_t b = descriptors.size(0);
  int64_t c = descriptors.size(1);
  int64_t h = descriptors.size(2);
  int64_t w = descriptors.size(3);
  
  // Adjust keypoints for descriptor sampling
  auto kpts = keypoints.clone();
  kpts = kpts - (s / 2.0f) + 0.5f;
  kpts.index({torch::indexing::Slice(), 0}) /= (w * s - s / 2.0f - 0.5f);
  kpts.index({torch::indexing::Slice(), 1}) /= (h * s - s / 2.0f - 0.5f);
  kpts = kpts * 2.0f - 1.0f;  // Normalize to [-1, 1]
  
  // Reshape for grid_sample: (B, 1, N, 2)
  auto grid = kpts.view({1, 1, -1, 2});
  
  // Sample descriptors
  auto sampled = torch::nn::functional::grid_sample(
    descriptors,
    grid,
    torch::nn::functional::GridSampleFuncOptions()
      .mode(torch::kBilinear)
      .align_corners(true)
  );
  
  // Reshape and normalize: (N, 256)
  sampled = sampled.reshape({c, -1}).t();
  sampled = torch::nn::functional::normalize(sampled, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
  
  return sampled;
}

}  // namespace

struct SuperPointLightGlueMatcher::Impl {
  SuperPointLightGlueConfig cfg;
  torch::Device device;
  torch::Dtype dtype;
  torch::jit::Module module_superpoint;
  torch::jit::Module module_lightglue;

  explicit Impl(const SuperPointLightGlueConfig& c)
      : cfg(c),
        device((c.device == "cuda" && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU),
        dtype(torch::kFloat32) {
    
    const std::string base = "matcher-cpp/lightglue/weights";
    const std::string device_suffix = (device.is_cuda()) ? "_cuda" : "";
    
    // Auto-select model paths if not provided
    std::string sp_path = cfg.superpoint_weights.empty()
      ? base + "/superpoint_fp32" + device_suffix + ".pt"
      : cfg.superpoint_weights;
    
    std::string lg_path = cfg.lightglue_weights.empty()
      ? base + "/superpoint_lightglue_fp32_k" + std::to_string(cfg.max_num_keypoints) + device_suffix + ".pt"
      : cfg.lightglue_weights;
    
    if (!std::filesystem::exists(sp_path)) {
      throw std::runtime_error("SuperPoint model not found: " + sp_path);
    }
    if (!std::filesystem::exists(lg_path)) {
      throw std::runtime_error("LightGlue model not found: " + lg_path);
    }
    
    // Load models
    module_superpoint = torch::jit::load(sp_path, device);
    module_superpoint.eval();
    
    module_lightglue = torch::jit::load(lg_path, device);
    module_lightglue.eval();
  }
  
  // Extract SuperPoint features
  void extract_features(
      const cv::Mat& img_bgr,
      std::vector<cv::Point2f>& out_kpts,
      cv::Mat& out_desc,
      double& ms_time
  ) {
    auto t0 = Clock::now();
    
    // Preprocess image
    cv::Mat rgb = preprocess_bgr(img_bgr);
    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32F, 1.0 / 255.0);
    
    // Convert to tensor (1, 3, H, W)
    auto tensor = torch::from_blob(
      float_img.data,
      {1, float_img.rows, float_img.cols, 3},
      torch::kFloat32
    ).permute({0, 3, 1, 2}).to(device).contiguous();
    
    // Run SuperPoint backbone
    std::vector<torch::jit::IValue> inputs{tensor};
    auto outputs = module_superpoint.forward(inputs).toTuple();
    auto scores = outputs->elements()[0].toTensor();     // (1, 1, H, W)
    auto desc_map = outputs->elements()[1].toTensor();   // (1, 256, H/8, W/8)
    
    // Apply NMS
    scores = simple_nms(scores, cfg.nms_radius);
    
    // Convert to CPU for processing
    scores = scores.squeeze(0).squeeze(0).cpu();  // (H, W)
    desc_map = desc_map.cpu();
    
    int H = scores.size(0);
    int W = scores.size(1);
    
    // Extract keypoints above threshold, excluding borders
    std::vector<cv::Point2f> kpts;
    std::vector<float> scores_vec;
    
    for (int y = cfg.remove_borders; y < H - cfg.remove_borders; ++y) {
      for (int x = cfg.remove_borders; x < W - cfg.remove_borders; ++x) {
        float score = scores[y][x].item<float>();
        if (score > cfg.detection_threshold) {
          kpts.push_back(cv::Point2f((float)x, (float)y));
          scores_vec.push_back(score);
        }
      }
    }
    
    // Top-k selection
    int k = std::min((int)kpts.size(), cfg.max_num_keypoints);
    if (k < (int)kpts.size()) {
      // Partial sort to get top k
      std::vector<int> indices(kpts.size());
      std::iota(indices.begin(), indices.end(), 0);
      std::partial_sort(
        indices.begin(),
        indices.begin() + k,
        indices.end(),
        [&scores_vec](int i1, int i2) { return scores_vec[i1] > scores_vec[i2]; }
      );
      
      std::vector<cv::Point2f> top_kpts;
      for (int i = 0; i < k; ++i) {
        top_kpts.push_back(kpts[indices[i]]);
      }
      kpts = std::move(top_kpts);
    }
    
    // Sample descriptors at keypoint locations
    if (!kpts.empty()) {
      auto kpts_tensor = torch::zeros({(int64_t)kpts.size(), 2}, torch::kFloat32);
      for (size_t i = 0; i < kpts.size(); ++i) {
        kpts_tensor[i][0] = kpts[i].x;
        kpts_tensor[i][1] = kpts[i].y;
      }
      
      auto desc_sampled = sample_descriptors(kpts_tensor, desc_map, 8);  // (N, 256)
      
      // Convert to cv::Mat
      desc_sampled = desc_sampled.cpu().contiguous();
      out_desc = cv::Mat(kpts.size(), 256, CV_32F);
      std::memcpy(out_desc.data, desc_sampled.data_ptr<float>(), kpts.size() * 256 * sizeof(float));
    } else {
      out_desc = cv::Mat();
    }
    
    out_kpts = std::move(kpts);
    
    auto t1 = Clock::now();
    ms_time = ms_since(t0, t1);
  }
  
  // Match features using LightGlue
  void match_features(
      const std::vector<cv::Point2f>& kpts0,
      const cv::Mat& desc0,
      const std::vector<cv::Point2f>& kpts1,
      const cv::Mat& desc1,
      const cv::Size& img_size0,
      const cv::Size& img_size1,
      std::vector<cv::Point2f>& out_mkpts0,
      std::vector<cv::Point2f>& out_mkpts1,
      double& ms_time
  ) {
    auto t0 = Clock::now();
    
    if (kpts0.empty() || kpts1.empty()) {
      out_mkpts0.clear();
      out_mkpts1.clear();
      ms_time = ms_since(t0, Clock::now());
      return;
    }
    
    int n0 = kpts0.size();
    int n1 = kpts1.size();
    
    // Convert keypoints to tensors (1, N, 2)
    auto kpts0_tensor = torch::zeros({1, n0, 2}, torch::kFloat32).to(device);
    auto kpts1_tensor = torch::zeros({1, n1, 2}, torch::kFloat32).to(device);
    for (int i = 0; i < n0; ++i) {
      kpts0_tensor[0][i][0] = kpts0[i].x;
      kpts0_tensor[0][i][1] = kpts0[i].y;
    }
    for (int i = 0; i < n1; ++i) {
      kpts1_tensor[0][i][0] = kpts1[i].x;
      kpts1_tensor[0][i][1] = kpts1[i].y;
    }
    
    // Convert descriptors to tensors (1, N, 256)
    auto desc0_tensor = torch::from_blob(
      (void*)desc0.data,
      {1, n0, 256},
      torch::kFloat32
    ).to(device).contiguous();
    
    auto desc1_tensor = torch::from_blob(
      (void*)desc1.data,
      {1, n1, 256},
      torch::kFloat32
    ).to(device).contiguous();
    
    // Image sizes (1, 2) as [width, height]
    auto size0_tensor = torch::tensor({{img_size0.width, img_size0.height}}, torch::kInt64).to(device);
    auto size1_tensor = torch::tensor({{img_size1.width, img_size1.height}}, torch::kInt64).to(device);
    
    // Run LightGlue matcher
    std::vector<torch::jit::IValue> inputs{
      kpts0_tensor, desc0_tensor, kpts1_tensor, desc1_tensor, size0_tensor, size1_tensor
    };
    auto scores = module_lightglue.forward(inputs).toTensor();  // (1, N0+1, N1+1)
    
    scores = scores.squeeze(0).cpu();  // (N0+1, N1+1)
    
    // Extract matches using LightGlue's filter_matches logic
    // Scores are log-probabilities, need to exp() them
    auto scores_mat = scores.index({
      torch::indexing::Slice(0, n0),
      torch::indexing::Slice(0, n1)
    });  // (N0, N1)
    
    // Find best match for each keypoint in img0
    auto max0_tuple = torch::max(scores_mat, 1);
    auto max0_vals = std::get<0>(max0_tuple);  // (N0,) log-probs
    auto m0 = std::get<1>(max0_tuple);         // (N0,) indices
    
    // Find best match for each keypoint in img1
    auto max1_tuple = torch::max(scores_mat, 0);
    auto max1_vals = std::get<0>(max1_tuple);  // (N1,) log-probs
    auto m1 = std::get<1>(max1_tuple);         // (N1,) indices
    
    // Convert log-probabilities to probabilities
    auto max0_exp = max0_vals.exp();
    
    // Check mutual consistency and threshold
    for (int i = 0; i < n0; ++i) {
      int j = m0[i].item<int>();
      int i_back = m1[j].item<int>();
      
      if (i == i_back) {  // Mutual best match
        float prob = max0_exp[i].item<float>();
        if (prob > cfg.match_threshold) {
          out_mkpts0.push_back(kpts0[i]);
          out_mkpts1.push_back(kpts1[j]);
        }
      }
    }
    
    auto t1 = Clock::now();
    ms_time = ms_since(t0, t1);
  }
};

SuperPointLightGlueMatcher::SuperPointLightGlueMatcher(const SuperPointLightGlueConfig& cfg)
    : impl_(new Impl(cfg)) {}

SuperPointLightGlueMatcher::~SuperPointLightGlueMatcher() {
  delete impl_;
}

MatchResult SuperPointLightGlueMatcher::match(const cv::Mat& img0_bgr, const cv::Mat& img1_bgr) {
  MatchResult res;
  
  if (img0_bgr.empty() || img1_bgr.empty()) {
    throw std::invalid_argument("Input images cannot be empty");
  }
  
  // Extract SuperPoint features from both images
  impl_->extract_features(img0_bgr, res.all_kpts0, res.all_desc0, res.ms_superpoint0);
  impl_->extract_features(img1_bgr, res.all_kpts1, res.all_desc1, res.ms_superpoint1);
  
  // Match features using LightGlue
  impl_->match_features(
    res.all_kpts0, res.all_desc0,
    res.all_kpts1, res.all_desc1,
    img0_bgr.size(), img1_bgr.size(),
    res.matched_kpts0, res.matched_kpts1,
    res.ms_lightglue
  );
  
  // RANSAC to find inliers
  auto t0 = Clock::now();
  if (res.matched_kpts0.size() >= 4) {
    res.inlier_mask.resize(res.matched_kpts0.size());
    res.H = cv::findHomography(
      res.matched_kpts0, res.matched_kpts1,
      cv::RANSAC, 3.0, res.inlier_mask
    );
    
    // Extract inliers
    for (size_t i = 0; i < res.matched_kpts0.size(); ++i) {
      if (res.inlier_mask[i]) {
        res.inlier_kpts0.push_back(res.matched_kpts0[i]);
        res.inlier_kpts1.push_back(res.matched_kpts1[i]);
      }
    }
  }
  auto t1 = Clock::now();
  res.ms_ransac = ms_since(t0, t1);
  
  return res;
}

}  // namespace dmw::lightglue
