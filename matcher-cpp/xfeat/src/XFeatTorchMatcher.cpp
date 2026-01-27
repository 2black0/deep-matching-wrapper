#include "xfeat/XFeatTorchMatcher.h"

#include <torch/script.h>
#include <torch/torch.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace dmw::xfeat {
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

// Mutual Nearest Neighbor matching using cosine similarity
void mnn_match_cv(const cv::Mat& desc0, const cv::Mat& desc1, float min_cossim,
                  std::vector<int>& out_i0, std::vector<int>& out_i1) {
  out_i0.clear();
  out_i1.clear();
  if (desc0.empty() || desc1.empty()) return;

  // Compute cosine similarity: desc0 @ desc1.T
  cv::Mat cossim = desc0 * desc1.t();

  const int n0 = cossim.rows;
  const int n1 = cossim.cols;
  std::vector<int> nn12(n0, 0);
  std::vector<int> nn21(n1, 0);

  // Find nearest neighbor for each descriptor in desc0
  for (int i = 0; i < n0; ++i) {
    const float* row = cossim.ptr<float>(i);
    int best = 0;
    float bestv = row[0];
    for (int j = 1; j < n1; ++j) {
      if (row[j] > bestv) {
        bestv = row[j];
        best = j;
      }
    }
    nn12[i] = best;
  }

  // Find nearest neighbor for each descriptor in desc1
  for (int j = 0; j < n1; ++j) {
    int best = 0;
    float bestv = cossim.at<float>(0, j);
    for (int i = 1; i < n0; ++i) {
      float v = cossim.at<float>(i, j);
      if (v > bestv) {
        bestv = v;
        best = i;
      }
    }
    nn21[j] = best;
  }

  // Mutual nearest neighbors with threshold
  for (int i = 0; i < n0; ++i) {
    int j = nn12[i];
    if (i != nn21[j]) continue;
    float score = cossim.at<float>(i, j);
    if (score < min_cossim) continue;
    out_i0.push_back(i);
    out_i1.push_back(j);
  }
}

}  // namespace

struct XFeatTorchMatcher::Impl {
  XFeatConfig cfg;
  torch::Device device;
  torch::Dtype dtype;
  torch::jit::Module module_xfeat;
  torch::jit::Module module_xfeat_star;

  explicit Impl(const XFeatConfig& c)
      : cfg(c),
        device((c.device == "cuda" && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU),
        dtype(torch::kFloat32) {
    
    const std::string base = "matcher-cpp/xfeat/weights";
    
    // Load appropriate TorchScript model based on mode
    if (cfg.mode == XFeatMode::XFEAT) {
      // Load sparse feature extractor
      const std::string pt_path = base + "/xfeat_fp32_k" + std::to_string(cfg.top_k) + ".pt";
      if (!std::filesystem::exists(pt_path)) {
        throw std::runtime_error(
            "Missing TorchScript weights: " + pt_path +
            ". Export with: python matcher/xfeat/torchscript/convert_torchscript_xfeat.py --topk " +
            std::to_string(cfg.top_k));
      }
      module_xfeat = torch::jit::load(pt_path, device);
      module_xfeat.eval();
      module_xfeat.to(device);
      module_xfeat.to(dtype);

    } else if (cfg.mode == XFeatMode::XFEAT_STAR) {
      // Load semi-dense matcher - auto-select CUDA version if on CUDA device
      const std::string device_suffix = (device.is_cuda()) ? "_cuda" : "";
      const std::string pt_path = base + "/xfeat_star_fp32_k" + std::to_string(cfg.top_k) + device_suffix + ".pt";
      if (!std::filesystem::exists(pt_path)) {
        throw std::runtime_error(
            "Missing TorchScript weights: " + pt_path + 
            ". Export with: python matcher/xfeat/torchscript/convert_torchscript_xfeat_star.py --topk " + 
            std::to_string(cfg.top_k) + " --device " + (device.is_cuda() ? "cuda" : "cpu"));
      }
      module_xfeat_star = torch::jit::load(pt_path, device);
      module_xfeat_star.eval();
      module_xfeat_star.to(device);
      module_xfeat_star.to(dtype);
    }
  }

  struct InferOut {
    torch::Tensor kpts;   // (N,2) float32 on device
    torch::Tensor scores; // (N,) float32 on device
    torch::Tensor desc;   // (N,64) float32 on device
  };

  InferOut infer_sparse(const cv::Mat& bgr) {
    cv::Mat rgb = preprocess_bgr(bgr);
    const int H = rgb.rows;
    const int W = rgb.cols;

    auto t = torch::from_blob(rgb.data, {H, W, 3}, torch::kUInt8);
    t = t.to(torch::kFloat32).div_(255.0);
    t = t.permute({2, 0, 1}).contiguous().unsqueeze(0).to(device);

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(t);
    auto out_iv = module_xfeat.forward(inputs);
    auto tup = out_iv.toTuple();

    InferOut out;
    out.kpts = tup->elements()[0].toTensor().to(device).to(dtype);
    out.scores = tup->elements()[1].toTensor().to(device).to(dtype);
    out.desc = tup->elements()[2].toTensor().to(device).to(dtype);
    return out;
  }

  torch::Tensor infer_xfeat_star(const cv::Mat& bgr0, const cv::Mat& bgr1) {
    cv::Mat rgb0 = preprocess_bgr(bgr0);
    cv::Mat rgb1 = preprocess_bgr(bgr1);
    const int H0 = rgb0.rows, W0 = rgb0.cols;
    const int H1 = rgb1.rows, W1 = rgb1.cols;

    auto t0 = torch::from_blob(rgb0.data, {H0, W0, 3}, torch::kUInt8);
    t0 = t0.to(torch::kFloat32).div_(255.0).permute({2, 0, 1}).contiguous().unsqueeze(0).to(device);
    
    auto t1 = torch::from_blob(rgb1.data, {H1, W1, 3}, torch::kUInt8);
    t1 = t1.to(torch::kFloat32).div_(255.0).permute({2, 0, 1}).contiguous().unsqueeze(0).to(device);

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(t0);
    inputs.emplace_back(t1);
    
    auto out_iv = module_xfeat_star.forward(inputs);
    auto matches = out_iv.toTensor().to(device).to(dtype);  // (N, 4) as (x0, y0, x1, y1)
    return matches;
  }
};

XFeatTorchMatcher::XFeatTorchMatcher(const XFeatConfig& cfg) : impl_(new Impl(cfg)) {}

XFeatTorchMatcher::~XFeatTorchMatcher() { delete impl_; }

MatchResult XFeatTorchMatcher::match(const cv::Mat& img0_bgr, const cv::Mat& img1_bgr) {
  if (img0_bgr.empty() || img1_bgr.empty()) {
    throw std::runtime_error("Empty input image");
  }
  
  MatchResult out;
  torch::NoGradGuard ng;
  const bool is_cuda = impl_->device.is_cuda();

  if (impl_->cfg.mode == XFeatMode::XFEAT) {
    // Sparse features + MNN matching
    
    // Inference
    if (is_cuda) torch::cuda::synchronize();
    const auto t_inf0 = Clock::now();
    auto o0 = impl_->infer_sparse(img0_bgr);
    auto o1 = impl_->infer_sparse(img1_bgr);
    if (is_cuda) torch::cuda::synchronize();
    const auto t_inf1 = Clock::now();
    out.ms_infer = ms_since(t_inf0, t_inf1) / 2.0;

    // Convert outputs to CPU
    auto k0_cpu = o0.kpts.to(torch::kCPU);
    auto k1_cpu = o1.kpts.to(torch::kCPU);
    auto d0_cpu = o0.desc.to(torch::kCPU);
    auto d1_cpu = o1.desc.to(torch::kCPU);

    // Store all keypoints
    {
      const int N0 = (int)k0_cpu.size(0);
      out.all_kpts0.reserve(N0);
      auto acc = k0_cpu.accessor<float, 2>();
      for (int i = 0; i < N0; ++i) {
        out.all_kpts0.emplace_back(acc[i][0], acc[i][1]);
      }
    }
    {
      const int N1 = (int)k1_cpu.size(0);
      out.all_kpts1.reserve(N1);
      auto acc = k1_cpu.accessor<float, 2>();
      for (int i = 0; i < N1; ++i) {
        out.all_kpts1.emplace_back(acc[i][0], acc[i][1]);
      }
    }

    // Store all descriptors
    {
      const int N0 = (int)d0_cpu.size(0);
      const int D = (int)d0_cpu.size(1);
      out.all_desc0 = cv::Mat(N0, D, CV_32F);
      std::memcpy(out.all_desc0.data, d0_cpu.contiguous().data_ptr<float>(), 
                  (size_t)N0 * (size_t)D * sizeof(float));
    }
    {
      const int N1 = (int)d1_cpu.size(0);
      const int D = (int)d1_cpu.size(1);
      out.all_desc1 = cv::Mat(N1, D, CV_32F);
      std::memcpy(out.all_desc1.data, d1_cpu.contiguous().data_ptr<float>(), 
                  (size_t)N1 * (size_t)D * sizeof(float));
    }

    // Matching with MNN
    const auto t_m0 = Clock::now();
    std::vector<int> i0, i1;
    mnn_match_cv(out.all_desc0, out.all_desc1, impl_->cfg.min_cossim, i0, i1);
    const auto t_m1 = Clock::now();
    out.ms_match = ms_since(t_m0, t_m1);

    // Store matched keypoints
    for (size_t i = 0; i < i0.size(); ++i) {
      out.matched_kpts0.push_back(out.all_kpts0[i0[i]]);
      out.matched_kpts1.push_back(out.all_kpts1[i1[i]]);
    }

  } else if (impl_->cfg.mode == XFeatMode::XFEAT_STAR) {
    // Semi-dense features + refinement matching
    
    // Inference (includes matching and refinement)
    if (is_cuda) torch::cuda::synchronize();
    const auto t_inf0 = Clock::now();
    auto matches_t = impl_->infer_xfeat_star(img0_bgr, img1_bgr);
    if (is_cuda) torch::cuda::synchronize();
    const auto t_inf1 = Clock::now();
    out.ms_infer = ms_since(t_inf0, t_inf1);
    out.ms_match = 0.0;  // Matching is included in inference for xfeat-star

    // Convert matches to CPU and extract
    auto matches_cpu = matches_t.to(torch::kCPU);
    const int N = (int)matches_cpu.size(0);
    
    if (N > 0) {
      auto acc = matches_cpu.accessor<float, 2>();
      out.matched_kpts0.reserve(N);
      out.matched_kpts1.reserve(N);
      for (int i = 0; i < N; ++i) {
        out.matched_kpts0.emplace_back(acc[i][0], acc[i][1]);
        out.matched_kpts1.emplace_back(acc[i][2], acc[i][3]);
      }
    }

    // For xfeat-star, we don't have individual keypoint/descriptor outputs in the TorchScript model
    // These would need to be computed separately if needed

  }

  // RANSAC for homography estimation
  const auto t_r0 = Clock::now();
  if (out.matched_kpts0.size() >= 4) {
    std::vector<unsigned char> mask;
    out.H = cv::findHomography(out.matched_kpts0, out.matched_kpts1, 
                                cv::USAC_MAGSAC, 3.0, mask, 2000, 0.95);
    out.inlier_mask.assign(mask.begin(), mask.end());
    for (size_t i = 0; i < mask.size(); ++i) {
      if (!mask[i]) continue;
      out.inlier_kpts0.push_back(out.matched_kpts0[i]);
      out.inlier_kpts1.push_back(out.matched_kpts1[i]);
    }
  }
  const auto t_r1 = Clock::now();
  out.ms_ransac = ms_since(t_r0, t_r1);

  return out;
}

}  // namespace dmw::xfeat
