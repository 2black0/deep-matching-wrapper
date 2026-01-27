#include "edm/EdmTorchMatcher.h"

#include <torch/script.h>
#include <torch/torch.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace dmw::edm {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(const Clock::time_point& t0, const Clock::time_point& t1) {
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

float clampf(float v, float lo, float hi) {
  return std::max(lo, std::min(hi, v));
}

cv::Mat to_gray_resized_f32_contig(const cv::Mat& bgr, int w, int h) {
  if (bgr.empty()) throw std::runtime_error("Empty input image");
  if (bgr.channels() != 3) throw std::runtime_error("Expected BGR 3-channel image");

  cv::Mat gray_u8;
  cv::cvtColor(bgr, gray_u8, cv::COLOR_BGR2GRAY);
  cv::Mat gray_rs_u8;
  cv::resize(gray_u8, gray_rs_u8, cv::Size(w, h), 0.0, 0.0, cv::INTER_LINEAR);
  cv::Mat gray_f32;
  gray_rs_u8.convertTo(gray_f32, CV_32F, 1.0 / 255.0);
  if (!gray_f32.isContinuous()) gray_f32 = gray_f32.clone();
  return gray_f32;
}

bool in_bounds(const cv::Point2f& p, int w, int h, float border) {
  return p.x >= border && p.y >= border && p.x <= (float)w - border && p.y <= (float)h - border;
}

}  // namespace

struct EdmTorchMatcher::Impl {
  EdmConfig cfg;
  torch::Device device;
  torch::jit::Module module;

  explicit Impl(const EdmConfig& c)
      : cfg(c), device((c.device == "cuda" && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU) {
    if (!std::filesystem::exists(cfg.weights_path)) {
      throw std::runtime_error(
          "Missing TorchScript weights: " + cfg.weights_path +
          ". Export with matcher/edm/torchscript/convert_torchscript.py");
    }

    module = torch::jit::load(cfg.weights_path, device);
    module.eval();
    module.to(device);
    module.to(torch::kFloat32);
  }

  torch::Tensor build_input(const cv::Mat& img0_bgr, const cv::Mat& img1_bgr, double& ms_pre) {
    const auto t0 = Clock::now();
    cv::Mat g0 = to_gray_resized_f32_contig(img0_bgr, cfg.input_w, cfg.input_h);
    cv::Mat g1 = to_gray_resized_f32_contig(img1_bgr, cfg.input_w, cfg.input_h);

    // from_blob -> (H,W) float32 CPU
    auto t0_cpu = torch::from_blob(g0.data, {cfg.input_h, cfg.input_w}, torch::kFloat32).clone();
    auto t1_cpu = torch::from_blob(g1.data, {cfg.input_h, cfg.input_w}, torch::kFloat32).clone();
    // (1,1,H,W)
    t0_cpu = t0_cpu.unsqueeze(0).unsqueeze(0);
    t1_cpu = t1_cpu.unsqueeze(0).unsqueeze(0);
    // (1,2,H,W)
    auto x = torch::cat({t0_cpu, t1_cpu}, 1).to(device);
    const auto t1 = Clock::now();
    ms_pre = ms_since(t0, t1);
    return x;
  }

  torch::Tensor forward(const torch::Tensor& x, double& ms_inf) {
    const bool is_cuda = device.is_cuda();
    torch::NoGradGuard ng;
    if (is_cuda) torch::cuda::synchronize();
    const auto t0 = Clock::now();
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(x);
    auto y = module.forward(inputs).toTensor();
    if (is_cuda) torch::cuda::synchronize();
    const auto t1 = Clock::now();
    ms_inf = ms_since(t0, t1);
    return y;
  }
};

EdmTorchMatcher::EdmTorchMatcher(const EdmConfig& cfg) : impl_(new Impl(cfg)) {}

MatchResult EdmTorchMatcher::match(const cv::Mat& img0_bgr, const cv::Mat& img1_bgr) {
  if (img0_bgr.empty() || img1_bgr.empty()) throw std::runtime_error("Empty input image");
  MatchResult out;

  const int W = impl_->cfg.input_w;
  const int H = impl_->cfg.input_h;
  const float sx0 = (float)img0_bgr.cols / (float)W;
  const float sy0 = (float)img0_bgr.rows / (float)H;
  const float sx1 = (float)img1_bgr.cols / (float)W;
  const float sy1 = (float)img1_bgr.rows / (float)H;

  // Preprocess + forward
  double ms_pre = 0.0;
  auto x = impl_->build_input(img0_bgr, img1_bgr, ms_pre);
  out.ms_preprocess = ms_pre;

  double ms_inf = 0.0;
  auto y = impl_->forward(x, ms_inf);
  out.ms_infer = ms_inf;

  const auto t_post0 = Clock::now();

  // Expect (topk, 11)
  if (y.dim() != 2 || (int)y.size(1) != 11) {
    throw std::runtime_error("Unexpected EDM output shape (expected [topk,11])");
  }
  if (impl_->cfg.topk > 0 && (int)y.size(0) != impl_->cfg.topk) {
    throw std::runtime_error("Unexpected EDM output rows (expected topk=" + std::to_string(impl_->cfg.topk) + ")");
  }

  auto y_cpu = y.to(torch::kCPU).contiguous();
  const float* ptr = y_cpu.data_ptr<float>();
  const int K = (int)y_cpu.size(0);

  const float border = (float)(impl_->cfg.border_rm_coarse * impl_->cfg.local_resolution);
  const float lr = (float)impl_->cfg.local_resolution;

  out.matched_kpts0.clear();
  out.matched_kpts1.clear();
  out.matched_kpts0.reserve((size_t)K);
  out.matched_kpts1.reserve((size_t)K);

  for (int i = 0; i < K; ++i) {
    const float* r = ptr + i * 11;

    const float x0c = r[0];
    const float y0c = r[1];
    const float x1c = r[2];
    const float y1c = r[3];

    const float off01x = clampf(r[4] * lr, -lr / 2.0f, lr / 2.0f);
    const float off01y = clampf(r[5] * lr, -lr / 2.0f, lr / 2.0f);
    const float off10x = clampf(r[6] * lr, -lr / 2.0f, lr / 2.0f);
    const float off10y = clampf(r[7] * lr, -lr / 2.0f, lr / 2.0f);

    const float score01 = r[8];
    const float score10 = r[9];
    const float mconf = r[10];
    if (mconf <= impl_->cfg.mconf_thr) continue;

    // Candidates in resized coords.
    cv::Point2f p0_fwd(x0c, y0c);
    cv::Point2f p1_fwd(x1c + off01x, y1c + off01y);
    cv::Point2f p0_bwd(x0c + off10x, y0c + off10y);
    cv::Point2f p1_bwd(x1c, y1c);

    const bool ok_fwd = in_bounds(p0_fwd, W, H, border) && in_bounds(p1_fwd, W, H, border);
    const bool ok_bwd = in_bounds(p0_bwd, W, H, border) && in_bounds(p1_bwd, W, H, border);
    if (!ok_fwd && !ok_bwd) continue;

    bool choose_fwd = true;
    float chosen_score = score01;
    if (impl_->cfg.sigma_selection) {
      choose_fwd = score01 >= score10;
      chosen_score = choose_fwd ? score01 : score10;
      if (chosen_score <= impl_->cfg.sigma_thr) continue;
    }

    cv::Point2f p0 = choose_fwd ? p0_fwd : p0_bwd;
    cv::Point2f p1 = choose_fwd ? p1_fwd : p1_bwd;
    if (choose_fwd && !ok_fwd) continue;
    if (!choose_fwd && !ok_bwd) continue;

    // Rescale to original.
    p0.x *= sx0;
    p0.y *= sy0;
    p1.x *= sx1;
    p1.y *= sy1;

    out.matched_kpts0.emplace_back(p0);
    out.matched_kpts1.emplace_back(p1);
  }

  const auto t_post1 = Clock::now();
  out.ms_postprocess = ms_since(t_post0, t_post1);

  const auto t_r0 = Clock::now();
  if (out.matched_kpts0.size() >= 4) {
    std::vector<unsigned char> mask;
    out.H = cv::findHomography(out.matched_kpts0, out.matched_kpts1, cv::USAC_MAGSAC, 3.0, mask, 2000, 0.95);
    out.inlier_mask.assign(mask.begin(), mask.end());
    out.inlier_kpts0.reserve(mask.size());
    out.inlier_kpts1.reserve(mask.size());
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

}  // namespace dmw::edm
