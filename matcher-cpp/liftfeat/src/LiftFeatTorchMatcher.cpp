#include "liftfeat/LiftFeatTorchMatcher.h"

#include <torch/script.h>
#include <torch/torch.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace dmw::liftfeat {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(const Clock::time_point& t0, const Clock::time_point& t1) {
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

cv::Mat ensure_contiguous_bgr(const cv::Mat& bgr) {
  if (bgr.isContinuous()) return bgr;
  return bgr.clone();
}

}  // namespace

struct LiftFeatTorchMatcher::Impl {
  LiftFeatConfig cfg;
  torch::Device device;
  torch::Dtype dtype;
  torch::jit::Module module;

  explicit Impl(const LiftFeatConfig& c)
      : cfg(c),
        device((c.device == "cuda" && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU),
        dtype(torch::kFloat32) {
    const std::string base = "matcher-cpp/liftfeat/weights";
    const std::string pt_path = base + "/liftfeat_fp32_k" + std::to_string(cfg.top_k) + ".pt";
    if (!std::filesystem::exists(pt_path)) {
      throw std::runtime_error(
          "Missing TorchScript weights: " + pt_path +
          ". Export with matcher/liftfeat/torchscript/convert_torchscript.py");
    }

    module = torch::jit::load(pt_path, device);
    module.eval();

    module.to(device);
    dtype = torch::kFloat32;
    module.to(torch::kFloat32);
  }

  struct InferOut {
    torch::Tensor kpts;    // (N,2) float32 on device
    torch::Tensor scores;  // (N,) float32 on device
    torch::Tensor desc;    // (N,64) float32 on device
  };

  InferOut infer_one_torch(const cv::Mat& bgr_in) {
    cv::Mat bgr = ensure_contiguous_bgr(bgr_in);
    const int H = bgr.rows;
    const int W = bgr.cols;

    auto t = torch::from_blob(bgr.data, {H, W, 3}, torch::kUInt8);
    t = t.to(torch::kFloat32).div_(255.0);
    t = t.permute({2, 0, 1}).contiguous().unsqueeze(0).to(device);

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(t);
    auto out_iv = module.forward(inputs);
    auto tup = out_iv.toTuple();

    InferOut out;
    out.kpts = tup->elements()[0].toTensor().to(device).to(torch::kFloat32);
    out.scores = tup->elements()[1].toTensor().to(device).to(torch::kFloat32);
    out.desc = tup->elements()[2].toTensor().to(device).to(torch::kFloat32);
    return out;
  }
};

LiftFeatTorchMatcher::LiftFeatTorchMatcher(const LiftFeatConfig& cfg) : impl_(new Impl(cfg)) {}

MatchResult LiftFeatTorchMatcher::match(const cv::Mat& img0_bgr, const cv::Mat& img1_bgr) {
  if (img0_bgr.empty() || img1_bgr.empty()) throw std::runtime_error("Empty input image");
  MatchResult out;

  torch::NoGradGuard ng;
  const bool is_cuda = impl_->device.is_cuda();

  if (is_cuda) torch::cuda::synchronize();
  const auto t_inf0 = Clock::now();
  auto o0 = impl_->infer_one_torch(img0_bgr);
  auto o1 = impl_->infer_one_torch(img1_bgr);
  if (is_cuda) torch::cuda::synchronize();
  const auto t_inf1 = Clock::now();
  out.ms_infer = ms_since(t_inf0, t_inf1) / 2.0;

  // Mutual nearest-neighbor by cosine similarity.
  if (is_cuda) torch::cuda::synchronize();
  const auto t_m0 = Clock::now();
  std::vector<torch::Tensor> keep;
  if (o0.desc.numel() > 0 && o1.desc.numel() > 0) {
    auto sim = o0.desc.matmul(o1.desc.t());
    auto nn12 = std::get<1>(sim.max(1));
    auto nn21 = std::get<1>(sim.max(0));
    auto ids0 = torch::arange(sim.size(0), torch::TensorOptions().dtype(torch::kLong).device(sim.device()));
    auto mutual = ids0.eq(nn21.index_select(0, nn12));
    auto ids_keep = ids0.index({mutual});
    auto nn12_keep = nn12.index({mutual});
    if (impl_->cfg.min_cossim > 0.0f && ids_keep.numel() > 0) {
      auto scores = sim.index({ids_keep, nn12_keep});
      auto good = scores.gt(impl_->cfg.min_cossim);
      ids_keep = ids_keep.index({good});
      nn12_keep = nn12_keep.index({good});
    }
    keep.emplace_back(ids_keep);
    keep.emplace_back(nn12_keep);
  } else {
    keep.emplace_back(torch::empty({0}, torch::TensorOptions().dtype(torch::kLong).device(impl_->device)));
    keep.emplace_back(torch::empty({0}, torch::TensorOptions().dtype(torch::kLong).device(impl_->device)));
  }

  auto mk0_t = o0.kpts.index_select(0, keep[0]);
  auto mk1_t = o1.kpts.index_select(0, keep[1]);
  if (is_cuda) torch::cuda::synchronize();
  const auto t_m1 = Clock::now();
  out.ms_match = ms_since(t_m0, t_m1);

  // Convert to CPU for reporting / RANSAC.
  auto k0_cpu = o0.kpts.to(torch::kCPU);
  auto k1_cpu = o1.kpts.to(torch::kCPU);
  auto d0_cpu = o0.desc.to(torch::kCPU);
  auto d1_cpu = o1.desc.to(torch::kCPU);
  auto mk0_cpu = mk0_t.to(torch::kCPU);
  auto mk1_cpu = mk1_t.to(torch::kCPU);

  // all_kpts
  {
    const int N0 = (int)k0_cpu.size(0);
    out.all_kpts0.reserve(N0);
    auto acc = k0_cpu.accessor<float, 2>();
    for (int i = 0; i < N0; ++i) out.all_kpts0.emplace_back(acc[i][0], acc[i][1]);
  }
  {
    const int N1 = (int)k1_cpu.size(0);
    out.all_kpts1.reserve(N1);
    auto acc = k1_cpu.accessor<float, 2>();
    for (int i = 0; i < N1; ++i) out.all_kpts1.emplace_back(acc[i][0], acc[i][1]);
  }

  // all_desc
  {
    const int N0 = (int)d0_cpu.size(0);
    const int D = (int)d0_cpu.size(1);
    out.all_desc0 = cv::Mat(N0, D, CV_32F);
    std::memcpy(out.all_desc0.data, d0_cpu.contiguous().data_ptr<float>(), (size_t)N0 * (size_t)D * sizeof(float));
  }
  {
    const int N1 = (int)d1_cpu.size(0);
    const int D = (int)d1_cpu.size(1);
    out.all_desc1 = cv::Mat(N1, D, CV_32F);
    std::memcpy(out.all_desc1.data, d1_cpu.contiguous().data_ptr<float>(), (size_t)N1 * (size_t)D * sizeof(float));
  }

  // matched_kpts
  {
    const int N = (int)mk0_cpu.size(0);
    out.matched_kpts0.reserve(N);
    out.matched_kpts1.reserve(N);
    auto a0 = mk0_cpu.accessor<float, 2>();
    auto a1 = mk1_cpu.accessor<float, 2>();
    for (int i = 0; i < N; ++i) {
      out.matched_kpts0.emplace_back(a0[i][0], a0[i][1]);
      out.matched_kpts1.emplace_back(a1[i][0], a1[i][1]);
    }
  }

  const auto t_r0 = Clock::now();
  if (out.matched_kpts0.size() >= 4) {
    std::vector<unsigned char> mask;
    out.H = cv::findHomography(out.matched_kpts0, out.matched_kpts1, cv::USAC_MAGSAC, 3.0, mask, 2000, 0.95);
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

}  // namespace dmw::liftfeat
