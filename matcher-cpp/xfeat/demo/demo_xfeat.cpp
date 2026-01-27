#include "xfeat/XFeatTorchMatcher.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <torch/torch.h>

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>

namespace {

std::string get_arg(int argc, char** argv, const std::string& key, const std::string& def = "") {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == key) return argv[i + 1];
  }
  return def;
}

bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == key) return true;
  }
  return false;
}

int get_arg_i(int argc, char** argv, const std::string& key, int def) {
  auto v = get_arg(argc, argv, key, "");
  if (v.empty()) return def;
  return std::stoi(v);
}

float get_arg_f(int argc, char** argv, const std::string& key, float def) {
  auto v = get_arg(argc, argv, key, "");
  if (v.empty()) return def;
  return std::stof(v);
}

void usage() {
  std::cout << "demo_xfeat --img1 <path> --img2 <path> [options]\n";
  std::cout << "Options:\n";
  std::cout << "  --mode xfeat|xfeat-star|xfeat-lightglue (default xfeat)\n";
  std::cout << "  --device cpu|cuda (default cpu)\n";
  std::cout << "  --dtype fp32 (TorchScript export)\n";
  std::cout << "  --topk 4096 (must match exported .pt)\n";
  std::cout << "  --detection-threshold 0.05 (for sparse detection)\n";
  std::cout << "  --fine-conf 0.25 (for xfeat-star refinement)\n";
  std::cout << "  --min-cossim -1 (for xfeat MNN matching; -1 disables threshold)\n";
  std::cout << "  --min-match-conf 0.1 (for xfeat-lightglue matching)\n";
  std::cout << "  --output yes|no (default no)\n";
  std::cout << "  --out <jpg_path> (optional; overrides --output)\n";
  std::cout << "  --draw-all (draw outliers too)\n";
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode_str = get_arg(argc, argv, "--mode", "xfeat");
  const std::string img1 = get_arg(argc, argv, "--img1");
  const std::string img2 = get_arg(argc, argv, "--img2");
  if (img1.empty() || img2.empty()) {
    usage();
    return 2;
  }

  dmw::xfeat::XFeatConfig cfg;
  
  // Parse mode
  if (mode_str == "xfeat") {
    cfg.mode = dmw::xfeat::XFeatMode::XFEAT;
  } else if (mode_str == "xfeat-star") {
    cfg.mode = dmw::xfeat::XFeatMode::XFEAT_STAR;
  } else if (mode_str == "xfeat-lightglue") {
    cfg.mode = dmw::xfeat::XFeatMode::XFEAT_LIGHTGLUE;
  } else {
    std::cerr << "Unknown mode: " << mode_str << "\n";
    return 2;
  }
  
  cfg.device = get_arg(argc, argv, "--device", "cpu");
  cfg.dtype = get_arg(argc, argv, "--dtype", "fp32");
  cfg.top_k = get_arg_i(argc, argv, "--topk", 4096);
  cfg.detection_threshold = get_arg_f(argc, argv, "--detection-threshold", 0.05f);
  cfg.fine_conf = get_arg_f(argc, argv, "--fine-conf", 0.25f);
  cfg.min_cossim = get_arg_f(argc, argv, "--min-cossim", -1.0f);
  cfg.min_match_conf = get_arg_f(argc, argv, "--min-match-conf", 0.1f);
  
  const bool draw_all = has_flag(argc, argv, "--draw-all");
  const std::string out_path = get_arg(argc, argv, "--out", "");
  const std::string output_flag = get_arg(argc, argv, "--output", "no");
  const bool output_enabled = (output_flag == "yes");

  cv::Mat im0 = cv::imread(img1, cv::IMREAD_COLOR);
  cv::Mat im1 = cv::imread(img2, cv::IMREAD_COLOR);
  if (im0.empty() || im1.empty()) {
    std::cerr << "Failed to read images\n";
    return 2;
  }

  dmw::xfeat::XFeatTorchMatcher matcher(cfg);
  
  // Warmup
  (void)matcher.match(im0, im1);
  if (cfg.device == "cuda") {
    torch::cuda::synchronize();
  }

  auto t0 = std::chrono::steady_clock::now();
  auto res = matcher.match(im0, im1);
  if (cfg.device == "cuda") {
    torch::cuda::synchronize();
  }
  auto t1 = std::chrono::steady_clock::now();
  const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  const int n_matches = (int)res.matched_kpts0.size();
  const int n_inliers = (int)res.inlier_kpts0.size();
  const double ratio = n_matches > 0 ? (double)n_inliers / (double)n_matches : 0.0;

  std::cout << "\nResults:\n";
  std::cout << "  Mode: " << mode_str << "\n";
  std::cout << "  Total Keypoints0: " << res.all_kpts0.size() << "\n";
  std::cout << "  Total Keypoints1: " << res.all_kpts1.size() << "\n";
  std::cout << "  Matched Keypoints: " << n_matches << "\n";
  std::cout << "  Inliers: " << n_inliers << "\n";
  std::cout << "  Ratio: " << ratio << "\n";
  std::cout << "\nTiming (ms): preprocess=" << res.ms_preprocess << " infer(per-img)=" << res.ms_infer
            << " match=" << res.ms_match << " ransac=" << res.ms_ransac << "\n";
  std::cout << "Time: " << total_ms << " ms\n";

  std::filesystem::path out_dir;
  std::filesystem::path out_jpg;
  std::filesystem::path out_txt;

  if (!out_path.empty()) {
    out_jpg = std::filesystem::path(out_path);
    out_dir = out_jpg.parent_path();
    out_txt = out_dir / "result.txt";
  } else if (output_enabled) {
    const std::string stem1 = std::filesystem::path(img1).stem().string();
    const std::string stem2 = std::filesystem::path(img2).stem().string();
    const std::string folder = mode_str + "_" + cfg.dtype + "_" + stem1 + "_" + stem2;
    out_dir = std::filesystem::path("outputs") / "matching-cpp" / folder;
    out_jpg = out_dir / "result.jpg";
    out_txt = out_dir / "result.txt";
  }

  if (!out_jpg.empty()) {
    std::filesystem::create_directories(out_dir);

    // Draw matches.
    cv::Mat vis;
    cv::hconcat(im0, im1, vis);
    const int off = im0.cols;

    for (int i = 0; i < n_matches; ++i) {
      bool is_inlier = false;
      if (!res.inlier_mask.empty() && i < (int)res.inlier_mask.size()) {
        is_inlier = res.inlier_mask[(size_t)i] != 0;
      }
      if (!draw_all && !is_inlier) continue;
      cv::Scalar color = is_inlier ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
      auto p0 = res.matched_kpts0[(size_t)i];
      auto p1 = res.matched_kpts1[(size_t)i];
      cv::Point pt0((int)std::round(p0.x), (int)std::round(p0.y));
      cv::Point pt1((int)std::round(p1.x + off), (int)std::round(p1.y));
      cv::line(vis, pt0, pt1, color, 1, cv::LINE_AA);
      cv::circle(vis, pt0, 2, cv::Scalar(255, 0, 0), -1);
      cv::circle(vis, pt1, 2, cv::Scalar(255, 0, 0), -1);
    }

    cv::imwrite(out_jpg.string(), vis);

    std::ofstream f(out_txt.string());
    f << "mode: " << mode_str << "\n";
    f << "device: " << cfg.device << "\n";
    f << "dtype: " << cfg.dtype << "\n";
    f << "topk: " << cfg.top_k << "\n";
    f << "num_kpts0: " << res.all_kpts0.size() << "\n";
    f << "num_kpts1: " << res.all_kpts1.size() << "\n";
    f << "num_matches: " << n_matches << "\n";
    f << "num_inliers: " << n_inliers << "\n";
    f << "ratio: " << ratio << "\n";
    f << "ms_infer_per_img: " << res.ms_infer << "\n";
    f << "ms_match: " << res.ms_match << "\n";
    f << "ms_ransac: " << res.ms_ransac << "\n";
    f << "time_total_ms: " << total_ms << "\n";
    f.close();

    std::cout << "Saved output to " << out_dir.string() << "\n";
  }

  return 0;
}
