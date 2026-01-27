#include "edm/EdmTorchMatcher.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
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
  std::cout << "demo_edm --img1 <path> --img2 <path> [options]\n";
  std::cout << "Options:\n";
  std::cout << "  --device cpu|cuda (default cpu)\n";
  std::cout << "  --dtype fp32 (TorchScript export)\n";
  std::cout << "  --weights <pt_path> (default matcher-cpp/edm/weights/edm_fp32_w640_h480_topk1680.pt)\n";
  std::cout << "  --w 640 --h 480 --topk 1680 (must match exported .pt)\n";
  std::cout << "  --local-res 8 (default 8)\n";
  std::cout << "  --mconf 0.2 (default 0.2)\n";
  std::cout << "  --border-rm 2 (default 2; multiplied by local-res)\n";
  std::cout << "  --sigma-select yes|no (default yes)\n";
  std::cout << "  --sigma-thr 0.0 (default 0.0)\n";
  std::cout << "  --output yes|no (default no)\n";
  std::cout << "  --out <jpg_path> (optional; overrides --output)\n";
  std::cout << "  --draw-all (draw outliers too)\n";
}

}  // namespace

int main(int argc, char** argv) {
  const std::string img1 = get_arg(argc, argv, "--img1");
  const std::string img2 = get_arg(argc, argv, "--img2");
  if (img1.empty() || img2.empty()) {
    usage();
    return 2;
  }

  dmw::edm::EdmConfig cfg;
  cfg.device = get_arg(argc, argv, "--device", "cpu");
  cfg.dtype = get_arg(argc, argv, "--dtype", "fp32");
  cfg.weights_path = get_arg(argc, argv, "--weights", cfg.weights_path);
  cfg.input_w = get_arg_i(argc, argv, "--w", cfg.input_w);
  cfg.input_h = get_arg_i(argc, argv, "--h", cfg.input_h);
  cfg.topk = get_arg_i(argc, argv, "--topk", cfg.topk);
  cfg.local_resolution = get_arg_i(argc, argv, "--local-res", cfg.local_resolution);
  cfg.mconf_thr = get_arg_f(argc, argv, "--mconf", cfg.mconf_thr);
  cfg.border_rm_coarse = get_arg_i(argc, argv, "--border-rm", cfg.border_rm_coarse);

  const std::string sigma_sel = get_arg(argc, argv, "--sigma-select", "yes");
  cfg.sigma_selection = (sigma_sel != "no");
  cfg.sigma_thr = get_arg_f(argc, argv, "--sigma-thr", cfg.sigma_thr);

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

  dmw::edm::EdmTorchMatcher matcher(cfg);

  // Warmup.
  (void)matcher.match(im0, im1);
  auto t0 = std::chrono::steady_clock::now();
  auto res = matcher.match(im0, im1);
  auto t1 = std::chrono::steady_clock::now();
  const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  const int n_matches = (int)res.matched_kpts0.size();
  const int n_inliers = (int)res.inlier_kpts0.size();
  const double ratio = n_matches > 0 ? (double)n_inliers / (double)n_matches : 0.0;

  std::cout << "\nResults:\n";
  std::cout << "  Matched Keypoints: " << n_matches << "\n";
  std::cout << "  Inliers: " << n_inliers << "\n";
  std::cout << "  Ratio: " << ratio << "\n";
  std::cout << "\nTiming (ms): preprocess=" << res.ms_preprocess << " infer=" << res.ms_infer
            << " post=" << res.ms_postprocess << " ransac=" << res.ms_ransac << "\n";
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
    const std::string folder = "edm_" + cfg.dtype + "_" + stem1 + "_" + stem2;
    out_dir = std::filesystem::path("outputs") / "matching-cpp" / folder;
    out_jpg = out_dir / "result.jpg";
    out_txt = out_dir / "result.txt";
  }

  if (!out_jpg.empty()) {
    std::filesystem::create_directories(out_dir);

    cv::Mat vis;
    cv::hconcat(im0, im1, vis);
    const int off = im0.cols;

    for (int i = 0; i < n_matches; ++i) {
      bool is_inlier = false;
      if (!res.inlier_mask.empty() && i < (int)res.inlier_mask.size()) is_inlier = res.inlier_mask[(size_t)i] != 0;
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
    f << "model: edm\n";
    f << "device: " << cfg.device << "\n";
    f << "dtype: " << cfg.dtype << "\n";
    f << "weights: " << cfg.weights_path << "\n";
    f << "input_w: " << cfg.input_w << "\n";
    f << "input_h: " << cfg.input_h << "\n";
    f << "topk: " << cfg.topk << "\n";
    f << "local_resolution: " << cfg.local_resolution << "\n";
    f << "mconf_thr: " << cfg.mconf_thr << "\n";
    f << "border_rm_coarse: " << cfg.border_rm_coarse << "\n";
    f << "sigma_selection: " << (cfg.sigma_selection ? "yes" : "no") << "\n";
    f << "sigma_thr: " << cfg.sigma_thr << "\n";
    f << "num_matches: " << n_matches << "\n";
    f << "num_inliers: " << n_inliers << "\n";
    f << "ratio: " << ratio << "\n";
    f << "ms_preprocess: " << res.ms_preprocess << "\n";
    f << "ms_infer: " << res.ms_infer << "\n";
    f << "ms_postprocess: " << res.ms_postprocess << "\n";
    f << "ms_ransac: " << res.ms_ransac << "\n";
    f << "time_total_ms: " << total_ms << "\n";
    f.close();

    std::cout << "Saved output to " << out_dir.string() << "\n";
  }

  return 0;
}
