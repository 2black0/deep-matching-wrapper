#include "lightglue/SuperPointLightGlueMatcher.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
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
  std::cout << "demo_superpoint_lightglue --img1 <path> --img2 <path> [options]\n";
  std::cout << "Options:\n";
  std::cout << "  --device cpu|cuda (default cpu)\n";
  std::cout << "  --dtype fp32\n";
  std::cout << "  --max-kpts 2048 (default 2048)\n";
  std::cout << "  --detection-thr 0.005 (default 0.005)\n";
  std::cout << "  --nms-radius 4 (default 4)\n";
  std::cout << "  --remove-borders 4 (default 4)\n";
  std::cout << "  --match-thr 0.0 (default 0.0; log-score threshold)\n";
  std::cout << "  --output yes|no (default no)\n";
  std::cout << "  --out <jpg_path> (optional; overrides --output)\n";
  std::cout << "  --draw-all (draw all matches, not just inliers)\n";
}

}  // namespace

int main(int argc, char** argv) {
  const std::string img1 = get_arg(argc, argv, "--img1");
  const std::string img2 = get_arg(argc, argv, "--img2");
  if (img1.empty() || img2.empty()) {
    usage();
    return 2;
  }

  dmw::lightglue::SuperPointLightGlueConfig cfg;
  cfg.device = get_arg(argc, argv, "--device", "cpu");
  cfg.dtype = get_arg(argc, argv, "--dtype", "fp32");
  cfg.max_num_keypoints = get_arg_i(argc, argv, "--max-kpts", cfg.max_num_keypoints);
  cfg.detection_threshold = get_arg_f(argc, argv, "--detection-thr", cfg.detection_threshold);
  cfg.nms_radius = get_arg_i(argc, argv, "--nms-radius", cfg.nms_radius);
  cfg.remove_borders = get_arg_i(argc, argv, "--remove-borders", cfg.remove_borders);
  cfg.match_threshold = get_arg_f(argc, argv, "--match-thr", cfg.match_threshold);

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

  dmw::lightglue::SuperPointLightGlueMatcher matcher(cfg);

  // Warmup
  (void)matcher.match(im0, im1);
  auto t0 = std::chrono::steady_clock::now();
  auto res = matcher.match(im0, im1);
  auto t1 = std::chrono::steady_clock::now();
  const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  const int n_all_kpts0 = (int)res.all_kpts0.size();
  const int n_all_kpts1 = (int)res.all_kpts1.size();
  const int n_matches = (int)res.matched_kpts0.size();
  const int n_inliers = (int)res.inlier_kpts0.size();
  const double ratio = n_matches > 0 ? (double)n_inliers / (double)n_matches : 0.0;

  std::cout << "\nResults:\n";
  std::cout << "  Total Keypoints0: " << n_all_kpts0 << "\n";
  std::cout << "  Total Keypoints1: " << n_all_kpts1 << "\n";
  std::cout << "  Matched Keypoints: " << n_matches << "\n";
  std::cout << "  Inliers: " << n_inliers << "\n";
  std::cout << "  Ratio: " << ratio << "\n";
  std::cout << "\nTiming (ms): superpoint0=" << res.ms_superpoint0
            << " superpoint1=" << res.ms_superpoint1
            << " lightglue=" << res.ms_lightglue
            << " ransac=" << res.ms_ransac << "\n";
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
    const std::string folder = "superpoint_lightglue_" + cfg.dtype + "_" + stem1 + "_" + stem2;
    out_dir = std::filesystem::path("outputs") / "matching-cpp" / folder;
    out_jpg = out_dir / "result.jpg";
    out_txt = out_dir / "result.txt";
  }

  if (!out_jpg.empty()) {
    std::filesystem::create_directories(out_dir);

    cv::Mat vis;
    cv::hconcat(im0, im1, vis);
    const int off = im0.cols;

    // Draw matches
    const auto& draw_kpts0 = draw_all ? res.matched_kpts0 : res.inlier_kpts0;
    const auto& draw_kpts1 = draw_all ? res.matched_kpts1 : res.inlier_kpts1;
    
    for (size_t i = 0; i < draw_kpts0.size(); ++i) {
      cv::Scalar color = cv::Scalar(0, 255, 0);  // Green for inliers
      if (draw_all && !res.inlier_mask.empty() && i < res.inlier_mask.size()) {
        color = res.inlier_mask[i] ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
      }
      
      auto p0 = draw_kpts0[i];
      auto p1 = draw_kpts1[i];
      cv::Point pt0((int)std::round(p0.x), (int)std::round(p0.y));
      cv::Point pt1((int)std::round(p1.x + off), (int)std::round(p1.y));
      cv::line(vis, pt0, pt1, color, 1, cv::LINE_AA);
      cv::circle(vis, pt0, 2, cv::Scalar(255, 0, 0), -1);
      cv::circle(vis, pt1, 2, cv::Scalar(255, 0, 0), -1);
    }

    cv::imwrite(out_jpg.string(), vis);

    std::ofstream f(out_txt.string());
    f << "model: superpoint-lightglue\n";
    f << "device: " << cfg.device << "\n";
    f << "dtype: " << cfg.dtype << "\n";
    f << "max_num_keypoints: " << cfg.max_num_keypoints << "\n";
    f << "detection_threshold: " << cfg.detection_threshold << "\n";
    f << "nms_radius: " << cfg.nms_radius << "\n";
    f << "remove_borders: " << cfg.remove_borders << "\n";
    f << "match_threshold: " << cfg.match_threshold << "\n";
    f << "num_all_kpts0: " << n_all_kpts0 << "\n";
    f << "num_all_kpts1: " << n_all_kpts1 << "\n";
    f << "num_matches: " << n_matches << "\n";
    f << "num_inliers: " << n_inliers << "\n";
    f << "ratio: " << ratio << "\n";
    f << "ms_superpoint0: " << res.ms_superpoint0 << "\n";
    f << "ms_superpoint1: " << res.ms_superpoint1 << "\n";
    f << "ms_lightglue: " << res.ms_lightglue << "\n";
    f << "ms_ransac: " << res.ms_ransac << "\n";
    f << "time_total_ms: " << total_ms << "\n";
    f.close();

    std::cout << "Saved output to " << out_dir.string() << "\n";
  }

  return 0;
}
