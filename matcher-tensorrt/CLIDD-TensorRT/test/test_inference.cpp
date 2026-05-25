#include <cassert>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "../src/clidd_trt.cpp"

namespace fs = std::filesystem;

namespace {

struct Config {
    std::string enginePath = "weights/clidd_m64_640x480_fp16.engine";
    std::string imgPath1 = "test/assets/ref.png";
    std::string imgPath2 = "test/assets/tgt.png";
    std::string outPath = "matches_visualization_cpp.png";
    int topK = 2048;
    float scoreThresh = -5.0f;
    int radius = 2;
    int border = 4;
    float matchThresh = 0.01f;
    float beta = 20.0f;
    int loop = 1;
    bool inliersOnly = false;
};

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --img1 <path>       First image (default: test/assets/ref.png)" << std::endl;
    std::cout << "  --img2 <path>       Second image (default: test/assets/tgt.png)" << std::endl;
    std::cout << "  --engine <path>     TensorRT engine path (required)" << std::endl;
    std::cout << "  --top-k <int>       Max keypoints (default: 2048)" << std::endl;
    std::cout << "  --score-thresh <f>  Score threshold (default: -5.0)" << std::endl;
    std::cout << "  --radius <int>      NMS radius (default: 2)" << std::endl;
    std::cout << "  --border <int>      Border suppression (default: 4)" << std::endl;
    std::cout << "  --match-thresh <f>  Match threshold (default: 0.01)" << std::endl;
    std::cout << "  --beta <f>          Matching beta (default: 20.0)" << std::endl;
    std::cout << "  --out <path>        Output visualization path" << std::endl;
    std::cout << "  --loop <int>        Number of loops for benchmarking (default: 1)" << std::endl;
    std::cout << "  --inliers-only      Draw only inliers" << std::endl;
    std::cout << "  --help, -h          Show this help" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << prog << " --engine weights/clidd_m64_640x480_fp16.engine --img1 test/assets/ref.png --img2 test/assets/tgt.png" << std::endl;
}

Config parseArgs(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--img1") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --img1");
            }
            cfg.imgPath1 = argv[++i];
        } else if (arg == "--img2") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --img2");
            }
            cfg.imgPath2 = argv[++i];
        } else if (arg == "--engine") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --engine");
            }
            cfg.enginePath = argv[++i];
        } else if (arg == "--top-k") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --top-k");
            }
            cfg.topK = std::stoi(argv[++i]);
        } else if (arg == "--score-thresh") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --score-thresh");
            }
            cfg.scoreThresh = std::stof(argv[++i]);
        } else if (arg == "--radius") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --radius");
            }
            cfg.radius = std::stoi(argv[++i]);
        } else if (arg == "--border") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --border");
            }
            cfg.border = std::stoi(argv[++i]);
        } else if (arg == "--match-thresh") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --match-thresh");
            }
            cfg.matchThresh = std::stof(argv[++i]);
        } else if (arg == "--beta") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --beta");
            }
            cfg.beta = std::stof(argv[++i]);
        } else if (arg == "--out") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --out");
            }
            cfg.outPath = argv[++i];
        } else if (arg == "--loop") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --loop");
            }
            cfg.loop = std::stoi(argv[++i]);
        } else if (arg == "--inliers-only") {
            cfg.inliersOnly = true;
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    return cfg;
}

void requireFileExists(const std::string& path, const std::string& label) {
    if (!fs::exists(path)) {
        throw std::runtime_error(label + " not found: " + path);
    }
}

cv::Mat ensureBgr(const cv::Mat& img) {
    if (img.channels() == 3) {
        return img.clone();
    }
    cv::Mat out;
    if (img.channels() == 1) {
        cv::cvtColor(img, out, cv::COLOR_GRAY2BGR);
        return out;
    }
    throw std::runtime_error("Unsupported image channel count for visualization");
}

}

int main(int argc, char** argv) {
    try {
        const Config cfg = parseArgs(argc, argv);

        using clock = std::chrono::high_resolution_clock;
        const auto tTotalStart = clock::now();

        std::cout << "=== CLIDD TRT Inference Test (C++) ===" << std::endl;
        requireFileExists(cfg.enginePath, "Engine file");
        requireFileExists(cfg.imgPath1, "Image 1");
        requireFileExists(cfg.imgPath2, "Image 2");

        std::cout << "Input images:" << std::endl;
        std::cout << "  - img1: " << cfg.imgPath1 << std::endl;
        std::cout << "  - img2: " << cfg.imgPath2 << std::endl;
        std::cout << "Output visualization: " << cfg.outPath << std::endl;

        std::cout << "\nInstantiating CLIDDTRT..." << std::endl;
        clidd::CLIDDTRT matcher(
            cfg.enginePath,
            cfg.topK,
            cfg.scoreThresh,
            cfg.radius,
            cfg.border,
            cfg.matchThresh,
            cfg.beta);
        std::cout << "  - topK: " << matcher.topK() << std::endl;
        std::cout << "  - scoreThresh: " << matcher.scoreThresh() << std::endl;
        std::cout << "  - radius: " << matcher.radius() << std::endl;
        std::cout << "  - border: " << matcher.border() << std::endl;
        std::cout << "  - matchThresh: " << matcher.matchThresh() << std::endl;
        std::cout << "  - beta: " << matcher.beta() << std::endl;
        std::cout << "  - loop: " << cfg.loop << std::endl;

        const auto tLoadStart = clock::now();
        cv::Mat img1 = cv::imread(cfg.imgPath1, cv::IMREAD_COLOR);
        cv::Mat img2 = cv::imread(cfg.imgPath2, cv::IMREAD_COLOR);
        const auto tLoadEnd = clock::now();
        if (img1.empty() || img2.empty()) {
            throw std::runtime_error("Failed to load one or both images");
        }

        std::cout << "\nRunning inference..." << std::endl;
        
        clidd::CLIDDTRT::MatchResult result;
        double msMatch = 0.0;
        
        if (cfg.loop > 1) {
            for (int w = 0; w < std::min(3, cfg.loop); ++w) {
                matcher.match(img1, img2);
            }
            
            std::vector<double> matchTimes;
            matchTimes.reserve(static_cast<size_t>(cfg.loop));
            
            for (int i = 0; i < cfg.loop; ++i) {
                const auto tMatchStart = clock::now();
                result = matcher.match(img1, img2);
                const auto tMatchEnd = clock::now();
                
                double ms = std::chrono::duration<double, std::milli>(tMatchEnd - tMatchStart).count();
                matchTimes.push_back(ms);
                
                if (i == 0) {
                    std::cout << "  First iteration match count: " << result.matchCount << std::endl;
                }
            }
            
            double sum = 0.0;
            double minTime = matchTimes[0];
            double maxTime = matchTimes[0];
            for (const double t : matchTimes) {
                sum += t;
                minTime = std::min(minTime, t);
                maxTime = std::max(maxTime, t);
            }
            msMatch = sum / static_cast<double>(matchTimes.size());
            
            std::cout << "\n=== Benchmark Results (" << cfg.loop << " loops) ===" << std::endl;
            std::cout << "  Min time:  " << minTime << " ms" << std::endl;
            std::cout << "  Max time:  " << maxTime << " ms" << std::endl;
            std::cout << "  Avg time:  " << msMatch << " ms" << std::endl;
        } else {
            const auto tMatchStart = clock::now();
            result = matcher.match(img1, img2);
            const auto tMatchEnd = clock::now();
            
            msMatch = std::chrono::duration<double, std::milli>(tMatchEnd - tMatchStart).count();
            std::cout << "  Match time: " << msMatch << " ms" << std::endl;
        }

        const int matchCount = result.matchCount;

        std::vector<cv::Point2f> matchedPts1;
        std::vector<cv::Point2f> matchedPts2;
        matchedPts1.reserve(static_cast<size_t>(matchCount));
        matchedPts2.reserve(static_cast<size_t>(matchCount));

        for (int i = 0; i < matchCount; ++i) {
            const int64_t idx1 = result.matchIndices1[static_cast<size_t>(i)];
            const int64_t idx2 = result.matchIndices2[static_cast<size_t>(i)];
            if (idx1 < 0 || idx2 < 0) {
                continue;
            }
            if (idx1 >= static_cast<int64_t>(result.keypoints1.size()) ||
                idx2 >= static_cast<int64_t>(result.keypoints2.size())) {
                continue;
            }

            const auto& p1 = result.keypoints1[static_cast<size_t>(idx1)];
            const auto& p2 = result.keypoints2[static_cast<size_t>(idx2)];
            matchedPts1.emplace_back(p1[0], p1[1]);
            matchedPts2.emplace_back(p2[0], p2[1]);
        }

        const auto tRansacStart = clock::now();
        cv::Mat inlierMask;
        cv::Mat homography;
        if (matchedPts1.size() >= 4U) {
            homography = cv::findHomography(
                matchedPts1,
                matchedPts2,
                cv::RANSAC,
                3.0,
                inlierMask);
        }
        const auto tRansacEnd = clock::now();

        int inlierCount = cv::countNonZero(inlierMask);

        const auto tVisStart = clock::now();
        cv::Mat left = ensureBgr(img1);
        cv::Mat right = ensureBgr(img2);
        cv::Mat canvas;
        cv::hconcat(left, right, canvas);

        for (size_t i = 0; i < matchedPts1.size(); ++i) {
            const cv::Point2f p1 = matchedPts1[i];
            const cv::Point2f p2Shifted = cv::Point2f(matchedPts2[i].x + static_cast<float>(left.cols), matchedPts2[i].y);
            const bool isInlier = i < static_cast<size_t>(inlierMask.rows) && inlierMask.at<uchar>(static_cast<int>(i), 0) != 0U;
            
            if (cfg.inliersOnly && !isInlier) continue;
            
            const cv::Scalar color = isInlier ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 0, 255);

            cv::line(canvas, p1, p2Shifted, color, 1, cv::LINE_AA);
            cv::circle(canvas, p1, 2, color, -1, cv::LINE_AA);
            cv::circle(canvas, p2Shifted, 2, color, -1, cv::LINE_AA);
        }

        const bool saved = cv::imwrite(cfg.outPath, canvas);
        const auto tVisEnd = clock::now();
        if (!saved) {
            throw std::runtime_error("Failed to save visualization to: " + cfg.outPath);
        }

        const auto tTotalEnd = clock::now();

        const double msLoad = std::chrono::duration<double, std::milli>(tLoadEnd - tLoadStart).count();
        const double msRansac = std::chrono::duration<double, std::milli>(tRansacEnd - tRansacStart).count();
        const double msVis = std::chrono::duration<double, std::milli>(tVisEnd - tVisStart).count();
        const double msTotal = std::chrono::duration<double, std::milli>(tTotalEnd - tTotalStart).count();
        const double msInferencePerImageEstimate = msMatch / 2.0;
        const double inlierRatio = matchedPts1.empty() ? 0.0 : static_cast<double>(inlierCount) / static_cast<double>(matchedPts1.size());

        std::cout << "\n=== Metrics ===" << std::endl;
        std::cout << std::left << std::setw(36) << "metric" << std::right << std::setw(14) << "value" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        std::cout << std::left << std::setw(36) << "keypoints image1" << std::right << std::setw(14) << result.keypoints1.size() << std::endl;
        std::cout << std::left << std::setw(36) << "keypoints image2" << std::right << std::setw(14) << result.keypoints2.size() << std::endl;
        std::cout << std::left << std::setw(36) << "raw matches (matcher)" << std::right << std::setw(14) << matchCount << std::endl;
        std::cout << std::left << std::setw(36) << "valid correspondences (indexed)" << std::right << std::setw(14) << matchedPts1.size() << std::endl;
        std::cout << std::left << std::setw(36) << "RANSAC inliers" << std::right << std::setw(14) << inlierCount << std::endl;
        std::cout << std::left << std::setw(36) << "RANSAC inlier ratio" << std::right << std::setw(13) << std::fixed << std::setprecision(3) << inlierRatio << std::endl;
        std::cout << std::left << std::setw(36) << "load images (ms)" << std::right << std::setw(14) << std::setprecision(3) << msLoad << std::endl;
        std::cout << std::left << std::setw(36) << "pair inference+matching (ms)" << std::right << std::setw(14) << msMatch << std::endl;
        std::cout << std::left << std::setw(36) << "est. inference per image (ms)" << std::right << std::setw(14) << msInferencePerImageEstimate << std::endl;
        std::cout << std::left << std::setw(36) << "RANSAC time (ms)" << std::right << std::setw(14) << msRansac << std::endl;
        std::cout << std::left << std::setw(36) << "visualization save time (ms)" << std::right << std::setw(14) << msVis << std::endl;
        std::cout << std::left << std::setw(36) << "total runtime (ms)" << std::right << std::setw(14) << msTotal << std::endl;
        std::cout << std::left << std::setw(36) << "homography computed" << std::right << std::setw(14) << (!homography.empty() ? "yes" : "no") << std::endl;

        std::cout << "\nSaved visualization: " << cfg.outPath << std::endl;
        std::cout << "\n=== TEST PASSED ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
