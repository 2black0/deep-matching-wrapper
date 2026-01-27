#include "LiftFeatOnnxMatcher.h"
#include <iostream>
#include <iomanip>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace dmw::liftfeat_onnx;

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " --img1 <path> --img2 <path> [options]\n"
              << "Options:\n"
              << "  --img1 <path>      First image path (required)\n"
              << "  --img2 <path>      Second image path (required)\n"
              << "  --device <dev>     Device: 'cpu' or 'cuda' (default: cuda)\n"
              << "  --dtype <type>     Data type: 'fp32' or 'fp16' (default: fp32)\n"
              << "  --width <w>        Resize width (default: 640)\n"
              << "  --height <h>       Resize height (default: 480)\n"
              << "  --top_k <k>        Top K keypoints (default: 4096)\n"
              << "  --threshold <t>    Detection threshold (default: 0.005)\n"
              << "  --min_cossim <c>   Min cosine similarity (default: -1.0)\n"
              << "  --weights <path>   Path to ONNX weights (optional)\n"
              << "  --output <path>    Save visualization (optional)\n";
}

int main(int argc, char** argv) {
    std::string img1_path, img2_path, output_path;
    LiftFeatConfig config;
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--img1" && i + 1 < argc) {
            img1_path = argv[++i];
        } else if (arg == "--img2" && i + 1 < argc) {
            img2_path = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            config.device = argv[++i];
        } else if (arg == "--dtype" && i + 1 < argc) {
            config.dtype = argv[++i];
        } else if (arg == "--width" && i + 1 < argc) {
            config.width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            config.height = std::stoi(argv[++i]);
        } else if (arg == "--top_k" && i + 1 < argc) {
            config.top_k = std::stoi(argv[++i]);
        } else if (arg == "--threshold" && i + 1 < argc) {
            config.detect_threshold = std::stof(argv[++i]);
        } else if (arg == "--min_cossim" && i + 1 < argc) {
            config.min_cossim = std::stof(argv[++i]);
        } else if (arg == "--weights" && i + 1 < argc) {
            config.weights_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    if (img1_path.empty() || img2_path.empty()) {
        std::cerr << "Error: --img1 and --img2 are required\n";
        print_usage(argv[0]);
        return 1;
    }
    
    // Load images
    cv::Mat img0 = cv::imread(img1_path);
    cv::Mat img1 = cv::imread(img2_path);
    
    if (img0.empty() || img1.empty()) {
        std::cerr << "Error: Could not load images\n";
        return 1;
    }
    
    try {
        // Create matcher
        LiftFeatOnnxMatcher matcher(config);
        
        // Match
        auto result = matcher.match(img0, img1);
        
        // Print results
        std::cout << "\nResults:\n"
                  << "  Total Keypoints0: " << result.kpts0.size() << "\n"
                  << "  Total Keypoints1: " << result.kpts1.size() << "\n"
                  << "  Matched Keypoints: " << result.mkpts0.size() << "\n";
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\nTiming (ms): infer(per-img)=" << result.ms_infer
                  << " postprocess=" << result.ms_postprocess
                  << " match=" << result.ms_match << "\n";
        std::cout << "Time: " << result.ms_total << " ms\n";
        
        // Visualize if output path provided
        if (!output_path.empty()) {
            // Create side-by-side image
            int h = std::max(img0.rows, img1.rows);
            int w = img0.cols + img1.cols;
            cv::Mat vis(h, w, CV_8UC3, cv::Scalar(0, 0, 0));
            
            img0.copyTo(vis(cv::Rect(0, 0, img0.cols, img0.rows)));
            img1.copyTo(vis(cv::Rect(img0.cols, 0, img1.cols, img1.rows)));
            
            // Draw matches
            for (size_t i = 0; i < result.mkpts0.size(); ++i) {
                cv::Point2f pt0 = result.mkpts0[i];
                cv::Point2f pt1 = result.mkpts1[i];
                pt1.x += img0.cols;
                
                cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
                cv::circle(vis, pt0, 3, color, -1);
                cv::circle(vis, pt1, 3, color, -1);
                cv::line(vis, pt0, pt1, color, 1);
            }
            
            cv::imwrite(output_path, vis);
            std::cout << "\nVisualization saved to: " << output_path << "\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
