#pragma once

#include <array>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>

namespace clidd {

struct ImageFeatures {
    std::vector<std::array<float, 2>> keypoints;
    std::vector<float> scores;
    std::vector<float> descriptors;
    int descriptorDim = 0;
    int originalH = 0;
    int originalW = 0;
    float scaleX = 1.0f;
    float scaleY = 1.0f;
};

class CLIDDTRT {
public:
    struct MatchResult {
        std::vector<std::array<float, 2>> keypoints1;
        std::vector<std::array<float, 2>> keypoints2;
        std::vector<int64_t> matchIndices1;
        std::vector<int64_t> matchIndices2;
        int matchCount = 0;
        cv::Mat H;
        std::vector<std::array<float, 2>> inlier_kpts0;
        std::vector<std::array<float, 2>> inlier_kpts1;
        std::vector<unsigned char> inlier_mask;
        double ms_infer = 0.0;
        double ms_match = 0.0;
        double ms_ransac = 0.0;
    };

    CLIDDTRT(const std::string& enginePath,
              int topK = 2048,
              float scoreThresh = -5.0f,
              int radius = 2,
              int border = 4,
              float matchThresh = 0.01f,
              float beta = 20.0f);

    ~CLIDDTRT();

    MatchResult match(const cv::Mat& img1, const cv::Mat& img2);
    MatchResult matchFromPaths(const std::string& imagePath1, const std::string& imagePath2);

    int topK() const;
    float scoreThresh() const;
    int radius() const;
    int border() const;
    float matchThresh() const;
    float beta() const;
    int modelInputH() const;
    int modelInputW() const;
};

}  // namespace clidd
