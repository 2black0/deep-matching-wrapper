#include "clidd_trt.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace clidd {

using Clock = std::chrono::steady_clock;

double ms_since(const Clock::time_point& t0, const Clock::time_point& t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static PreprocessOutput preprocessImage(const cv::Mat& img, int targetH, int targetW) {
    int originalH = img.rows;
    int originalW = img.cols;

    // Pad to multiple of 32 (like Python implementation)
    int paddedH = ((originalH + 31) / 32) * 32;
    int paddedW = ((originalW + 31) / 32) * 32;

    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    
    // Normalize to [0, 1]
    cv::Mat normalized;
    rgb.convertTo(normalized, CV_32F, 1.0 / 255.0);
    
    // Pad if necessary
    cv::Mat padded;
    if (paddedH != originalH || paddedW != originalW) {
        cv::copyMakeBorder(normalized, padded, 
                          0, paddedH - originalH,
                          0, paddedW - originalW,
                          cv::BORDER_CONSTANT, cv::Scalar(0));
    } else {
        padded = normalized;
    }
    
    // Reorder to (C, H, W) format
    std::vector<cv::Mat> channels;
    cv::split(padded, channels);
    
    std::vector<float> input(3LL * paddedH * paddedW);
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < paddedH; ++h) {
            for (int w = 0; w < paddedW; ++w) {
                input[static_cast<size_t>(c) * paddedH * paddedW + static_cast<size_t>(h) * paddedW + static_cast<size_t>(w)] =
                    channels[c].at<float>(h, w);
            }
        }
    }

    return {std::move(input), originalH, originalW, paddedH, paddedW};
}
        }
    }

    return {std::move(input), originalH, originalW, targetH, targetW};
}

static std::pair<std::vector<int64_t>, std::vector<int64_t>> matchFeaturesCPU(
    const ImageFeatures& feat1,
    const ImageFeatures& feat2,
    float matchThresh,
    float beta) {
    
    const int n1 = static_cast<int>(feat1.keypoints.size());
    const int n2 = static_cast<int>(feat2.keypoints.size());
    const int dim = feat1.descriptorDim;

    if (n1 == 0 || n2 == 0 || dim == 0) {
        return {{}, {}};
    }

    std::vector<float> distMatrix(static_cast<size_t>(n1) * static_cast<size_t>(n2));
    
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < dim; ++d) {
                dot += feat1.descriptors[static_cast<size_t>(i) * dim + d] * 
                       feat2.descriptors[static_cast<size_t>(j) * dim + d];
            }
            distMatrix[static_cast<size_t>(i) * n2 + j] = dot;
        }
    }

    std::vector<float> expMatrix(static_cast<size_t>(n1) * static_cast<size_t>(n2));
    for (size_t i = 0; i < distMatrix.size(); ++i) {
        expMatrix[i] = std::exp((distMatrix[i] - 1.0f) * beta);
    }

    std::vector<float> sum1(n1, 0.0f), sum2(n2, 0.0f);
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            float val = expMatrix[static_cast<size_t>(i) * n2 + j];
            sum1[i] += val;
            sum2[j] += val;
        }
    }

    std::vector<float> simMatrix(static_cast<size_t>(n1) * static_cast<size_t>(n2));
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            float val = expMatrix[static_cast<size_t>(i) * n2 + j];
            float sim = (val * val) / (sum1[i] * sum2[j] + 1e-12f);
            simMatrix[static_cast<size_t>(i) * n2 + j] = sim;
        }
    }

    std::vector<int64_t> nn12(n1);
    for (int i = 0; i < n1; ++i) {
        float maxSim = simMatrix[static_cast<size_t>(i) * n2];
        int maxIdx = 0;
        for (int j = 1; j < n2; ++j) {
            float val = simMatrix[static_cast<size_t>(i) * n2 + j];
            if (val > maxSim) {
                maxSim = val;
                maxIdx = j;
            }
        }
        nn12[i] = maxIdx;
    }

    std::vector<int64_t> nn21(n2);
    for (int j = 0; j < n2; ++j) {
        float maxSim = simMatrix[j];
        int maxIdx = 0;
        for (int i = 1; i < n1; ++i) {
            float val = simMatrix[static_cast<size_t>(i) * n2 + j];
            if (val > maxSim) {
                maxSim = val;
                maxIdx = i;
            }
        }
        nn21[j] = maxIdx;
    }

    std::vector<int64_t> matchIdx1;
    std::vector<int64_t> matchIdx2;

    for (int i = 0; i < n1; ++i) {
        int j = nn12[i];
        if (nn21[j] == i) {
            float sim = simMatrix[static_cast<size_t>(i) * n2 + j];
            if (sim > matchThresh) {
                matchIdx1.push_back(i);
                matchIdx2.push_back(j);
            }
        }
    }

    return {matchIdx1, matchIdx2};
}

struct CLIDDTRT::Impl {
    TrtEngine engine_;
    int topK_;
    float scoreThresh_;
    int radius_;
    int border_;
    float matchThresh_;
    float beta_;
    Dims inputDims_;
    Dims keypointsDims_;
    Dims scoresDims_;
    Dims descriptorsDims_;
    int descriptorDim_;

    explicit Impl(const std::string& enginePath, int topK, float scoreThresh, 
               int radius, int border, float matchThresh, float beta)
        : engine_(std::move(enginePath)),
          topK_(topK),
          scoreThresh_(scoreThresh),
          radius_(radius),
          border_(border),
          matchThresh_(matchThresh),
          beta_(beta) {
        
        if (topK_ <= 0) {
            throw std::runtime_error("topK must be positive");
        }

        if (!engine_.hasOutput("keypoints") || !engine_.hasOutput("scores") || !engine_.hasOutput("descriptors")) {
            throw std::runtime_error("Engine must have outputs: keypoints, scores, descriptors");
        }

        inputDims_ = engine_.inputDims();
        keypointsDims_ = engine_.outputDims("keypoints");
        scoresDims_ = engine_.outputDims("scores");
        descriptorsDims_ = engine_.outputDims("descriptors");

        if (keypointsDims_.nbDims != 3 || scoresDims_.nbDims != 2 || descriptorsDims_.nbDims != 3) {
            throw std::runtime_error("Unexpected output dimensions");
        }

        descriptorDim_ = descriptorsDims_.d[descriptorsDims_.nbDims - 1];
    }

    ImageFeatures runSingleImage(const cv::Mat& image) {
        auto prep = preprocessImage(image, engine_.targetH(), engine_.targetW());

        std::vector<float> input(3LL * prep.inputH * prep.inputW);
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < prep.inputH; ++h) {
                for (int w = 0; w < prep.inputW; ++w) {
                    input[static_cast<size_t>(c) * prep.inputH * prep.inputW + 
                           static_cast<size_t>(h) * prep.inputW + static_cast<size_t>(w)] = 
                        prep.input[static_cast<size_t>(h) * prep.inputW * 3 + 
                                  static_cast<size_t>(w) * 3 + static_cast<size_t>(c)];
                }
            }
        }

        engine_.setInput("image", input.data());

        engine_.infer();

        const int numKeypoints = keypointsDims_.d[1];

        std::vector<float> keypointsHost(engine_.outputElementCount("keypoints"));
        std::vector<float> scoresHost(engine_.outputElementCount("scores"));
        std::vector<float> descriptorsHost(engine_.outputElementCount("descriptors"));

        engine_.copyOutputToHost("keypoints", keypointsHost.data(), keypointsHost.size() * sizeof(float));
        engine_.copyOutputToHost("scores", scoresHost.data(), scoresHost.size() * sizeof(float));
        engine_.copyOutputToHost("descriptors", descriptorsHost.data(), descriptorsHost.size() * sizeof(float));

        std::vector<std::array<float, 2>> validKeypoints;
        std::vector<float> validScores;
        std::vector<float> validDescriptors;
        validKeypoints.reserve(numKeypoints);
        validScores.reserve(numKeypoints);
        validDescriptors.reserve(static_cast<size_t>(numKeypoints) * descriptorDim_);

        const float scaleX = prep.inputW > 0 ? static_cast<float>(prep.originalW) / static_cast<float>(prep.inputW) : 1.0f;
        const float scaleY = prep.inputH > 0 ? static_cast<float>(prep.originalH) / static_cast<float>(prep.inputH) : 1.0f;

        for (int i = 0; i < numKeypoints; ++i) {
            float score = scoresHost[i];
            if (std::isfinite(score) && score > scoreThresh_) {
                bool descFinite = true;
                for (int d = 0; d < descriptorDim_; ++d) {
                    if (!std::isfinite(descriptorsHost[i * descriptorDim_ + d])) {
                        descFinite = false;
                        break;
                    }
                }
                if (!descFinite) {
                    continue;
                }
                std::array<float, 2> kpt = {
                    keypointsHost[i * 2] * scaleX,
                    keypointsHost[i * 2 + 1] * scaleY
                };
                validKeypoints.emplace_back(kpt);
                validScores.push_back(score);
                for (int d = 0; d < descriptorDim_; ++d) {
                    validDescriptors.emplace_back(descriptorsHost[i * descriptorDim_ + d]);
                }
            }
        }

        if (validKeypoints.empty() && !scoresHost.empty()) {
            float maxScore = scoresHost[0];
            float minScore = scoresHost[0];
            int finiteCount = 0;
            for (float s : scoresHost) {
                if (std::isfinite(s)) {
                    ++finiteCount;
                    maxScore = std::max(maxScore, s);
                    minScore = std::min(minScore, s);
                }
            }
            std::cerr << "DEBUG no_valid_kpts: scoreThresh=" << scoreThresh_
                      << " finite=" << finiteCount << "/" << scoresHost.size()
                      << " min=" << minScore << " max=" << maxScore << " first10=";
            for (int i = 0; i < std::min<int>(10, static_cast<int>(scoresHost.size())); ++i) {
                std::cerr << scoresHost[static_cast<size_t>(i)] << " ";
            }
            std::cerr << std::endl;
        }

        ImageFeatures features;
        features.keypoints = validKeypoints;
        features.scores = validScores;
        features.descriptors = validDescriptors;
        features.descriptorDim = descriptorDim_;
        features.originalH = prep.originalH;
        features.originalW = prep.originalW;
        features.scaleX = scaleX;
        features.scaleY = scaleY;

        return features;
    }

    int topK() const { return topK_; }
    float scoreThresh() const { return scoreThresh_; }
    int radius() const { return radius_; }
    int border() const { return border_; }
    float matchThresh() const { return matchThresh_; }
    float beta() const { return beta_; }
    int modelInputH() const { return inputDims_.d[2]; }
    int modelInputW() const { return inputDims_.d[3]; }
};

CLIDDTRT::CLIDDTRT(const std::string& enginePath, int topK, float scoreThresh,
                     int radius, int border, float matchThresh, float beta)
    : impl_(new Impl(enginePath, topK, scoreThresh, radius, border, matchThresh, beta)) {}

CLIDDTRT::~CLIDDTRT() = default;

CLIDDTRT::MatchResult CLIDDTRT::match(const cv::Mat& img0_bgr, const cv::Mat& img1_bgr) {
    if (img0_bgr.empty() || img1_bgr.empty()) throw std::runtime_error("Empty input image");
    MatchResult out;

    const auto t0 = Clock::now();
    auto feat1 = impl_->runSingleImage(img0_bgr);
    auto feat2 = impl_->runSingleImage(img1_bgr);
    const auto t1 = Clock::now();
    out.ms_infer = ms_since(t0, t1) / 2.0;

    const auto t2 = Clock::now();
    auto [matchIdx1, matchIdx2] = matchFeaturesCPU(feat1, feat2, impl_->matchThresh(), impl_->beta());
    const auto t3 = Clock::now();
    out.ms_match = ms_since(t2, t3);

    for (size_t i = 0; i < matchIdx1.size(); ++i) {
        std::array<float, 2> kpt0 = {
            feat1.keypoints[matchIdx1[i]][0],
            feat1.keypoints[matchIdx1[i]][1]
        };
        std::array<float, 2> kpt1 = {
            feat2.keypoints[matchIdx2[i]][0],
            feat2.keypoints[matchIdx2[i]][1]
        };
        out.keypoints1.emplace_back(kpt0);
        out.keypoints2.emplace_back(kpt1);
    }

    const auto t4 = Clock::now();
    if (out.keypoints1.size() >= 4) {
        std::vector<unsigned char> mask;
        out.H = cv::findHomography(out.keypoints1, out.keypoints2, 
                                   cv::USAC_MAGSAC, 3.0, mask, 2000, 0.95);
        out.inlier_mask.assign(mask.begin(), mask.end());
        for (size_t i = 0; i < mask.size(); ++i) {
            if (!mask[i]) continue;
            out.inlier_kpts0.emplace_back(out.keypoints1[i]);
            out.inlier_kpts1.emplace_back(out.keypoints2[i]);
        }
    }
    const auto t5 = Clock::now();
    out.ms_ransac = ms_since(t4, t5);

    return out;
}

int CLIDDTRT::topK() const { return impl_->topK(); }
float CLIDDTRT::scoreThresh() const { return impl_->scoreThresh(); }
int CLIDDTRT::radius() const { return impl_->radius(); }
int CLIDDTRT::border() const { return impl_->border(); }
float CLIDDTRT::matchThresh() const { return impl_->matchThresh(); }
float CLIDDTRT::beta() const { return impl_->beta(); }
int CLIDDTRT::modelInputH() const { return impl_->modelInputH(); }
int CLIDDTRT::modelInputW() const { return impl_->modelInputW(); }

}  // namespace clidd
