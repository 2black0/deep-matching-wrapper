#include "LiftFeatOnnxMatcher.h"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <filesystem>

namespace fs = std::filesystem;

namespace dmw {
namespace liftfeat_onnx {

// Helper functions matching Python implementation

static void softmax(float* data, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float* row = data + i * cols;
        float max_val = *std::max_element(row, row + cols);
        
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            row[j] = std::exp(row[j] - max_val);
            sum += row[j];
        }
        
        float inv_sum = 1.0f / (sum + 1e-12f);
        for (int j = 0; j < cols; ++j) {
            row[j] *= inv_sum;
        }
    }
}

static cv::Mat logits_to_heatmap(const std::vector<float>& kpt_logits, 
                                  int B, int C, int h_feat, int w_feat) {
    // Input layout: (B, C, h_feat, w_feat)
    // Need to apply softmax across C dimension for each spatial location
    
    std::vector<float> scores_raw(kpt_logits);
    
    // Apply softmax across channel dimension (C) for each (B, h, w) location
    // Reshape from (B, C, h_feat, w_feat) to (B * h_feat * w_feat, C)
    // Apply softmax, then reshape back
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < h_feat; ++h) {
            for (int w = 0; w < w_feat; ++w) {
                // Gather values across all channels for this spatial location
                std::vector<float> channel_vals(C);
                for (int c = 0; c < C; ++c) {
                    int idx = b * (C * h_feat * w_feat) + c * (h_feat * w_feat) + h * w_feat + w;
                    channel_vals[c] = scores_raw[idx];
                }
                
                // Apply softmax
                float max_val = *std::max_element(channel_vals.begin(), channel_vals.end());
                float sum = 0.0f;
                for (int c = 0; c < C; ++c) {
                    channel_vals[c] = std::exp(channel_vals[c] - max_val);
                    sum += channel_vals[c];
                }
                float inv_sum = 1.0f / (sum + 1e-12f);
                for (int c = 0; c < C; ++c) {
                    channel_vals[c] *= inv_sum;
                }
                
                // Write back
                for (int c = 0; c < C; ++c) {
                    int idx = b * (C * h_feat * w_feat) + c * (h_feat * w_feat) + h * w_feat + w;
                    scores_raw[idx] = channel_vals[c];
                }
            }
        }
    }
    
    // Take only first 64 channels and reshape to (B, h_feat * 8, w_feat * 8)
    int out_h = h_feat * 8;
    int out_w = w_feat * 8;
    cv::Mat heat = cv::Mat::zeros(out_h, out_w, CV_32F);
    
    // Reshape from (B, C, h_feat, w_feat) selecting first 64 channels
    // to (B, h_feat, w_feat, 8, 8) then to (B, h_feat*8, w_feat*8)
    for (int h = 0; h < h_feat; ++h) {
        for (int w = 0; w < w_feat; ++w) {
            for (int ch = 0; ch < 64; ++ch) {
                int sub_h = ch / 8;
                int sub_w = ch % 8;
                // Index into (B, C, h_feat, w_feat) layout
                int idx = ch * (h_feat * w_feat) + h * w_feat + w;
                heat.at<float>(h * 8 + sub_h, w * 8 + sub_w) = scores_raw[idx];
            }
        }
    }
    
    return heat;
}

static cv::Mat simple_nms(const cv::Mat& heatmap, float threshold, int kernel_size) {
    cv::Mat local_max;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, 
                                                cv::Size(kernel_size, kernel_size));
    cv::dilate(heatmap, local_max, kernel);
    
    cv::Mat peaks;
    cv::compare(heatmap, local_max - 1e-6f, peaks, cv::CMP_GE);
    
    cv::Mat thresh_mask;
    cv::compare(heatmap, threshold, thresh_mask, cv::CMP_GT);
    
    cv::bitwise_and(peaks, thresh_mask, peaks);
    
    return peaks;
}

static std::vector<float> remap_sample_channel(const cv::Mat& feat_channel,
                                                const std::vector<int>& x,
                                                const std::vector<int>& y,
                                                int H, int W,
                                                int interpolation) {
    int h_f = feat_channel.rows;
    int w_f = feat_channel.cols;
    std::vector<float> out(x.size());
    
    for (size_t i = 0; i < x.size(); ++i) {
        float map_x = x[i] * (float(w_f) / float(W - 1)) - 0.5f;
        float map_y = y[i] * (float(h_f) / float(H - 1)) - 0.5f;
        
        // Bilinear interpolation
        int x0 = std::floor(map_x);
        int y0 = std::floor(map_y);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        
        float wx = map_x - x0;
        float wy = map_y - y0;
        
        auto get_val = [&](int yy, int xx) -> float {
            if (yy < 0 || yy >= h_f || xx < 0 || xx >= w_f) return 0.0f;
            return feat_channel.at<float>(yy, xx);
        };
        
        float val00 = get_val(y0, x0);
        float val01 = get_val(y0, x1);
        float val10 = get_val(y1, x0);
        float val11 = get_val(y1, x1);
        
        float val0 = val00 * (1.0f - wx) + val01 * wx;
        float val1 = val10 * (1.0f - wx) + val11 * wx;
        out[i] = val0 * (1.0f - wy) + val1 * wy;
    }
    
    return out;
}

static void extract_keypoints(const cv::Mat& heat,
                              const std::vector<float>& desc_map,
                              int desc_channels,
                              int top_k,
                              float threshold,
                              std::vector<cv::Point2f>& kpts,
                              std::vector<std::vector<float>>& descs) {
    int H = heat.rows;
    int W = heat.cols;
    
    cv::Mat peaks = simple_nms(heat, threshold, 5);
    
    std::vector<int> x_coords, y_coords;
    std::vector<float> scores_list;
    
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (peaks.at<uint8_t>(y, x) > 0) {
                x_coords.push_back(x);
                y_coords.push_back(y);
                scores_list.push_back(heat.at<float>(y, x));
            }
        }
    }
    
    if (x_coords.empty()) {
        return;
    }
    
    // Select top-k by score
    if (top_k > 0 && scores_list.size() > size_t(top_k)) {
        std::vector<size_t> indices(scores_list.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                         [&scores_list](size_t i, size_t j) {
                             return scores_list[i] > scores_list[j];
                         });
        
        std::vector<int> x_sel, y_sel;
        for (int i = 0; i < top_k; ++i) {
            x_sel.push_back(x_coords[indices[i]]);
            y_sel.push_back(y_coords[indices[i]]);
        }
        x_coords = x_sel;
        y_coords = y_sel;
    }
    
    // Extract descriptors
    int h_desc = H / 8;
    int w_desc = W / 8;
    
    for (int c = 0; c < desc_channels; ++c) {
        // Extract channel from desc_map
        cv::Mat channel(h_desc, w_desc, CV_32F);
        for (int y = 0; y < h_desc; ++y) {
            for (int x = 0; x < w_desc; ++x) {
                int idx = (c * h_desc + y) * w_desc + x;
                channel.at<float>(y, x) = desc_map[idx];
            }
        }
        
        // Normalize channel
        cv::Mat channel_norm;
        cv::normalize(channel, channel_norm, 1.0, 0.0, cv::NORM_L2);
        
        auto sampled = remap_sample_channel(channel_norm, x_coords, y_coords, H, W, cv::INTER_CUBIC);
        
        if (c == 0) {
            descs.resize(x_coords.size(), std::vector<float>(desc_channels));
        }
        for (size_t i = 0; i < x_coords.size(); ++i) {
            descs[i][c] = sampled[i];
        }
    }
    
    // Normalize descriptors
    for (auto& desc : descs) {
        float norm = 0.0f;
        for (float v : desc) {
            norm += v * v;
        }
        norm = std::sqrt(norm) + 1e-8f;
        for (float& v : desc) {
            v /= norm;
        }
    }
    
    // Store keypoints
    kpts.reserve(x_coords.size());
    for (size_t i = 0; i < x_coords.size(); ++i) {
        kpts.emplace_back(x_coords[i], y_coords[i]);
    }
}

static void match_mnn(const std::vector<std::vector<float>>& desc0,
                     const std::vector<std::vector<float>>& desc1,
                     float min_cossim,
                     std::vector<int>& idx0,
                     std::vector<int>& idx1) {
    if (desc0.empty() || desc1.empty()) {
        return;
    }
    
    int n0 = desc0.size();
    int n1 = desc1.size();
    
    // Compute similarity matrix
    std::vector<std::vector<float>> sims(n0, std::vector<float>(n1));
    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            float sim = 0.0f;
            for (size_t k = 0; k < desc0[i].size(); ++k) {
                sim += desc0[i][k] * desc1[j][k];
            }
            sims[i][j] = sim;
        }
    }
    
    // Mutual nearest neighbors
    std::vector<int> m01(n0), m10(n1);
    for (int i = 0; i < n0; ++i) {
        m01[i] = std::max_element(sims[i].begin(), sims[i].end()) - sims[i].begin();
    }
    for (int j = 0; j < n1; ++j) {
        float max_sim = sims[0][j];
        int max_idx = 0;
        for (int i = 1; i < n0; ++i) {
            if (sims[i][j] > max_sim) {
                max_sim = sims[i][j];
                max_idx = i;
            }
        }
        m10[j] = max_idx;
    }
    
    // Filter mutual matches
    for (int i = 0; i < n0; ++i) {
        int j = m01[i];
        if (m10[j] == i) {
            if (min_cossim > 0 && sims[i][j] <= min_cossim) {
                continue;
            }
            idx0.push_back(i);
            idx1.push_back(j);
        }
    }
}

// Pimpl implementation
struct LiftFeatOnnxMatcher::Impl {
    LiftFeatConfig config;
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memory_info{nullptr};
    
    Impl(const LiftFeatConfig& cfg) : config(cfg) {
        // Resolve weights path
        if (config.weights_path.empty()) {
            fs::path weights_dir = fs::path(__FILE__).parent_path().parent_path() / "weights";
            std::string filename = "liftfeat_" + config.dtype + "_" + 
                                  std::to_string(config.width) + "x" + 
                                  std::to_string(config.height) + ".onnx";
            config.weights_path = (weights_dir / filename).string();
        }
        
        if (!fs::exists(config.weights_path)) {
            throw std::runtime_error("Missing LiftFeat ONNX weights: " + config.weights_path);
        }
        
        // Initialize ONNX Runtime
        env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "LiftFeatOnnxMatcher");
        
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        if (config.device == "cuda") {
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        }
        
        session = std::make_unique<Ort::Session>(*env, config.weights_path.c_str(), session_options);
        memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    }
    
    std::tuple<std::vector<cv::Point2f>, std::vector<std::vector<float>>, double, double>
    extract_features(const cv::Mat& img) {
        auto t0 = std::chrono::high_resolution_clock::now();
        
        // Preprocess
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(config.width, config.height));
        
        std::vector<float> input_tensor(1 * 3 * config.height * config.width);
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < config.height; ++y) {
                for (int x = 0; x < config.width; ++x) {
                    int idx = (c * config.height + y) * config.width + x;
                    input_tensor[idx] = resized.at<cv::Vec3b>(y, x)[c] / 255.0f;
                }
            }
        }
        
        if (config.dtype == "fp16") {
            // Convert to fp16 if needed (simplified - would need proper fp16 support)
            // For now, keeping as fp32
        }
        
        auto t1 = std::chrono::high_resolution_clock::now();
        
        // Run inference
        std::vector<int64_t> input_shape = {1, 3, config.height, config.width};
        auto input_tensor_ort = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor.data(), input_tensor.size(),
            input_shape.data(), input_shape.size());
        
        const char* input_names[] = {"image"};
        const char* output_names[] = {"kpt_logits", "descriptors_map"};
        
        auto output_tensors = session->Run(Ort::RunOptions{nullptr},
                                          input_names, &input_tensor_ort, 1,
                                          output_names, 2);
        
        auto t2 = std::chrono::high_resolution_clock::now();
        
        // Get outputs
        float* kpt_logits_data = output_tensors[0].GetTensorMutableData<float>();
        float* desc_map_data = output_tensors[1].GetTensorMutableData<float>();
        
        auto kpt_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        auto desc_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
        
        int B = kpt_shape[0];
        int C = kpt_shape[1];
        int h_feat = kpt_shape[2];
        int w_feat = kpt_shape[3];
        int desc_C = desc_shape[1];
        
        // Convert to heatmap
        std::vector<float> kpt_logits_vec(kpt_logits_data, 
                                          kpt_logits_data + B * C * h_feat * w_feat);
        cv::Mat heat = logits_to_heatmap(kpt_logits_vec, B, C, h_feat, w_feat);
        
        // Extract keypoints
        std::vector<float> desc_map_vec(desc_map_data,
                                       desc_map_data + B * desc_C * h_feat * w_feat);
        std::vector<cv::Point2f> kpts_resized;
        std::vector<std::vector<float>> descs;
        extract_keypoints(heat, desc_map_vec, desc_C, config.top_k, 
                         config.detect_threshold, kpts_resized, descs);
        
        // Rescale keypoints to original image size
        float scale_x = float(img.cols) / float(config.width);
        float scale_y = float(img.rows) / float(config.height);
        std::vector<cv::Point2f> kpts;
        for (const auto& pt : kpts_resized) {
            kpts.emplace_back(pt.x * scale_x, pt.y * scale_y);
        }
        
        auto t3 = std::chrono::high_resolution_clock::now();
        
        double ms_preprocess = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double ms_infer = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double ms_postprocess = std::chrono::duration<double, std::milli>(t3 - t2).count();
        
        return {kpts, descs, ms_preprocess + ms_infer, ms_postprocess};
    }
};

// Public API
LiftFeatOnnxMatcher::LiftFeatOnnxMatcher(const LiftFeatConfig& config)
    : impl_(std::make_unique<Impl>(config)) {
}

LiftFeatOnnxMatcher::~LiftFeatOnnxMatcher() = default;

MatchResult LiftFeatOnnxMatcher::match(const cv::Mat& img0, const cv::Mat& img1) {
    auto t0 = std::chrono::high_resolution_clock::now();
    
    // Convert to RGB
    cv::Mat img0_rgb, img1_rgb;
    cv::cvtColor(img0, img0_rgb, cv::COLOR_BGR2RGB);
    cv::cvtColor(img1, img1_rgb, cv::COLOR_BGR2RGB);
    
    // Extract features
    auto [kpts0, desc0, ms_infer0, ms_post0] = impl_->extract_features(img0_rgb);
    auto [kpts1, desc1, ms_infer1, ms_post1] = impl_->extract_features(img1_rgb);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    
    // Match
    std::vector<int> idx0, idx1;
    match_mnn(desc0, desc1, impl_->config.min_cossim, idx0, idx1);
    
    auto t2 = std::chrono::high_resolution_clock::now();
    
    // Build result
    MatchResult result;
    result.kpts0 = kpts0;
    result.kpts1 = kpts1;
    result.desc0 = desc0;
    result.desc1 = desc1;
    
    for (size_t i = 0; i < idx0.size(); ++i) {
        result.mkpts0.push_back(kpts0[idx0[i]]);
        result.mkpts1.push_back(kpts1[idx1[i]]);
    }
    
    result.ms_infer = (ms_infer0 + ms_infer1) / 2.0;
    result.ms_postprocess = ms_post0 + ms_post1;
    result.ms_match = std::chrono::duration<double, std::milli>(t2 - t1).count();
    result.ms_total = std::chrono::duration<double, std::milli>(t2 - t0).count();
    
    return result;
}

} // namespace liftfeat_onnx
} // namespace dmw
