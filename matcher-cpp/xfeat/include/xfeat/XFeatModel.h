#pragma once

#include <torch/torch.h>

namespace dmw::xfeat {

/**
 * BasicLayer: Conv2d -> BatchNorm2d -> ReLU
 * Mirrors Python: matcher/xfeat/modules/model.py:12-25
 */
class BasicLayerImpl : public torch::nn::Module {
 public:
  BasicLayerImpl(int64_t in_channels, int64_t out_channels, 
                 int64_t kernel_size = 3, int64_t stride = 1, 
                 int64_t padding = 1, int64_t dilation = 1, bool bias = false);
  
  torch::Tensor forward(torch::Tensor x);
  
 private:
  torch::nn::Conv2d conv{nullptr};
  torch::nn::BatchNorm2d bn{nullptr};
  torch::nn::ReLU relu{nullptr};
};
TORCH_MODULE(BasicLayer);

/**
 * XFeatModel: CNN backbone + heads for feature extraction
 * Mirrors Python: matcher/xfeat/modules/model.py:27-155
 * 
 * Forward pass:
 *   Input: (B, C, H, W) RGB or grayscale image
 *   Output: 
 *     - feats:     (B, 64, H/8, W/8) dense descriptors
 *     - keypoints: (B, 65, H/8, W/8) keypoint logits
 *     - heatmap:   (B,  1, H/8, W/8) reliability map
 */
class XFeatModelImpl : public torch::nn::Module {
 public:
  XFeatModelImpl();
  
  /**
   * Forward pass through the network.
   * Returns: {feats, keypoints, heatmap}
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x);
  
  /**
   * Fine matcher MLP for xfeat-star refinement.
   * Input: (N, 128) concatenated descriptor pairs
   * Output: (N, 64) offset distribution logits
   */
  torch::Tensor fine_matcher_forward(torch::Tensor x);
  
 private:
  // Normalization
  torch::nn::InstanceNorm2d norm{nullptr};
  
  // Skip connection
  torch::nn::Sequential skip1{nullptr};
  
  // CNN blocks
  torch::nn::Sequential block1{nullptr};
  torch::nn::Sequential block2{nullptr};
  torch::nn::Sequential block3{nullptr};
  torch::nn::Sequential block4{nullptr};
  torch::nn::Sequential block5{nullptr};
  torch::nn::Sequential block_fusion{nullptr};
  
  // Output heads
  torch::nn::Sequential heatmap_head{nullptr};
  torch::nn::Sequential keypoint_head{nullptr};
  
  // Fine matcher MLP (for xfeat-star)
  torch::nn::Sequential fine_matcher{nullptr};
  
  /**
   * Unfold tensor in 2D with window size ws.
   * Used for keypoint head to extract 8x8 patches.
   */
  torch::Tensor unfold2d(torch::Tensor x, int64_t ws = 2);
};
TORCH_MODULE(XFeatModel);

}  // namespace dmw::xfeat
