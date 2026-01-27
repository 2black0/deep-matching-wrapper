#include "xfeat/XFeatModel.h"

#include <torch/torch.h>

namespace dmw::xfeat {
namespace F = torch::nn::functional;

// BasicLayer implementation
BasicLayerImpl::BasicLayerImpl(int64_t in_channels, int64_t out_channels,
                               int64_t kernel_size, int64_t stride,
                               int64_t padding, int64_t dilation, bool bias) {
  conv = register_module("0", torch::nn::Conv2d(
      torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
          .stride(stride)
          .padding(padding)
          .dilation(dilation)
          .bias(bias)));
  
  bn = register_module("1", torch::nn::BatchNorm2d(
      torch::nn::BatchNorm2dOptions(out_channels).affine(false)));
  
  relu = register_module("2", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
}

torch::Tensor BasicLayerImpl::forward(torch::Tensor x) {
  x = conv->forward(x);
  x = bn->forward(x);
  x = relu->forward(x);
  return x;
}

// XFeatModel implementation
XFeatModelImpl::XFeatModelImpl() {
  // Normalization
  norm = register_module("norm", torch::nn::InstanceNorm2d(1));
  
  // Skip connection: AvgPool2d + Conv2d
  skip1 = register_module("skip1", torch::nn::Sequential(
      torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(4).stride(4)),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 24, 1).stride(1).padding(0))
  ));
  
  // Block 1
  block1 = register_module("block1", torch::nn::Sequential(
      BasicLayer(1, 4, /*kernel=*/3, /*stride=*/1),
      BasicLayer(4, 8, 3, 2),
      BasicLayer(8, 8, 3, 1),
      BasicLayer(8, 24, 3, 2)
  ));
  
  // Block 2
  block2 = register_module("block2", torch::nn::Sequential(
      BasicLayer(24, 24, 3, 1),
      BasicLayer(24, 24, 3, 1)
  ));
  
  // Block 3
  block3 = register_module("block3", torch::nn::Sequential(
      BasicLayer(24, 64, 3, 2),
      BasicLayer(64, 64, 3, 1),
      BasicLayer(64, 64, /*kernel=*/1, /*stride=*/1, /*padding=*/0)
  ));
  
  // Block 4
  block4 = register_module("block4", torch::nn::Sequential(
      BasicLayer(64, 64, 3, 2),
      BasicLayer(64, 64, 3, 1),
      BasicLayer(64, 64, 3, 1)
  ));
  
  // Block 5
  block5 = register_module("block5", torch::nn::Sequential(
      BasicLayer(64, 128, 3, 2),
      BasicLayer(128, 128, 3, 1),
      BasicLayer(128, 128, 3, 1),
      BasicLayer(128, 64, 1, 1, 0)
  ));
  
  // Fusion block
  block_fusion = register_module("block_fusion", torch::nn::Sequential(
      BasicLayer(64, 64, 3, 1),
      BasicLayer(64, 64, 3, 1),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 1).padding(0))
  ));
  
  // Heatmap head (reliability map)
  heatmap_head = register_module("heatmap_head", torch::nn::Sequential(
      BasicLayer(64, 64, 1, 1, 0),
      BasicLayer(64, 64, 1, 1, 0),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 1, 1)),
      torch::nn::Sigmoid()
  ));
  
  // Keypoint head
  keypoint_head = register_module("keypoint_head", torch::nn::Sequential(
      BasicLayer(64, 64, 1, 1, 0),
      BasicLayer(64, 64, 1, 1, 0),
      BasicLayer(64, 64, 1, 1, 0),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 65, 1))
  ));
  
  // Fine matcher MLP (for xfeat-star)
  fine_matcher = register_module("fine_matcher", torch::nn::Sequential(
      torch::nn::Linear(128, 512),
      torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512).affine(false)),
      torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
      torch::nn::Linear(512, 512),
      torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512).affine(false)),
      torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
      torch::nn::Linear(512, 512),
      torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512).affine(false)),
      torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
      torch::nn::Linear(512, 512),
      torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512).affine(false)),
      torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
      torch::nn::Linear(512, 64)
  ));
}

torch::Tensor XFeatModelImpl::unfold2d(torch::Tensor x, int64_t ws) {
  // Unfolds tensor in 2D with window size ws
  // Mimics: x.unfold(2, ws, ws).unfold(3, ws, ws)
  const int64_t B = x.size(0);
  const int64_t C = x.size(1);
  const int64_t H = x.size(2);
  const int64_t W = x.size(3);
  
  // x: (B, C, H, W)
  // unfold(2, ws, ws): extract patches of size ws in dim 2, stride ws
  // unfold(3, ws, ws): extract patches of size ws in dim 3, stride ws
  x = x.unfold(2, ws, ws).unfold(3, ws, ws);
  // Now: (B, C, H/ws, W/ws, ws, ws)
  
  // Reshape to (B, C, H/ws, W/ws, ws*ws)
  x = x.reshape({B, C, x.size(2), x.size(3), ws * ws});
  
  // Permute to (B, C, ws*ws, H/ws, W/ws)
  x = x.permute({0, 1, 4, 2, 3});
  
  // Reshape to (B, C*ws*ws, H/ws, W/ws)
  return x.reshape({B, C * ws * ws, x.size(3), x.size(4)});
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
XFeatModelImpl::forward(torch::Tensor x) {
  // Don't backprop through normalization
  {
    torch::NoGradGuard no_grad;
    // Convert to grayscale if needed
    if (x.size(1) > 1) {
      x = x.mean(/*dim=*/1, /*keepdim=*/true);
    }
    x = norm->forward(x);
  }
  
  // Main backbone
  auto x1 = block1->forward(x);
  auto x2 = block2->forward(x1 + skip1->forward(x));
  auto x3 = block3->forward(x2);
  auto x4 = block4->forward(x3);
  auto x5 = block5->forward(x4);
  
  // Pyramid fusion
  x4 = F::interpolate(x4, F::InterpolateFuncOptions()
      .size(std::vector<int64_t>{x3.size(-2), x3.size(-1)})
      .mode(torch::kBilinear)
      .align_corners(false));
  
  x5 = F::interpolate(x5, F::InterpolateFuncOptions()
      .size(std::vector<int64_t>{x3.size(-2), x3.size(-1)})
      .mode(torch::kBilinear)
      .align_corners(false));
  
  auto feats = block_fusion->forward(x3 + x4 + x5);
  
  // Heads
  auto heatmap = heatmap_head->forward(feats);
  auto keypoints = keypoint_head->forward(unfold2d(x, /*ws=*/8));
  
  return std::make_tuple(feats, keypoints, heatmap);
}

torch::Tensor XFeatModelImpl::fine_matcher_forward(torch::Tensor x) {
  return fine_matcher->forward(x);
}

}  // namespace dmw::xfeat
