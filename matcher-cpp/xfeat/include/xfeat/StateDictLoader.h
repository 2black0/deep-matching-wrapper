#pragma once

#include <torch/torch.h>
#include <string>
#include <unordered_map>

namespace dmw::xfeat {

/**
 * Utility to load PyTorch state_dict (.pt files) and populate LibTorch modules.
 * 
 * Usage:
 *   StateDictLoader loader("path/to/xfeat.pt");
 *   loader.load_into_module(my_module);
 */
class StateDictLoader {
 public:
  explicit StateDictLoader(const std::string& pt_path);
  
  /**
   * Load state dict from .pt file.
   * Returns a map of parameter names to tensors.
   */
  std::unordered_map<std::string, torch::Tensor> load();
  
  /**
   * Populate a torch::nn::Module with weights from the state dict.
   * Automatically matches parameter names.
   */
  void load_into_module(torch::nn::Module& module);
  
  /**
   * Get a specific tensor from the state dict.
   */
  torch::Tensor get_tensor(const std::string& key);
  
  /**
   * Check if a key exists in the state dict.
   */
  bool has_key(const std::string& key) const;
  
 private:
  std::string pt_path_;
  std::unordered_map<std::string, torch::Tensor> state_dict_;
  bool loaded_ = false;
  
  void ensure_loaded();
};

}  // namespace dmw::xfeat
