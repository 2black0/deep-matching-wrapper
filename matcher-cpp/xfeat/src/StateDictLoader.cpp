#include "xfeat/StateDictLoader.h"

#include <filesystem>
#include <stdexcept>

namespace dmw::xfeat {

StateDictLoader::StateDictLoader(const std::string& pt_path) : pt_path_(pt_path) {
  if (!std::filesystem::exists(pt_path_)) {
    throw std::runtime_error("State dict file not found: " + pt_path_);
  }
}

std::unordered_map<std::string, torch::Tensor> StateDictLoader::load() {
  if (loaded_) {
    return state_dict_;
  }
  
  try {
    // Load the .pt file as a dictionary (IValue)
    torch::jit::script::Module container = torch::jit::load(pt_path_);
    
    // Try to extract as a state dict (OrderedDict<string, Tensor>)
    // PyTorch state dicts saved with torch.save() are pickled dicts
    auto dict = container.attr("state_dict");
    
    if (dict.isGenericDict()) {
      auto generic_dict = dict.toGenericDict();
      for (const auto& item : generic_dict) {
        std::string key = item.key().toStringRef();
        torch::Tensor value = item.value().toTensor();
        state_dict_[key] = value;
      }
    }
  } catch (const c10::Error& e) {
    // Fallback: try loading directly as a dict (older PyTorch format)
    try {
      torch::IValue ivalue = torch::pickle_load(std::ifstream(pt_path_, std::ios::binary));
      
      if (ivalue.isGenericDict()) {
        auto dict = ivalue.toGenericDict();
        for (const auto& item : dict) {
          std::string key = item.key().toStringRef();
          torch::Tensor value = item.value().toTensor();
          state_dict_[key] = value;
        }
      } else {
        throw std::runtime_error("Failed to parse state dict from: " + pt_path_);
      }
    } catch (const std::exception& e2) {
      throw std::runtime_error("Failed to load state dict: " + std::string(e2.what()));
    }
  }
  
  loaded_ = true;
  return state_dict_;
}

void StateDictLoader::load_into_module(torch::nn::Module& module) {
  ensure_loaded();
  
  auto named_params = module.named_parameters();
  auto named_buffers = module.named_buffers();
  
  int loaded_count = 0;
  int missing_count = 0;
  
  // Load parameters
  for (const auto& param : named_params) {
    const std::string& name = param.key();
    if (state_dict_.count(name)) {
      param.value().data().copy_(state_dict_[name]);
      loaded_count++;
    } else {
      missing_count++;
      // Optional: warn about missing keys
      // std::cerr << "Warning: Missing parameter in state dict: " << name << std::endl;
    }
  }
  
  // Load buffers (e.g., BatchNorm running stats)
  for (const auto& buffer : named_buffers) {
    const std::string& name = buffer.key();
    if (state_dict_.count(name)) {
      buffer.value().data().copy_(state_dict_[name]);
      loaded_count++;
    }
  }
  
  if (loaded_count == 0) {
    throw std::runtime_error("No parameters loaded from state dict!");
  }
}

torch::Tensor StateDictLoader::get_tensor(const std::string& key) {
  ensure_loaded();
  if (!has_key(key)) {
    throw std::runtime_error("Key not found in state dict: " + key);
  }
  return state_dict_[key];
}

bool StateDictLoader::has_key(const std::string& key) const {
  return state_dict_.count(key) > 0;
}

void StateDictLoader::ensure_loaded() {
  if (!loaded_) {
    load();
  }
}

}  // namespace dmw::xfeat
