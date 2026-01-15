#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "scheduler/managers/d2d_transmission_optimizer.h"

using xllm_service::D2DTransmissionOptimizer;

int main() {
  // Scenario:
  // - Existing 5 deepseek-v3 instances in xllm_service
  // - Each instance has 16 NPUs
  // - ep-size = 16 (one expert per device per layer for simplicity)
  // - A new deepseek-v3 instance with the same config joins
  // Goal: Compute D2D plan for one layer: which instance/device to fetch each expert.

  const int existing_instances = 5;
  const int total_experts = 256; 
  const int npus_per_instance = 16;
  const int experts_per_npu = total_experts / npus_per_instance;

  std::vector<std::string> inst_names;
  inst_names.reserve(existing_instances);
  for (int i = 0; i < existing_instances; ++i) {
    inst_names.push_back("deepseekv3-" + std::to_string(i + 1));
  }
  std::string target_inst = "deepseekv3-new";

  // Build expert_to_src: each expert id [0..15] is available on every existing instance
  // at local_npu = expert_id / experts_per_device (two experts per NPU).
  std::unordered_map<int, std::vector<D2DTransmissionOptimizer::GlobalNpu>>
      expert_to_src;
  for (int expert_id = 0; expert_id < total_experts; ++expert_id) {
    std::vector<D2DTransmissionOptimizer::GlobalNpu> sources;
    for (const auto& name : inst_names) {
        // 计算该专家在源实例的哪张卡上
        // 假设专家是按顺序切分的：NPU 0 [0-15], NPU 1 [16-31]...
        int local_npu = expert_id / experts_per_npu; 
        sources.push_back({name, local_npu});
    }
    expert_to_src[expert_id] = std::move(sources);
}

  // Required experts for the target instance's single layer
  std::vector<int> required;
  for (int e = 0; e < total_experts; ++e) {
    required.push_back(e);
  }

  D2DTransmissionOptimizer opt;
  auto steps = opt.optimize_layer(required, expert_to_src);

  // Print D2D plan
  std::cout << "Target instance: " << target_inst << std::endl;
  std::cout << "Layer 0 D2D steps (expert_id -> src_instance:src_npu)" << std::endl;
  for (const auto& s : steps) {
    std::cout << "  expert " << s.expert_id << " <- "
              << s.src.instance << ":" << s.src.local_npu << std::endl;
  }

  return 0;
}
