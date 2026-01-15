#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <random>
#include <iomanip>
#include "scheduler/managers/d2d_transmission_optimizer.h"

using xllm_service::D2DTransmissionOptimizer;

// 辅助函数：打印并统计专家负载情况
void analyze_expert_plan(const std::vector<D2DTransmissionOptimizer::Step>& steps, 
                         std::unordered_map<std::string, std::vector<int>>& out_npu_loads,
                         int num_instances, int npus_per_instance) {
    int max_load = 0;
    
    // 初始化负载矩阵
    for (int i = 0; i < num_instances; ++i) {
        out_npu_loads["inst-" + std::to_string(i)].assign(npus_per_instance, 0);
    }

    for (const auto& s : steps) {
        out_npu_loads[s.src.instance][s.src.local_npu]++;
        max_load = std::max(max_load, out_npu_loads[s.src.instance][s.src.local_npu]);
    }

    std::cout << "  [Expert] Total Transferred: " << steps.size() << std::endl;
    std::cout << "  [Expert] Global Max NPU Load: " << max_load << std::endl;
}

// 辅助函数：展示非专家权重的选择逻辑
void analyze_non_expert_plan(const D2DTransmissionOptimizer::NonExpertStep& ne_step,
                             const std::unordered_map<std::string, std::vector<int>>& npu_loads,
                             int dp_group_num,
                             int npus_per_instance) {
    std::cout << "  [Non-Expert] Selected Instance: " << ne_step.src_instance << std::endl;
    std::cout << "  [Non-Expert] Selected DP Group Index: " << ne_step.dp_group_index << std::endl;
    int npu_per_group = 0;
    if (dp_group_num > 0 && npus_per_instance % dp_group_num == 0) {
        npu_per_group = npus_per_instance / dp_group_num;
    }
    std::cout << "  [Non-Expert] NPU Range: [" << ne_step.start_npu_index << " - " 
              << ne_step.start_npu_index + std::max(0, npu_per_group - 1) << "]" << std::endl;
    
    // 计算被选中的 DP 组的实际最大专家负载
    int group_max_expert_load = 0;
    if (npu_loads.count(ne_step.src_instance)) {
        const auto& loads = npu_loads.at(ne_step.src_instance);
        for (int i = 0; i < npu_per_group; ++i) {
            group_max_expert_load = std::max(group_max_expert_load, loads[ne_step.start_npu_index + i]);
        }
    }
    std::cout << "  [Non-Expert] Chosen DP Group Max Expert Load: " << group_max_expert_load << std::endl;
}

void run_test_case(int case_id, int num_instances, int redundancy_per_npu, int dp_size) {
    std::cout << "\n--- Test Case " << case_id << ": Insts=" << num_instances 
              << ", Redundancy=" << redundancy_per_npu << ", DP_Size=" << dp_size << " ---" << std::endl;

    const int total_experts = 256;
    const int npus_per_instance = 16;
    const int experts_per_npu_base = total_experts / npus_per_instance;

    std::unordered_map<int, std::vector<D2DTransmissionOptimizer::GlobalNpu>> expert_to_src;
    std::unordered_map<std::string, D2DTransmissionOptimizer::InstanceConfig> inst_configs;
    
    std::mt19937 rng(case_id);

    for (int inst_idx = 0; inst_idx < num_instances; ++inst_idx) {
        std::string inst_name = "inst-" + std::to_string(inst_idx);
        inst_configs[inst_name] = {npus_per_instance, dp_size};
        
        for (int npu_idx = 0; npu_idx < npus_per_instance; ++npu_idx) {
            D2DTransmissionOptimizer::GlobalNpu gn{inst_name, npu_idx};

            // 1. 基础专家
            for (int e = 0; e < experts_per_npu_base; ++e) {
                int expert_id = npu_idx * experts_per_npu_base + e;
                expert_to_src[expert_id].push_back(gn);
            }

            // 2. 随机冗余专家
            std::vector<int> all_experts(total_experts);
            std::iota(all_experts.begin(), all_experts.end(), 0);
            std::shuffle(all_experts.begin(), all_experts.end(), rng);
            for (int i = 0; i < redundancy_per_npu; ++i) {
                expert_to_src[all_experts[i]].push_back(gn);
            }
        }
    }

    std::vector<int> required(total_experts);
    std::iota(required.begin(), required.end(), 0);

    D2DTransmissionOptimizer opt;
    
    // 第一步：优化专家权重路径
    auto expert_steps = opt.optimize_layer(required, expert_to_src);
    
    std::unordered_map<std::string, std::vector<int>> npu_loads;
    analyze_expert_plan(expert_steps, npu_loads, num_instances, npus_per_instance);

    // 第二步：根据专家负载，优化非专家权重路径
    auto ne_step = opt.optimize_non_expert(expert_steps, inst_configs);
    
    analyze_non_expert_plan(ne_step, npu_loads, dp_size, npus_per_instance);
}

int main() {
    // 情况 1: 标准 5 实例。由于前 4 个实例已经承载了所有专家，后方的 DP 组应该是负载 0。
    run_test_case(1, 5, 0, 4);

    // 情况 2: 高冗余。专家负载分布更广，观察是否能避开高负载 DP 组。
    run_test_case(2, 5, 50, 4);

    // 情况 3: DP 组很大 (DP=8)。每台机只有两个组可选。
    run_test_case(3, 3, 20, 8);

    // 情况 4: 只有 1 个实例。必须在专家负载都很高的组里“矮子里拔将军”。
    run_test_case(4, 1, 10, 4);

    return 0;
}
