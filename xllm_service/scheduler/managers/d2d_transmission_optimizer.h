#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <tuple>

namespace xllm_service {

class D2DTransmissionOptimizer {
 public:
  struct InstanceConfig {
    int device_size = 0; // 实例总 NPU 数
    int dp_size = 0;     // DP 组个数
  };

  struct GlobalNpu {
    std::string instance;
    int local_npu;

    // 为了能作为 map 的 key
    bool operator==(const GlobalNpu& o) const {
      return instance == o.instance && local_npu == o.local_npu;
    }
  };

  struct GlobalNpuHasher {
    std::size_t operator()(const GlobalNpu& gn) const {
      return std::hash<std::string>{}(gn.instance) ^ (std::hash<int>{}(gn.local_npu) << 1);
    }
  };
  
  struct Step {
    GlobalNpu src;
    int expert_id;
  };

  struct NonExpertStep {
    std::string src_instance;
    int dp_group_index = -1;   // 选中的 DP 组索引
    int start_npu_index = -1;  // 该组起始的 NPU 编号
    int dp_size = 0;           // DP 组个数
  };

  std::vector<Step> optimize_layer(
      const std::vector<int>& required_per_target,
      const std::unordered_map<int, std::vector<GlobalNpu>>& expert_to_src);

  NonExpertStep optimize_non_expert(
    const std::vector<Step>& expert_steps,
    const std::unordered_map<std::string, InstanceConfig>& instance_configs);

 private:
  struct Edge {
    int to;
    int rev;
    int cap;
  };

  struct MaxFlow {
    std::vector<std::vector<Edge>> g;
    std::vector<int> level;
    std::vector<int> it;
    int s;
    int t;

    void init(int n, int ss, int tt);
    void add_edge(int u, int v, int c);
    bool bfs();
    int dfs(int v, int f);
    int dinic();
  };

  bool feasible(
    int K,
    const std::vector<int>& required_per_target,
    const std::unordered_map<int, std::vector<GlobalNpu>>& expert_to_src,
    std::vector<GlobalNpu>& npu_index_map,
    std::vector<int>& req_to_expert_id,
    MaxFlow& mf);

  std::vector<Step> extract_plan(
    const MaxFlow& mf,
    const std::vector<GlobalNpu>& npu_index_map,
    const std::vector<int>& req_to_expert_id,
    int npu_offset,
    int req_offset);
};

}
