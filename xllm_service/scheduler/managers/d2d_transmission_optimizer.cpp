#include "d2d_transmission_optimizer.h"

#include <algorithm>
#include <unordered_set>

namespace xllm_service {

void D2DTransmissionOptimizer::MaxFlow::init(int n, int ss, int tt) {
  g.assign(n, {});
  level.assign(n, 0);
  it.assign(n, 0);
  s = ss;
  t = tt;
}

void D2DTransmissionOptimizer::MaxFlow::add_edge(int u, int v, int c) {
  Edge a{v, (int)g[v].size(), c};
  Edge b{u, (int)g[u].size(), 0};
  g[u].push_back(a);
  g[v].push_back(b);
}

bool D2DTransmissionOptimizer::MaxFlow::bfs() {
  std::fill(level.begin(), level.end(), -1);
  std::vector<int> q;
  q.reserve(g.size());
  level[s] = 0;
  q.push_back(s);
  for (size_t i = 0; i < q.size(); ++i) {
    int v = q[i];
    for (const auto& e : g[v]) {
      if (e.cap > 0 && level[e.to] < 0) {
        level[e.to] = level[v] + 1;
        q.push_back(e.to);
      }
    }
  }
  return level[t] >= 0;
}

int D2DTransmissionOptimizer::MaxFlow::dfs(int v, int f) {
  if (v == t) return f;
  for (int& i = it[v]; i < (int)g[v].size(); ++i) {
    Edge& e = g[v][i];
    if (e.cap > 0 && level[v] < level[e.to]) {
      int d = dfs(e.to, std::min(f, e.cap));
      if (d > 0) {
        e.cap -= d;
        g[e.to][e.rev].cap += d;
        return d;
      }
    }
  }
  return 0;
}

int D2DTransmissionOptimizer::MaxFlow::dinic() {
  int flow = 0;
  while (bfs()) {
    std::fill(it.begin(), it.end(), 0);
    int f;
    while ((f = dfs(s, 1e9)) > 0) {
      flow += f;
    }
  }
  return flow;
}

bool D2DTransmissionOptimizer::feasible(
    int K,
    const std::vector<int>& required_per_target,
    const std::unordered_map<int, std::vector<GlobalNpu>>& expert_to_src,
    std::vector<GlobalNpu>& npu_index_map,
    std::vector<int>& req_to_expert_id,
    MaxFlow& mf) {

  // 1. 建立 NPU 到索引的映射 (固定顺序)
  std::unordered_map<GlobalNpu, int, GlobalNpuHasher> npu_to_idx;
  npu_index_map.clear();
  for (const auto& [expert_id, sources] : expert_to_src) {
    for (const auto& gn : sources) {
      if (npu_to_idx.find(gn) == npu_to_idx.end()) {
        npu_to_idx[gn] = npu_index_map.size();
        npu_index_map.push_back(gn);
      }
    }
  }

  int npu_num = npu_index_map.size();
  int total_req = (int)required_per_target.size();
  
  // 2. 初始化流图
  // S: 0 | NPU nodes: 1..npu_num | Req nodes: npu_num+1..npu_num+total_req | T: npu_num+total_req+1
  int s = 0;
  int npu_offset = 1;
  int req_offset = npu_offset + npu_num;
  int t = req_offset + total_req;
  mf.init(t + 1, s, t);

  // S -> NPU (容量为 K，限制单卡最大输出)
  for (int i = 0; i < npu_num; ++i) {
    mf.add_edge(s, npu_offset + i, K);
  }

  // NPU -> Req & Req -> T
  req_to_expert_id.clear();
  for (int i = 0; i < total_req; ++i) {
    int expert_id = required_per_target[i];
    int cur_req_node = req_offset + i;
    req_to_expert_id.push_back(expert_id);

    if (expert_to_src.count(expert_id)) {
      for (const auto& gn : expert_to_src.at(expert_id)) {
        mf.add_edge(npu_offset + npu_to_idx[gn], cur_req_node, 1);
      }
    }
    mf.add_edge(cur_req_node, t, 1);
  }

  return mf.dinic() >= total_req;
}

std::vector<D2DTransmissionOptimizer::Step> D2DTransmissionOptimizer::extract_plan(
    const MaxFlow& mf,
    const std::vector<GlobalNpu>& npu_index_map,
    const std::vector<int>& req_to_expert_id,
    int npu_offset,
    int req_offset) {

  std::vector<Step> plan;
  // 遍历所有请求节点 (Request Nodes)
  for (int i = 0; i < (int)req_to_expert_id.size(); ++i) {
    int req_node = req_offset + i;
    int expert_id = req_to_expert_id[i];

    // 寻找哪个 NPU 节点流向了它
    for (const auto& edge : mf.g[req_node]) {
      // 在残余网络中，反向边 edge.cap > 0 代表正向边流量 > 0
      int u = edge.to;
      if (u >= npu_offset && u < npu_offset + (int)npu_index_map.size()) {
        // 进一步确认对应的正向边是否耗尽容量
        for (const auto& forward_e : mf.g[u]) {
          if (forward_e.to == req_node && forward_e.cap == 0) {
            plan.push_back({npu_index_map[u - npu_offset], expert_id});
            goto next_expert;
          }
        }
      }
    }
    next_expert:;
  }
  return plan;
}

std::vector<D2DTransmissionOptimizer::Step> D2DTransmissionOptimizer::optimize_layer(
    const std::vector<int>& required_per_target,
    const std::unordered_map<int, std::vector<GlobalNpu>>& expert_to_src) {

  int total_req = (int)required_per_target.size();
  if (total_req == 0) return {};

  int low = 1, high = total_req;
  int bestK = high;
  
  // 缓存最终结果所需的数据
  std::vector<GlobalNpu> final_npu_map;
  std::vector<int> final_req_to_expert;
  MaxFlow final_mf;

  while (low <= high) {
    int mid = low + (high - low) / 2;
    std::vector<GlobalNpu> tmp_npu_map;
    std::vector<int> tmp_req_expert;
    MaxFlow tmp_mf;

    if (feasible(mid, required_per_target, expert_to_src, tmp_npu_map, tmp_req_expert, tmp_mf)) {
      bestK = mid;
      high = mid - 1;
      // 记录当前最优解状态
      final_npu_map = std::move(tmp_npu_map);
      final_req_to_expert = std::move(tmp_req_expert);
      final_mf = std::move(tmp_mf);
    } else {
      low = mid + 1;
    }
  }

  return extract_plan(final_mf, final_npu_map, final_req_to_expert, 1, 1 + (int)final_npu_map.size());
}

D2DTransmissionOptimizer::NonExpertStep D2DTransmissionOptimizer::optimize_non_expert(
    const std::vector<Step>& expert_steps,
    const std::unordered_map<std::string, InstanceConfig>& instance_configs) {

  // 1. 统计每张 NPU 的专家传输负载
  std::unordered_map<std::string, std::vector<int>> inst_npu_loads;
  for (const auto& [name, config] : instance_configs) {
      inst_npu_loads[name].assign(config.device_size, 0);
  }

  for (const auto& s : expert_steps) {
      if (inst_npu_loads.count(s.src.instance)) {
          inst_npu_loads[s.src.instance][s.src.local_npu]++;
      }
  }

  // 2. 寻找负载最优（组内最大专家负载最小）的 DP 组
  std::string best_inst;
  int best_group_idx = -1;
  int best_dp_group_num = 0;
  int best_npu_per_group = 0;
  int min_max_load = 1e9;

  for (const auto& [inst_name, config] : instance_configs) {
      // 健壮性检查：防止除零或不合理的配置
      if (config.device_size <= 0 || config.dp_size <= 0) continue;
      if (config.device_size % config.dp_size != 0) continue;

      int num_dp_groups = config.dp_size;
      int npu_per_group = config.device_size / config.dp_size;
      const auto& loads = inst_npu_loads[inst_name];

      for (int g = 0; g < num_dp_groups; ++g) {
          int current_group_max_load = 0;
          
          // 计算该 DP 组内 NPU 的最大负载
          // 组内第 k 个 NPU 的负载: Load_{inst, g * npu_per_group + k}
          for (int i = 0; i < npu_per_group; ++i) {
              int local_npu = g * npu_per_group + i;
              if (local_npu < (int)loads.size()) {
                  current_group_max_load = std::max(current_group_max_load, loads[local_npu]);
              }
          }

          // 更新全局最优：寻找 max(load) 最小的组
          if (current_group_max_load < min_max_load) {
              min_max_load = current_group_max_load;
              best_inst = inst_name;
              best_group_idx = g;
              best_dp_group_num = config.dp_size;
              best_npu_per_group = npu_per_group;
          }
      }
  }

  // 3. 构建结果
  NonExpertStep result;
  if (best_group_idx != -1) {
      result.src_instance = best_inst;
      result.dp_group_index = best_group_idx;
      result.start_npu_index = best_group_idx * best_npu_per_group;
      result.dp_size = best_dp_group_num;
  }
  
  return result;
}
}
