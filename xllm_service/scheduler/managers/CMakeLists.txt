include(cc_library)
include(cc_test)

cc_library(
  NAME
   managers
  HDRS
    instance_mgr.h
    global_kvcache_mgr.h
    d2d_transmission_optimizer.h
  SRCS
    instance_mgr.cpp
    global_kvcache_mgr.cpp
    d2d_transmission_optimizer.cpp
  DEPS
    :chat_template
    :common
    :etcd_client
    :request
    absl::random_random
    absl::strings
    glog::glog
    proto::proto_rpc_service
    proto_xllm
)
target_link_libraries(managers PRIVATE brpc-static)
