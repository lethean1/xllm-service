#pragma once

#include <memory>
#include <mutex>
#include <cstdint>

#include "common/call_data.h"
#include "request/request.h"

namespace xllm_service {

// 最小 RequestContext：
// 1) 持有 call_data 与 request
// 2) 维护请求状态（未开始、运行中、完成、失败、取消）
class RequestContext {
 public:

  RequestContext(::xllm::proto::CompletionRequest* request_message,
                 std::shared_ptr<CallData> call_data,
                 std::shared_ptr<std::string> req_attachment,
                 std::shared_ptr<Request> request,
                 const std::string& method,
                 ::google::protobuf::Closure* done = nullptr)
      : call_data_(std::move(call_data)),
        request_message_(request_message),
        request_(std::move(request)),
        req_attachment_(std::move(req_attachment)),
        method_(method),
        done_(done),
        attempt_(0) {
          stream_ = request_->stream;
          // 将 done 的控制权交给 call_data 的生命周期管理
          // if (stream_) {
          //   done_->Run();
          // }
        }  

  ~RequestContext() {
    // if(!stream_) {
    //   done_->Run();
    // }
  }

  // 访问器
  ::xllm::proto::CompletionRequest* request_message() const { return request_message_; }
  std::shared_ptr<CallData> call_data() const { return call_data_; }
  std::shared_ptr<Request> request() const { return request_; }
  std::shared_ptr<std::string> req_attachment() const { return req_attachment_; }
  const std::string& method() const { return method_; }
  ::google::protobuf::Closure* done() { return done_; }
  int attempt() const { return attempt_; }
  void increment_attempt() { ++attempt_; }

  // 类型转换辅助：按需取得派生类型指针
  template <typename T_>
  std::shared_ptr<T_> call_data_as() const {
    return std::dynamic_pointer_cast<T_>(call_data_);
  }

  void set_req_attachment(std::shared_ptr<std::string> req_attachment) {
    std::lock_guard<std::mutex> lk(mu_);
    req_attachment_ = std::move(req_attachment);
  }
  bool is_instance_in_use(const std::string& instance_name) const {
    return request_->routing.prefill_name == instance_name || request_->routing.decode_name == instance_name;
  }
  bool is_instance_prefill_used(const std::string& instance_name) const {
    return request_->routing.prefill_name == instance_name;
  }
  bool is_instance_decode_used(const std::string& instance_name) const {
    return request_->routing.decode_name == instance_name;
  }

  // 类似 StreamCallData 提供接口
  // 由于使用了模板，没法直接作为成员变量实现
  // [TODO] 使用更优雅的方式实现
  bool write_and_finish(const std::string& attachment) {
    if (auto c = call_data_as<CompletionCallData>()) return c->write_and_finish(attachment);
    if (auto c = call_data_as<ChatCallData>()) return c->write_and_finish(attachment);
    return false;
  }

  bool write_and_finish(xllm::proto::CompletionResponse response) {
    if (auto c = call_data_as<CompletionCallData>()) return c->write_and_finish(response);
    return false;
  }

  bool write_and_finish(xllm::proto::ChatResponse response) {
    if (auto c = call_data_as<ChatCallData>()) return c->write_and_finish(response);
    return false;
  }

  bool finish_with_error(const std::string& error_message) {
    if (auto c = call_data_as<CompletionCallData>()) return c->finish_with_error(error_message);
    if (auto c = call_data_as<ChatCallData>()) return c->finish_with_error(error_message);
    return false;
  }

  bool write(const butil::IOBuf& attachment) {
    if (auto c = call_data_as<CompletionCallData>()) return c->write(attachment);
    if (auto c = call_data_as<ChatCallData>()) return c->write(attachment);
    return false;
  }

  bool write(const std::string& attachment) {
    if (auto c = call_data_as<CompletionCallData>()) return c->write(attachment);
    if (auto c = call_data_as<ChatCallData>()) return c->write(attachment);
    return false;
  }

  bool write(xllm::proto::CompletionResponse response) {
    if (auto c = call_data_as<CompletionCallData>()) return c->write(response);
    return false;
  }

  bool write(xllm::proto::ChatResponse response) {
    if (auto c = call_data_as<ChatCallData>()) return c->write(response);
    return false;
  }

  bool finish() {
    if (auto c = call_data_as<CompletionCallData>()) return c->finish();
    if (auto c = call_data_as<ChatCallData>()) return c->finish();
    return false;
  }

 private:
  mutable std::mutex mu_;
  ::xllm::proto::CompletionRequest* request_message_;
  std::shared_ptr<CallData> call_data_;
  std::shared_ptr<Request> request_;
  std::shared_ptr<std::string> req_attachment_;
  std::string method_;
  int attempt_;
  bool stream_;
  ::google::protobuf::Closure* done_;
};

}  // namespace xllm_service