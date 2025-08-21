#ifndef CGEMMA_SCHEDULER_HPP
#define CGEMMA_SCHEDULER_HPP

#include <lua.hpp>
#include <util/threading_context.h>
#include <ops/matmul.h>
#include <memory>

namespace cgemma {

class scheduler {
public:
  scheduler() { init(); }
  scheduler(int argc, char* argv[]) : args_(argc, argv) { init(); }

  const char* cpu_topology() const { return ctx_->topology.TopologyString(); }
  gcpp::ThreadingContext& threading_ctx() const { return *ctx_; }
  gcpp::MatMulEnv& matmul_env() const { return *env_; }

  static void declare(lua_State* L);
  static scheduler* to(lua_State* L, int index);
  static scheduler* check(lua_State* L, int index);
  static int create(lua_State* L);

private:
  void init() {
    ctx_ = std::make_unique<gcpp::ThreadingContext>(args_);
    env_ = std::make_unique<gcpp::MatMulEnv>(*ctx_);
  }

  gcpp::ThreadingArgs args_;
  std::unique_ptr<gcpp::ThreadingContext> ctx_;
  std::unique_ptr<gcpp::MatMulEnv> env_;
};

}

#endif  // CGEMMA_SCHEDULER_HPP
