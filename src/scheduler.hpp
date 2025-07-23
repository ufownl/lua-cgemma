#ifndef CGEMMA_SCHEDULER_HPP
#define CGEMMA_SCHEDULER_HPP

#include <lua.hpp>
#include <util/threading_context.h>
#include <ops/matmul.h>

namespace cgemma {

class scheduler {
public:
  scheduler();
  scheduler(int argc, char* argv[]);

  const char* cpu_topology() const { return ctx_.topology.TopologyString(); }
  gcpp::ThreadingContext& threading_ctx() { return ctx_; }
  gcpp::MatMulEnv& matmul_env() { return env_; }

  static void declare(lua_State* L);
  static scheduler* to(lua_State* L, int index);
  static scheduler* check(lua_State* L, int index);
  static int create(lua_State* L);

private:
  gcpp::ThreadingArgs args_;
  gcpp::ThreadingContext ctx_;
  gcpp::MatMulEnv env_;
};

}

#endif  // CGEMMA_SCHEDULER_HPP
