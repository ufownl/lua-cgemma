#ifndef CGEMMA_SCHEDULER_HPP
#define CGEMMA_SCHEDULER_HPP

#include <lua.hpp>
#include <util/app.h>
#include <util/threading.h>
#include <ops/matmul.h>
#include <memory>

namespace cgemma {

class scheduler {
public:
  scheduler();
  scheduler(int argc, char* argv[]);

  const char* cpu_topology() const { return topology_->TopologyString(); }
  gcpp::MatMulEnv& env() { return *env_; }

  static void declare(lua_State* L);
  static scheduler* to(lua_State* L, int index);
  static scheduler* check(lua_State* L, int index);
  static int create(lua_State* L);

private:
  void init();

  gcpp::AppArgs args_;
  std::unique_ptr<const gcpp::BoundedTopology> topology_;
  std::unique_ptr<gcpp::NestedPools> pools_;
  std::unique_ptr<gcpp::MatMulEnv> env_;
};

}

#endif  // CGEMMA_SCHEDULER_HPP
