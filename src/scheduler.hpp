#ifndef CGEMMA_SCHEDULER_HPP
#define CGEMMA_SCHEDULER_HPP

#include <lua.hpp>
#include <util/threading.h>

namespace cgemma {

class scheduler {
public:
  scheduler();
  scheduler(size_t max_threads, size_t max_clusters, int pin_threads);

  gcpp::PerClusterPools& pools() { return pools_; }

  static void declare(lua_State* L);
  static scheduler* to(lua_State* L, int index);
  static scheduler* check(lua_State* L, int index);
  static int create(lua_State* L);

private:
  gcpp::PerClusterPools pools_;
};

}

#endif  // CGEMMA_SCHEDULER_HPP
