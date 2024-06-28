#ifndef CGEMMA_SCHEDULER_HPP
#define CGEMMA_SCHEDULER_HPP

#include <lua.hpp>
#include <hwy/contrib/thread_pool/thread_pool.h>

namespace cgemma {

class scheduler {
public:
  scheduler();
  explicit scheduler(size_t num_threads);

  hwy::ThreadPool& pool() { return pool_; }

  static void declare(lua_State* L);
  static scheduler* to(lua_State* L, int index);
  static scheduler* check(lua_State* L, int index);
  static int create(lua_State* L);

private:
  hwy::ThreadPool pool_;
};

}

#endif  // CGEMMA_SCHEDULER_HPP
