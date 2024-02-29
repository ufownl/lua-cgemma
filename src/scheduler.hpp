#ifndef CGEMMA_SCHEDULER_HPP
#define CGEMMA_SCHEDULER_HPP

#include <lua.hpp>
#include <hwy/contrib/thread_pool/thread_pool.h>

namespace cgemma {

class scheduler {
public:
  scheduler();
  explicit scheduler(size_t num_threads) : inner_pool_(0), pool_(num_threads) { pin_threads(); };

  hwy::ThreadPool& inner_pool() { return inner_pool_; }
  hwy::ThreadPool& pool() { return pool_; }

  static void declare(lua_State* L);
  static scheduler* to(lua_State* L, int index);
  static scheduler* check(lua_State* L, int index);
  static int create(lua_State* L);

private:
  void pin_threads();

  hwy::ThreadPool inner_pool_;
  hwy::ThreadPool pool_;
};

}

#endif  // CGEMMA_SCHEDULER_HPP
