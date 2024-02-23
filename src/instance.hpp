#ifndef CGEMMA_INSTANCE_HPP
#define CGEMMA_INSTANCE_HPP

#include <lua.hpp>
#include <hwy/contrib/thread_pool/thread_pool.h>
#include <gemma.h>
#include <memory>

namespace cgemma {

class session;

class instance {
public:
  explicit instance(size_t num_threads, int argc, char* argv[]);

  hwy::ThreadPool& inner_pool() { return inner_pool_; }
  hwy::ThreadPool& pool() { return pool_; }
  gcpp::Gemma& model() const { return *model_; }

  std::unique_ptr<session> current_session;

  static void declare(lua_State* L);
  static instance* check(lua_State* L, int index);
  static int create(lua_State* L);

private:
  hwy::ThreadPool inner_pool_;
  hwy::ThreadPool pool_;
  gcpp::LoaderArgs args_;
  std::unique_ptr<gcpp::Gemma> model_;
};

}

#endif  // CGEMMA_INSTANCE_HPP
