#ifndef CGEMMA_SESSION_HPP
#define CGEMMA_SESSION_HPP

#include <lua.hpp>
#include <gemma.h>
#include <util/app.h>
#include <random>

namespace cgemma {

class instance;

class session {
public:
  explicit session(const instance* inst, unsigned int seed, int argc, char* argv[]);

  const instance* inst() { return inst_; }
  std::mt19937& rnd() { return rnd_; }
  const gcpp::InferenceArgs& args() const { return args_; }
  size_t pos() const { return pos_; }
  gcpp::KVCache& kv_cache() { return kv_cache_; }

  void set_pos(size_t pos) { pos_ = pos; }
  void incr_pos(size_t n) { pos_ += n; }

  static void declare(lua_State* L);
  static session* check(lua_State* L, int index);
  static int create(lua_State* L);

private:
  const instance* inst_;
  std::mt19937 rnd_;
  gcpp::InferenceArgs args_;
  size_t pos_ {0};
  gcpp::KVCache kv_cache_;
};

}

#endif  // CGEMMA_SESSION_HPP
