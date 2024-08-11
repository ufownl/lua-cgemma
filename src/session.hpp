#ifndef CGEMMA_SESSION_HPP
#define CGEMMA_SESSION_HPP

#include <lua.hpp>
#include <gemma/gemma.h>
#include <util/app.h>
#include <vector>

namespace cgemma {

class instance;

class session {
public:
  session(instance* inst, int argc, char* argv[]);

  instance* inst() const { return inst_; }
  const gcpp::InferenceArgs& args() const { return args_; }
  size_t pos() const { return pos_; }
  const gcpp::KVCache& kv_cache() const { return kv_cache_; }
  gcpp::KVCache& kv_cache() { return kv_cache_; }
  const gcpp::TimingInfo& timing_info() const { return timing_info_; }
  gcpp::TimingInfo& timing_info() { return timing_info_; }

  void set_pos(size_t pos) { pos_ = pos; }
  void incr_pos(size_t n) { pos_ += n; }

  std::vector<int> tokenize(const char* text, size_t len) const;

  static void declare(lua_State* L);
  static session* check(lua_State* L, int index);
  static int create(lua_State* L);

private:
  instance* inst_;
  gcpp::InferenceArgs args_;
  size_t pos_ {0};
  gcpp::KVCache kv_cache_;
  gcpp::TimingInfo timing_info_;
};

void push_timing(lua_State*L, const gcpp::TimingInfo& timing);

}

#endif  // CGEMMA_SESSION_HPP
