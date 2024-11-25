#ifndef CGEMMA_BATCH_HPP
#define CGEMMA_BATCH_HPP

#include <lua.hpp>
#include <gemma/gemma.h>
#include <vector>
#include <unordered_map>
#include <string>

namespace cgemma {

int batch(lua_State* L);

class session;

struct session_context {
  explicit session_context(session* s);

  session* sess;
  std::vector<int> prompt;
  size_t start_pos;
  std::vector<int> output;
  int stream_fn = 0;
};

class batch_result {
public:
  batch_result(std::vector<session_context>&& sess_ctxs, gcpp::TimingInfo&& timing);

  const session_context* get(session* sess) const;
  const gcpp::TimingInfo& timing_info() const { return timing_; }

  static void declare(lua_State* L);
  static batch_result* check(lua_State* L, int index);

private:
  std::vector<session_context> sess_ctxs_;
  std::unordered_map<session*, const session_context*> sess2ctx_;
  gcpp::TimingInfo timing_;
};

}

#endif  // CGEMMA_BATCH_HPP
