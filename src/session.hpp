#ifndef CGEMMA_SESSION_HPP
#define CGEMMA_SESSION_HPP

#include <lua.hpp>
#include <gemma/gemma.h>
#include <gemma/gemma_args.h>
#include <paligemma/image.h>
#include <string>
#include <vector>

namespace cgemma {

class instance;

class session {
public:
  session(instance* inst, int argc, char* argv[], bool no_wrapping);

  instance* inst() const { return inst_; }
  const gcpp::InferenceArgs& args() const { return args_; }
  size_t pos() const { return pos_; }
  gcpp::KVCache& kv_cache() const { return *kv_cache_; }
  const gcpp::TimingInfo& timing_info() const { return timing_info_; }
  gcpp::TimingInfo& timing_info() { return timing_info_; }

  void set_pos(size_t pos) { pos_ = pos; }

  std::vector<int> tokenize(const char* text, size_t len) const;
  std::vector<int> tokenize(const gcpp::ImageTokens& image, const char* text, size_t len) const;
  void embed(const gcpp::Image& img);

  static void declare(lua_State* L);
  static session* check(lua_State* L, int index);
  static int create(lua_State* L);

private:
  std::vector<int> tokenize_text(const std::string& text) const;

  instance* inst_;
  gcpp::InferenceArgs args_;
  bool no_wrapping_;
  size_t pos_ {0};
  std::unique_ptr<gcpp::KVCache> kv_cache_;
  gcpp::TimingInfo timing_info_;
};

void push_timing(lua_State*L, const gcpp::TimingInfo& timing);

}

#endif  // CGEMMA_SESSION_HPP
