#ifndef CGEMMA_INSTANCE_HPP
#define CGEMMA_INSTANCE_HPP

#include <lua.hpp>
#include <gemma/gemma.h>
#include <util/app.h>
#include <unordered_set>
#include <random>
#include <memory>

namespace cgemma {

constexpr const int PAD_ID = 0;
constexpr const int UNK_ID = 3;
constexpr const int EOT_ID = 107;

class scheduler;
class session;

class instance {
public:
  instance(int argc, char* argv[], unsigned int seed, scheduler* sched);

  const gcpp::LoaderArgs& args() const { return args_; }
  std::mt19937& rnd() { return rnd_; }
  gcpp::Gemma& model() const { return *model_; }
  const std::unordered_set<int>& disabled_tokens() const { return disabled_tokens_; }
  size_t max_tokens() const;

  static void declare(lua_State* L);
  static instance* check(lua_State* L, int index);
  static int create(lua_State* L);

private:
  gcpp::LoaderArgs args_;
  std::mt19937 rnd_;
  std::unique_ptr<scheduler> default_sched_;
  std::unique_ptr<gcpp::Gemma> model_;
  std::unordered_set<int> disabled_tokens_;
};

}

#endif  // CGEMMA_INSTANCE_HPP
