#ifndef CGEMMA_INSTANCE_HPP
#define CGEMMA_INSTANCE_HPP

#include <lua.hpp>
#include <gemma/gemma.h>
#include <util/app.h>
#include <memory>

namespace cgemma {

class scheduler;
class session;

class instance {
public:
  explicit instance(int argc, char* argv[], scheduler* s);

  const gcpp::LoaderArgs& args() const { return args_; }
  scheduler& sched() const { return *sched_; }
  gcpp::Gemma& model() const { return *model_; }

  static void declare(lua_State* L);
  static instance* check(lua_State* L, int index);
  static int create(lua_State* L);

private:
  gcpp::LoaderArgs args_;
  scheduler* sched_;
  std::unique_ptr<scheduler> default_sched_;
  std::unique_ptr<gcpp::Gemma> model_;
};

}

#endif  // CGEMMA_INSTANCE_HPP
