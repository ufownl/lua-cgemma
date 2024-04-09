#include "cgemma.hpp"
#include "instance.hpp"
#include "session.hpp"
#include "scheduler.hpp"

namespace {

template <gcpp::Model MODEL_T>
int compress_weights(lua_State* L) {
  auto w = luaL_checkstring(L, 1);
  auto cw = luaL_checkstring(L, 2);
  auto s = cgemma::scheduler::to(L, 3);
  try {
    std::unique_ptr<cgemma::scheduler> default_sched;
    if (!s) {
      default_sched = std::make_unique<cgemma::scheduler>();
      s = default_sched.get();
    }
    CompressWeights(MODEL_T, gcpp::Path(w), gcpp::Path(cw), s->pool());
    lua_pushboolean(L, 1);
    return 1;
  } catch (const std::exception& e) {
    lua_pushboolean(L, 0);
    lua_pushstring(L, e.what());
    return 2;
  }
}

}

int luaopen_cgemma(lua_State* L) {
  constexpr const luaL_Reg entries[] = {
    {"new", cgemma::instance::create},
    {"scheduler", cgemma::scheduler::create},
    {"compress_2b_weights", compress_weights<gcpp::Model::GEMMA_2B>},
    {"compress_7b_weights", compress_weights<gcpp::Model::GEMMA_7B>},
    {nullptr, nullptr}
  };
  cgemma::instance::declare(L);
  cgemma::session::declare(L);
  cgemma::scheduler::declare(L);
  lua_newtable(L);
  luaL_register(L, nullptr, entries);
  lua_pushliteral(L, "cgemma");
  lua_setfield(L, -2, "_NAME");
  lua_pushliteral(L, "1.0");
  lua_setfield(L, -2, "_VERSION");
  return 1;
}
