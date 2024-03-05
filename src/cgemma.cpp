#include "cgemma.hpp"
#include "instance.hpp"
#include "session.hpp"
#include "scheduler.hpp"

int luaopen_cgemma(lua_State* L) {
  constexpr const luaL_Reg entries[] = {
    {"new", cgemma::instance::create},
    {"scheduler", cgemma::scheduler::create},
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
