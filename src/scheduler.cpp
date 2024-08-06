#include "scheduler.hpp"
#include <util/app.h>

namespace {

constexpr const char name[] = "cgemma.scheduler";

int cpu_topology(lua_State* L) {
  auto sched = cgemma::scheduler::check(L, 1);
  lua_newtable(L);
  size_t i = 0;
  for (auto& cluster: sched->pools().CoresPerCluster()) {
    lua_pushinteger(L, ++i);
    lua_newtable(L);
    size_t j = 0;
    cluster.Foreach([&](size_t cpu) {
      lua_pushinteger(L, ++j);
      lua_pushinteger(L, cpu);
      lua_settable(L, -3);
    });
    lua_settable(L, -3);
  }
  return 1;
}

int destroy(lua_State* L) {
  cgemma::scheduler::check(L, 1)->~scheduler();
  return 0;
}

}

namespace cgemma {

scheduler::scheduler()
  : pools_(0, 0) {
  // nop
}

scheduler::scheduler(size_t max_threads, size_t max_clusters)
  : pools_(max_clusters, max_threads) {
  // nop
}

void scheduler::declare(lua_State* L) {
  constexpr const luaL_Reg metatable[] = {
    {"__gc", destroy},
    {nullptr, nullptr}
  };
  constexpr const luaL_Reg methods[] = {
    {"cpu_topology", cpu_topology},
    {nullptr, nullptr}
  };
  luaL_newmetatable(L, name);
  luaL_register(L, nullptr, metatable);
  lua_pushlstring(L, name, sizeof(name) - 1);
  lua_setfield(L, -2, "_NAME");
  lua_newtable(L);
  luaL_register(L, nullptr, methods);
  lua_setfield(L, -2, "__index");
}

scheduler* scheduler::to(lua_State* L, int index) {
  return lua_isuserdata(L, index) && luaL_checkudata(L, index, name) ? static_cast<scheduler*>(lua_touserdata(L, index)) : nullptr;
}

scheduler* scheduler::check(lua_State* L, int index) {
  auto ud = scheduler::to(L, index);
  if (!ud) {
    luaL_error(L, "Bad argument #%d, %s expected", index, name);
  }
  return ud;
}

int scheduler::create(lua_State* L) {
  auto max_threads = lua_tointeger(L, 1);
  auto max_clusters = lua_tointeger(L, 2);
  auto ud = lua_newuserdata(L, sizeof(scheduler));
  try {
    new(ud) scheduler(max_threads, max_clusters);
    luaL_getmetatable(L, name);
    lua_setmetatable(L, -2);
    return 1;
  } catch (const std::exception& e) {
    lua_pop(L, 1);
    lua_pushnil(L);
    lua_pushstring(L, e.what());
    return 2;
  }
}

}
