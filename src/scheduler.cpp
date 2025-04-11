#include "scheduler.hpp"
#include "utils/laux.hpp"
#include <util/threading_context.h>

namespace {

constexpr const char name[] = "cgemma.scheduler";

int config(lua_State* L) {
  if (gcpp::ThreadingContext2::IsInitialized()) {
    lua_pushnil(L);
    lua_pushstring(L, "Scheduler had been initialized.");
    return 2;
  }
  constexpr const char* available_options[] = {
    "--num_threads", "--pin", "--bind",
    "--skip_packages", "--max_packages",
    "--skip_clusters", "--max_clusters",
    "--skip_lps", "--max_lps",
  };
  constexpr const int n = sizeof(available_options) / sizeof(available_options[0]);
  int argc = 1;
  char* argv[n * 2 + 1] = {const_cast<char*>("lua-cgemma")};
  luaL_checktype(L, 1, LUA_TTABLE);
  for (auto opt: available_options) {
    auto k = opt + 2;
    lua_getfield(L, 1, k);
    auto v = lua_tostring(L, -1);
    if (v) {
      argv[argc++] = const_cast<char*>(opt);
      argv[argc++] = const_cast<char*>(v);
    }
    lua_pop(L, 1);
  }
  gcpp::ThreadingContext2::SetArgs(gcpp::ThreadingArgs(argc, argv));
  if (gcpp::ThreadingContext2::IsInitialized()) {
    lua_pushnil(L);
    lua_pushstring(L, "Scheduler had been initialized.");
    return 2;
  }
  lua_pushboolean(L, 1);
  return 1;
}

int cpu_topology(lua_State* L) {
  try {
    lua_pushstring(L, gcpp::ThreadingContext2::Get().topology.TopologyString());
    return 1;
  } catch (const std::exception& e) {
    lua_pushnil(L);
    lua_pushstring(L, e.what());
    return 2;
  }
}

}

namespace cgemma { namespace scheduler {

void declare(lua_State* L) {
  constexpr const luaL_Reg entries[] = {
    {"config", config},
    {"cpu_topology", cpu_topology},
    {nullptr, nullptr}
  };
  lua_newtable(L);
  luaL_register(L, nullptr, entries);
  lua_pushliteral(L, "cgemma.scheduler");
  lua_setfield(L, -2, "_NAME");
}

} }
