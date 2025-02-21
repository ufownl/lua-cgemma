#include "scheduler.hpp"
#include <util/allocator.h>

namespace {

constexpr const char name[] = "cgemma.scheduler";

int cpu_topology(lua_State* L) {
  auto sched = cgemma::scheduler::check(L, 1);
  lua_pushstring(L, sched->pools().TopologyString());
  return 1;
}

int destroy(lua_State* L) {
  cgemma::scheduler::check(L, 1)->~scheduler();
  return 0;
}

std::unique_ptr<gcpp::NestedPools> make_pools(const gcpp::AppArgs& args) {
  auto pools = std::make_unique<gcpp::NestedPools>(
    args.max_threads, args.pin,
    gcpp::BoundedSlice(args.skip_packages, args.max_packages),
    gcpp::BoundedSlice(args.skip_clusters, args.max_clusters),
    gcpp::BoundedSlice(args.skip_lps, args.max_lps)
  );
  gcpp::Allocator::Init(pools->Topology());
  return pools;
}

}

namespace cgemma {

scheduler::scheduler() {
  pools_ = make_pools(args_);
}

scheduler::scheduler(int args, char* argv[])
  : args_(args, argv) {
  pools_ = make_pools(args_);
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
  auto nargs = lua_gettop(L);
  constexpr const char* available_options[] = {
    "--num_threads", "--pin",
    "--skip_packages", "--max_packages",
    "--skip_clusters", "--max_clusters",
    "--skip_lps", "--max_lps",
  };
  constexpr const int n = sizeof(available_options) / sizeof(available_options[0]);
  int argc = 1;
  char* argv[n * 2 + 1] = {const_cast<char*>("lua-cgemma")};
  if (nargs > 0) {
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
  }
  auto ud = lua_newuserdata(L, sizeof(scheduler));
  try {
    new(ud) scheduler(argc, argv);
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
