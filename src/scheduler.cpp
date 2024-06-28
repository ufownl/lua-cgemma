#include "scheduler.hpp"
#include <util/app.h>
#include <algorithm>
#include <thread>

namespace {

constexpr const char name[] = "cgemma.scheduler";

int pin_threads(lua_State* L) {
  gcpp::PinWorkersToCores(cgemma::scheduler::check(L, 1)->pool());
  return 0;
}

int destroy(lua_State* L) {
  cgemma::scheduler::check(L, 1)->~scheduler();
  return 0;
}

}

namespace cgemma {

scheduler::scheduler()
  : pool_(std::min(static_cast<size_t>(std::thread::hardware_concurrency()), gcpp::kMaxThreads)) {
  // nop
}

scheduler::scheduler(size_t num_threads)
  : pool_(std::min(num_threads, gcpp::kMaxThreads)) {
  // nop
}

void scheduler::declare(lua_State* L) {
  constexpr const luaL_Reg metatable[] = {
    {"__gc", destroy},
    {nullptr, nullptr}
  };
  constexpr const luaL_Reg methods[] = {
    {"pin_threads", pin_threads},
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
  auto num_threads = lua_tointeger(L, 1);
  auto ud = lua_newuserdata(L, sizeof(scheduler));
  try {
    if (num_threads > 0) {
      new(ud) scheduler(num_threads);
    } else {
      new(ud) scheduler();
    }
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
