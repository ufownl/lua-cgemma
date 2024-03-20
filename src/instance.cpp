#include "instance.hpp"
#include "scheduler.hpp"
#include "session.hpp"
#include <stdexcept>

namespace {

constexpr const char name[] = "cgemma.instance";

int destroy(lua_State* L) {
  cgemma::instance::check(L, 1)->~instance();
  return 0;
}

}

namespace cgemma {

instance::instance(int argc, char* argv[], scheduler* s)
  : args_(argc, argv)
  , sched_(s) {
  if (auto err = args_.Validate()) {
    throw std::invalid_argument(err);
  }
  if (!s) {
    default_sched_ = std::make_unique<scheduler>();
    sched_ = default_sched_.get();
  }
  model_ = std::make_unique<gcpp::Gemma>(args_.tokenizer, args_.compressed_weights, args_.weights, args_.ModelType(), args_.ModelTraining(), sched_->pool());
}

void instance::declare(lua_State* L) {
  constexpr const luaL_Reg metatable[] = {
    {"__gc", destroy},
    {nullptr, nullptr}
  };
  constexpr const luaL_Reg methods[] = {
    {"session", session::create},
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

instance* instance::check(lua_State* L, int index) {
  if (!lua_isuserdata(L, index) || !luaL_checkudata(L, index, name)) {
    luaL_error(L, "Bad argument #%d, %s expected", index, name);
  }
  return static_cast<instance*>(lua_touserdata(L, index));
}

int instance::create(lua_State* L) {
  luaL_checktype(L, 1, LUA_TTABLE);
  lua_getfield(L, 1, "scheduler");
  auto s = scheduler::to(L, -1);
  lua_pop(L, 1);
  constexpr const char* required_options[] = {"--tokenizer", "--model", "--compressed_weights"};
  constexpr const int n = sizeof(required_options) / sizeof(required_options[0]);
  constexpr const char* optional_options[] = {"--weights"};
  constexpr const int m = sizeof(optional_options) / sizeof(optional_options[0]);
  char* argv[(n + m) * 2 + 1] = {const_cast<char*>("lua-cgemma")};
  for (int i = 0; i < n; ++i) {
    auto k = required_options[i] + 2;
    lua_getfield(L, 1, k);
    auto v = lua_tostring(L, -1);
    if (!v) {
      luaL_error(L, "Option %s is required", k);
    }
    argv[i * 2 + 1] = const_cast<char*>(required_options[i]);
    argv[i * 2 + 2] = const_cast<char*>(v);
    lua_pop(L, 1);
  }
  auto argc = n * 2 + 1;
  for (auto opt: optional_options) {
    auto k = opt + 2;
    lua_getfield(L, 1, k);
    auto v = lua_tostring(L, -1);
    if (v) {
      argv[argc++] = const_cast<char*>(opt);
      argv[argc++] = const_cast<char*>(v);
    }
    lua_pop(L, 1);
  }
  auto ud = lua_newuserdata(L, sizeof(instance));
  try {
    new(ud) instance(argc, argv, s);
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
