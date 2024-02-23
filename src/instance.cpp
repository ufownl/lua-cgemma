#include "instance.hpp"
#include <util/app.h>
#include <stdexcept>

namespace {

constexpr const char* name = "cgemma.instance";

int destroy(lua_State* L) {
  cgemma::instance::check(L, 1)->~instance();
  return 0;
}

}

namespace cgemma {

instance::instance(size_t num_threads, int argc, char* argv[])
  : inner_pool_(0)
  , pool_(num_threads)
  , args_(argc, argv) {
  if (auto err = args_.Validate()) {
    throw std::invalid_argument(err);
  }
  if (num_threads > 10) {
    gcpp::PinThreadToCore(num_threads - 1);
    pool_.Run(0, pool_.NumThreads(), [](uint64_t, size_t thread) { gcpp::PinThreadToCore(thread); });
  }
  model_ = std::make_unique<gcpp::Gemma>(args_, pool_);
}

void instance::declare(lua_State* L) {
  constexpr const luaL_Reg metatable[] = {
    {"__gc", destroy},
    {nullptr, nullptr}
  };
  constexpr const luaL_Reg methods[] = {
    {nullptr, nullptr}
  };
  luaL_newmetatable(L, name);
  luaL_register(L, nullptr, metatable);
  lua_pushstring(L, name);
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
  lua_getfield(L, 1, "num_threads");
  auto num_threads = lua_tointeger(L, -1);
  lua_pop(L, 1);
  constexpr const char* required_options[] = {"--tokenizer", "--model", "--compressed_weights"};
  constexpr const int n = sizeof(required_options) / sizeof(required_options[0]);
  char* argv[n * 2 + 1] = {const_cast<char*>("lua-cgemma")};
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
  auto ud = lua_newuserdata(L, sizeof(instance));
  try {
    new(ud) instance(num_threads, n * 2 + 1, argv);
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
