#include "cgemma.hpp"
#include "instance.hpp"
#include "session.hpp"
#include "scheduler.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <cctype>

namespace {

gcpp::Model model_type(const char* m) {
  std::string model(m);
  std::transform(model.begin(), model.end(), model.begin(), [](unsigned char c) {
    return std::tolower(c);
  });
  if (model.substr(0, 2) == "2b") {
    return gcpp::Model::GEMMA_2B;
  } else if (model.substr(0, 2) == "7b") {
    return gcpp::Model::GEMMA_7B;
  } else {
    throw std::invalid_argument("Model type must be 2b-pt, 7b-pt, 2b-it, 7b-it.");
  }
}

int compress_weights(lua_State* L) {
  auto m = luaL_checkstring(L, 1);
  auto w = luaL_checkstring(L, 2);
  auto cw = luaL_checkstring(L, 3);
  auto s = cgemma::scheduler::to(L, 4);
  try {
    std::unique_ptr<cgemma::scheduler> default_sched;
    if (!s) {
      default_sched = std::make_unique<cgemma::scheduler>();
      s = default_sched.get();
    }
    CompressWeights(model_type(m), gcpp::Path(w), gcpp::Path(cw), s->pool());
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
    {"compress_weights", compress_weights},
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
