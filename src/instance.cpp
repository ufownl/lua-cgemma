#include "instance.hpp"
#include "session.hpp"
#include <util/app.h>
#include <stdexcept>
#include <string>

namespace {

constexpr const char* name = "cgemma.instance";

int call(lua_State* L) {
  auto inst = cgemma::instance::check(L, 1);
  if (!inst->current_session) {
    lua_pushnil(L);
    lua_pushliteral(L, "Session has ended.");
    return 2;
  }
  try {
    std::vector<int> prompt;
    std::vector<int> output;
    size_t pos = 0;
    auto stream_token = [&](int token, float) {
      inst->current_session->incr_pos(1);
      ++pos;
      if (pos >= prompt.size() && token != gcpp::EOS_ID) {
        output.push_back(token);
      }
      return true;
    };
    std::string text(luaL_checkstring(L, 2));
    if (inst->model().model_training == gcpp::ModelTraining::GEMMA_IT) {
      text = "<start_of_turn>user\n" + text + "<end_of_turn>\n<start_of_turn>model\n";
      if (inst->current_session->pos() > 0) {
        text = "<end_of_turn>\n" + text;
      }
    }
    if (inst->current_session->pos() == 0) {
      prompt.push_back(2);
    }
    if (auto status = inst->model().Tokenizer().Encode(text, &prompt); !status.ok()) {
      lua_pushnil(L);
      lua_pushstring(L, status.ToString().c_str());
      return 2;
    }
    gcpp::GenerateGemma(inst->model(), inst->current_session->args(), prompt, inst->current_session->pos(), inst->pool(), inst->inner_pool(), stream_token, [](int) { return true; }, inst->current_session->rnd(), 0);
    if (inst->current_session->pos() >= inst->current_session->args().max_tokens) {
      inst->current_session = nullptr;
    }
    std::string resp;
    if (auto status = inst->model().Tokenizer().Decode(output, &resp); !status.ok()) {
      lua_pushnil(L);
      lua_pushstring(L, status.ToString().c_str());
      return 2;
    }
    if (auto i = resp.find_first_not_of(" \t\n"); i != std::string::npos) {
      lua_pushlstring(L, resp.data() + i, resp.size() - i);
    } else {
      lua_pushliteral(L, "");
    }
    return 1;
  } catch (const std::exception& e) {
    lua_pushnil(L);
    lua_pushstring(L, e.what());
    return 2;
  }
}

int destroy(lua_State* L) {
  cgemma::instance::check(L, 1)->~instance();
  return 0;
}

int start_session(lua_State* L) {
  auto nargs = lua_gettop(L);
  auto inst = cgemma::instance::check(L, 1);
  try {
    std::random_device rd;
    auto seed = rd();
    constexpr const char* available_options[] = {"--max_tokens", "--max_generated_tokens", "--temperature"};
    constexpr const int n = sizeof(available_options) / sizeof(available_options[0]);
    int argc = 1;
    char* argv[n * 2 + 1] = {const_cast<char*>("lua-cgemma")};
    if (nargs > 1) {
      luaL_checktype(L, 2, LUA_TTABLE);
      lua_getfield(L, 2, "seed");
      if (lua_isnumber(L, -1)) {
        seed = lua_tointeger(L, -1);
      }
      lua_pop(L, 1);
      for (auto opt: available_options) {
        auto k = opt + 2;
        lua_getfield(L, 2, k);
        auto v = lua_tostring(L, -1);
        if (v) {
          argv[argc++] = const_cast<char*>(opt);
          argv[argc++] = const_cast<char*>(v);
        }
        lua_pop(L, 1);
      }
    }
    inst->current_session = std::make_unique<cgemma::session>(seed, argc, argv);
    lua_pushboolean(L, 1);
    return 1;
  } catch (const std::exception& e) {
    lua_pushnil(L);
    lua_pushstring(L, e.what());
    return 2;
  }
}

int ready(lua_State* L) {
  auto inst = cgemma::instance::check(L, 1);
  lua_pushboolean(L, inst->current_session ? 1 : 0);
  return 1;
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
    {"__call", call},
    {"__gc", destroy},
    {nullptr, nullptr}
  };
  constexpr const luaL_Reg methods[] = {
    {"start_session", start_session},
    {"ready", ready},
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
