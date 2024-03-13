#include "session.hpp"
#include "instance.hpp"
#include "scheduler.hpp"
#include <stdexcept>
#include <vector>
#include <cstring>

namespace {

constexpr const char name[] = "cgemma.session";

std::vector<int> text2prompt(cgemma::session* sess, std::string text) {
  if (sess->inst()->model().model_training == gcpp::ModelTraining::GEMMA_IT) {
    text = "<start_of_turn>user\n" + text + "<end_of_turn>\n<start_of_turn>model\n";
    if (sess->pos() > 0) {
      text = "<end_of_turn>\n" + text;
    }
  }
  std::vector<int> prompt;
  if (sess->pos() == 0) {
    prompt.push_back(2);
  }
  if (auto status = sess->inst()->model().Tokenizer()->Encode(text, &prompt); !status.ok()) {
    throw status;
  }
  return prompt;
}

void generate(cgemma::session* sess, std::vector<int>& prompt, const gcpp::StreamFunc& stream_token) {
  gcpp::GenerateGemma(sess->inst()->model(), {
    .max_tokens = sess->args().max_tokens,
    .max_generated_tokens = sess->args().max_generated_tokens,
    .temperature = sess->args().temperature,
    .verbosity = 0
  }, prompt, sess->pos(), sess->kv_cache(), sess->inst()->sched().pool(), stream_token, sess->rnd());
}

int stream_mode(lua_State* L, cgemma::session* sess, const char* text) {
  try {
    auto prompt = text2prompt(sess, text);
    size_t pos = 0;
    generate(sess, prompt, [&](int token, float) {
      lua_pushvalue(L, 3);
      if (pos >= prompt.size() && token != gcpp::EOS_ID) {
        std::string token_text;
        if (auto status = sess->inst()->model().Tokenizer()->Decode(std::vector<int>{token}, &token_text); !status.ok()) {
          throw status;
        }
        lua_pushlstring(L, token_text.data(), token_text.size());
      } else {
        lua_pushnil(L);
      }
      lua_pushinteger(L, pos);
      lua_pushinteger(L, prompt.size());
      lua_call(L, 3, 1);
      auto res = lua_toboolean(L, -1);
      lua_pop(L, 1);
      sess->incr_pos(1);
      ++pos;
      return res ? true : false;
    });
    lua_pushboolean(L, 1);
    return 1;
  } catch (const sentencepiece::util::Status& status) {
    lua_pushnil(L);
    lua_pushstring(L, status.ToString().c_str());
    return 2;
  }
}

int normal_mode(lua_State* L, cgemma::session* sess, const char* text) {
  try {
    auto prompt = text2prompt(sess, text);
    std::vector<int> output;
    size_t pos = 0;
    generate(sess, prompt, [&](int token, float) {
      if (pos >= prompt.size() && token != gcpp::EOS_ID) {
        output.push_back(token);
      }
      sess->incr_pos(1);
      ++pos;
      return true;
    });
    std::string resp;
    if (auto status = sess->inst()->model().Tokenizer()->Decode(output, &resp); !status.ok()) {
      throw status;
    }
    lua_pushlstring(L, resp.data(), resp.size());
    return 1;
  } catch (const sentencepiece::util::Status& status) {
    lua_pushnil(L);
    lua_pushstring(L, status.ToString().c_str());
    return 2;
  }
}

int call(lua_State* L) {
  auto sess = cgemma::session::check(L, 1);
  if (sess->pos() >= sess->args().max_tokens) {
    lua_pushnil(L);
    lua_pushliteral(L, "Session has ended.");
    return 2;
  }
  try {
    auto text = luaL_checkstring(L, 2);
    return lua_isfunction(L, 3) ? stream_mode(L, sess, text) : normal_mode(L, sess, text);
  } catch (const std::exception& e) {
    lua_pushnil(L);
    lua_pushstring(L, e.what());
    return 2;
  }
}

int destroy(lua_State* L) {
  cgemma::session::check(L, 1)->~session();
  return 0;
}

int ready(lua_State* L) {
  auto sess = cgemma::session::check(L, 1);
  lua_pushboolean(L, sess->pos() < sess->args().max_tokens ? 1 : 0);
  return 1;
}

int reset(lua_State* L) {
  cgemma::session::check(L, 1)->set_pos(0);
  return 0;
}

template <class Config>
size_t kv_cache_size(size_t pos) {
  return Config::kLayers * Config::kKVHeads * Config::kQKVDim * pos;
}

size_t kv_cache_size(gcpp::Model type, size_t pos) {
  switch (type) {
    case gcpp::Model::GEMMA_2B:
      return kv_cache_size<gcpp::ConfigGemma2B>(pos);
    case gcpp::Model::GEMMA_7B:
      return kv_cache_size<gcpp::ConfigGemma7B>(pos);
    default:
      throw std::invalid_argument("Invalid model type.");
  }
}

int dump(lua_State* L) {
  auto ud = cgemma::session::check(L, 1);
  try {
    auto typ = ud->inst()->args().ModelType();
    auto len = kv_cache_size(typ, ud->pos()) * sizeof(ud->kv_cache().key_cache[0]);
    uint16_t pos = ud->pos();
    std::vector<uint8_t> buf(sizeof(name) + sizeof(pos) + 2 * len);
    std::memcpy(buf.data(), name, sizeof(name) - 1);
    buf[sizeof(name) - 1] = static_cast<uint8_t>(typ);
    std::memcpy(buf.data() + sizeof(name), &pos, sizeof(pos));
    if (len > 0) {
      std::memcpy(buf.data() + sizeof(name) + sizeof(pos), ud->kv_cache().key_cache.get(), len);
      std::memcpy(buf.data() + sizeof(name) + sizeof(pos) + len, ud->kv_cache().value_cache.get(), len);
    }
    lua_pushlstring(L, reinterpret_cast<const char*>(buf.data()), buf.size());
    return 1;
  } catch (const std::exception& e) {
    lua_pushnil(L);
    lua_pushstring(L, e.what());
    return 2;
  }
}

int restore(lua_State* L) {
  auto ud = cgemma::session::check(L, 1);
  size_t n;
  auto buf = luaL_checklstring(L, 2, &n);
  if (n < sizeof(name) + sizeof(uint16_t)) {
    lua_pushboolean(L, 0);
    lua_pushliteral(L, "Invalid dump format: length too short");
    return 2;
  }
  for (size_t i = 0; i < sizeof(name) - 1; ++i) {
    if (buf[i] != name[i]) {
      lua_pushboolean(L, 0);
      lua_pushliteral(L, "Invalid dump format: magic mismatch");
      return 2;
    }
  }
  auto typ = static_cast<gcpp::Model>(buf[sizeof(name) - 1]);
  if (typ != ud->inst()->args().ModelType()) {
    lua_pushboolean(L, 0);
    lua_pushliteral(L, "Invalid dump format: model type mismatch");
    return 2;
  }
  size_t pos = *reinterpret_cast<const uint16_t*>(buf + sizeof(name));
  try {
    auto len = kv_cache_size(typ, pos) * sizeof(ud->kv_cache().key_cache[0]);
    if (n != sizeof(name) + sizeof(uint16_t) + 2 * len) {
      lua_pushboolean(L, 0);
      lua_pushliteral(L, "Invalid dump format: KVCache length mismatch");
      return 2;
    }
    ud->set_pos(pos);
    if (len > 0) {
      std::memcpy(ud->kv_cache().key_cache.get(), buf + sizeof(name) + sizeof(uint16_t), len);
      std::memcpy(ud->kv_cache().value_cache.get(), buf + sizeof(name) + sizeof(uint16_t) + len, len);
    }
    lua_pushboolean(L, 1);
    return 1;
  } catch (const std::exception& e) {
    lua_pushboolean(L, 0);
    lua_pushstring(L, e.what());
    return 2;
  }
}

}

namespace cgemma {

session::session(const instance* inst, unsigned int seed, int argc, char* argv[])
  : inst_(inst)
  , rnd_(seed)
  , args_(argc, argv)
  , kv_cache_(gcpp::CreateKVCache(inst->args().ModelType())) {
  if (auto err = args_.Validate()) {
    throw std::invalid_argument(err);
  }
}

void session::declare(lua_State* L) {
  constexpr const luaL_Reg metatable[] = {
    {"__call", call},
    {"__gc", destroy},
    {nullptr, nullptr}
  };
  constexpr const luaL_Reg methods[] = {
    {"ready", ready},
    {"reset", reset},
    {"dump", dump},
    {"restore", restore},
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

session* session::check(lua_State* L, int index) {
  if (!lua_isuserdata(L, index) || !luaL_checkudata(L, index, name)) {
    luaL_error(L, "Bad argument #%d, %s expected", index, name);
  }
  return static_cast<session*>(lua_touserdata(L, index));
}

int session::create(lua_State* L) {
  auto nargs = lua_gettop(L);
  auto inst = instance::check(L, 1);
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
    auto ud = lua_newuserdata(L, sizeof(session));
    new(ud) session(inst, seed, argc, argv);
    luaL_getmetatable(L, name);
    lua_setmetatable(L, -2);
    lua_pushinteger(L, seed);
  } catch (const std::exception& e) {
    lua_pushnil(L);
    lua_pushstring(L, e.what());
  }
  return 2;
}

}
