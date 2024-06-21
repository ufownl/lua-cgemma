#include "session.hpp"
#include "instance.hpp"
#include "scheduler.hpp"
#include "utils/file_io.hpp"
#include <stdexcept>
#include <algorithm>
#include <utility>
#include <numeric>
#include <vector>
#include <array>
#include <cstring>

namespace {

constexpr const char name[] = "cgemma.session";

std::vector<int> text2prompt(cgemma::session* sess, const char* text) {
  constexpr const char user_sot[] = "<start_of_turn>user\n";
  constexpr const char model_sot[] = "<start_of_turn>model\n";
  constexpr const char eot[] = "<end_of_turn>\n";
  std::string s;
  if (sess->inst()->args().ModelTrainingType() == gcpp::ModelTraining::GEMMA_IT) {
    s.reserve(sizeof(eot) - 1
            + sizeof(user_sot) - 1
            + std::strlen(text)
            + sizeof(eot) - 1
            + sizeof(model_sot) - 1);
    if (sess->pos() > 0) {
      s.append(eot);
    }
    s.append(user_sot);
    s.append(text);
    s.append(eot);
    s.append(model_sot);
  } else {
    s.append(text, std::strlen(text));
  }
  std::vector<int> prompt;
  if (!sess->inst()->model().Tokenizer().Encode(s, &prompt)) {
    throw std::runtime_error("Tokenizer encoding failed. (text2prompt)");
  }
  if (sess->pos() == 0) {
    prompt.insert(prompt.cbegin(), gcpp::BOS_ID);
  }
  return prompt;
}

void generate(cgemma::session* sess, std::vector<int>& prompt, const gcpp::StreamFunc& stream_token) {
  gcpp::TimingInfo timing_info;
  sess->inst()->model().Generate({
    .max_tokens = sess->args().max_tokens,
    .max_generated_tokens = sess->args().max_generated_tokens,
    .temperature = sess->args().temperature,
    .verbosity = 0,
    .gen = &sess->rnd(),
    .stream_token = stream_token
  }, prompt, sess->pos(), sess->kv_cache(), timing_info);
}

int stream_mode(lua_State* L, cgemma::session* sess, const char* text) {
  auto prompt = text2prompt(sess, text);
  size_t pos = 0;
  generate(sess, prompt, [&](int token, float) {
    lua_pushvalue(L, 3);
    if (pos >= prompt.size() && token != gcpp::EOS_ID) {
      std::string token_text;
      if (!sess->inst()->model().Tokenizer().Decode(std::vector<int>{token}, &token_text)) {
        throw std::runtime_error("Tokenizer decoding failed. (stream_mode)");
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
    if (res) {
      sess->incr_pos(1);
      ++pos;
      return true;
    } else {
      return false;
    }
  });
  lua_pushboolean(L, 1);
  return 1;
}

int normal_mode(lua_State* L, cgemma::session* sess, const char* text) {
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
  if (!sess->inst()->model().Tokenizer().Decode(output, &resp)) {
    throw std::runtime_error("Tokenizer decoding failed. (normal_mode)");
  }
  lua_pushlstring(L, resp.data(), resp.size());
  return 1;
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

enum class kv_cache_field: size_t {
  kv_cache,
  conv1d_cache,
  rglru_cache,
  end
};

class kv_cache_size_store {
public:
  template <class Config>
  kv_cache_size_store(const Config&, size_t pos) {
    gcpp::CachePosSize<Config> pos_size;
    store_[static_cast<size_t>(kv_cache_field::kv_cache)] = pos_size() * pos * sizeof(std::declval<gcpp::KVCache>().kv_cache[0]);
    store_[static_cast<size_t>(kv_cache_field::conv1d_cache)] = Config::kGriffinLayers * std::max(Config::kConv1dWidth - 1, 0) * Config::kModelDim * sizeof(std::declval<gcpp::KVCache>().conv1d_cache[0]);
    store_[static_cast<size_t>(kv_cache_field::rglru_cache)] = Config::kGriffinLayers * Config::kModelDim * sizeof(std::declval<gcpp::KVCache>().rglru_cache[0]);
  }

  template <kv_cache_field Field>
  size_t get() const { return store_[static_cast<size_t>(Field)]; }

  size_t total() const {
    return std::accumulate(store_.begin() + 1, store_.end(), store_.front());
  }

private:
  std::array<size_t, static_cast<size_t>(kv_cache_field::end)> store_;
};

template <class Config>
struct kv_cache_size {
  kv_cache_size_store operator()(size_t pos) {
    return kv_cache_size_store(Config(), pos);
  }
};

size_t dump_impl(char* buf, const cgemma::session* sess) {
  auto type = sess->inst()->args().ModelType();
  auto size = gcpp::CallForModel<void, kv_cache_size>(type, sess->pos());
  uint16_t pos = sess->pos();
  if (buf) {
    std::memcpy(buf, name, sizeof(name) - 1);
    buf[sizeof(name) - 1] = static_cast<char>(type);
    buf += sizeof(name);
    std::memcpy(buf, &pos, sizeof(pos));
    buf += sizeof(pos);
#define DUMP_CACHE(FIELD)                                                                 \
  do {                                                                                    \
    if (size.get<kv_cache_field::FIELD>() > 0) {                                          \
      std::memcpy(buf, sess->kv_cache().FIELD.get(), size.get<kv_cache_field::FIELD>());  \
      buf += size.get<kv_cache_field::FIELD>();                                           \
    }                                                                                     \
  } while (false)
    DUMP_CACHE(kv_cache);
    DUMP_CACHE(conv1d_cache);
    DUMP_CACHE(rglru_cache);
#undef DUMP_CACHE
  }
  return sizeof(name) + sizeof(pos) + size.total();
}

void load_impl(cgemma::session* sess, const char* buf, size_t n) {
  if (n < sizeof(name) + sizeof(uint16_t)) {
    throw std::invalid_argument("Invalid dump format: length too short");
  }
  for (size_t i = 0; i < sizeof(name) - 1; ++i) {
    if (buf[i] != name[i]) {
      throw std::invalid_argument("Invalid dump format: magic mismatch");
    }
  }
  auto type = static_cast<gcpp::Model>(buf[sizeof(name) - 1]);
  if (type != sess->inst()->args().ModelType()) {
    throw std::invalid_argument("Invalid dump format: model type mismatch");
  }
  buf += sizeof(name);
  size_t pos = *reinterpret_cast<const uint16_t*>(buf);
  buf += sizeof(uint16_t);
  auto size = gcpp::CallForModel<void, kv_cache_size>(type, pos);
  if (n != sizeof(name) + sizeof(uint16_t) + size.total()) {
    throw std::invalid_argument("Invalid dump format: KVCache length mismatch");
  }
  sess->set_pos(pos);
#define LOAD_CACHE(FIELD)                                                                 \
  do {                                                                                    \
    if (size.get<kv_cache_field::FIELD>() > 0) {                                          \
      std::memcpy(sess->kv_cache().FIELD.get(), buf, size.get<kv_cache_field::FIELD>());  \
      buf += size.get<kv_cache_field::FIELD>();                                           \
    }                                                                                     \
  } while (false)
  LOAD_CACHE(kv_cache);
  LOAD_CACHE(conv1d_cache);
  LOAD_CACHE(rglru_cache);
#undef LOAD_CACHE
}

int dumps(lua_State* L) {
  auto ud = cgemma::session::check(L, 1);
  try {
    std::vector<char> buf(dump_impl(nullptr, ud));
    dump_impl(buf.data(), ud);
    lua_pushlstring(L, buf.data(), buf.size());
    return 1;
  } catch (const std::exception& e) {
    lua_pushnil(L);
    lua_pushstring(L, e.what());
    return 2;
  }
}

int loads(lua_State* L) {
  auto ud = cgemma::session::check(L, 1);
  size_t n;
  auto buf = luaL_checklstring(L, 2, &n);
  try {
    load_impl(ud, buf, n);
    lua_pushboolean(L, 1);
    return 1;
  } catch (const std::exception& e) {
    lua_pushboolean(L, 0);
    lua_pushstring(L, e.what());
    return 2;
  }
}

int dump(lua_State* L) {
  auto ud = cgemma::session::check(L, 1);
  auto path = luaL_checkstring(L, 2);
  try {
    cgemma::utils::file_writer fout(path, dump_impl(nullptr, ud));
    dump_impl(fout.buffer(), ud);
    lua_pushboolean(L, 1);
    return 1;
  } catch (const std::exception& e) {
    lua_pushboolean(L, 0);
    lua_pushstring(L, e.what());
    return 2;
  }
}

int load(lua_State* L) {
  auto ud = cgemma::session::check(L, 1);
  auto path = luaL_checkstring(L, 2);
  try {
    cgemma::utils::file_reader fin(path);
    load_impl(ud, fin.buffer(), fin.size());
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
  , kv_cache_(gcpp::KVCache::Create(inst->args().ModelType())) {
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
    {"dumps", dumps},
    {"loads", loads},
    {"dump", dump},
    {"load", load},
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
