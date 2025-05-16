#include "session.hpp"
#include "instance.hpp"
#include "image_tokens.hpp"
#include "utils/file_io.hpp"
#include <stdexcept>
#include <algorithm>
#include <utility>
#include <numeric>
#include <array>

namespace {

constexpr const char name[] = "cgemma.session";

void generate(cgemma::session* sess, const gcpp::ImageTokens* image, const std::vector<int>& prompt, const gcpp::BatchStreamFunc& stream_token) {
  gcpp::RuntimeConfig cfg;
  sess->args().CopyTo(cfg);
  cfg.verbosity = 0;
  cfg.gen = &sess->inst()->rnd();
  cfg.batch_stream_token = stream_token;
  if (!sess->inst()->disabled_tokens().empty()) {
    cfg.accept_token = [&](int token, float) {
      return sess->inst()->disabled_tokens().find(token) == sess->inst()->disabled_tokens().end();
    };
  }
  if (image) {
    cfg.prefill_tbatch_size = prompt.size();
    cfg.image_tokens = image;
    sess->inst()->model().Generate(cfg, gcpp::PromptTokens(prompt.data(), prompt.size()), sess->pos(), prompt.size(), sess->kv_cache(), sess->timing_info());
  } else {
    sess->inst()->model().Generate(cfg, gcpp::PromptTokens(prompt.data(), prompt.size()), sess->pos(), sess->kv_cache(), sess->timing_info());
  }
}

int stream_mode(lua_State* L, cgemma::session* sess, const gcpp::ImageTokens* image, const std::vector<int>& prompt, int stream_fn) {
  if (sess->inst()->model().GetModelConfig().wrapping == gcpp::PromptWrapping::PALIGEMMA) {
    sess->set_pos(0);
  }
  auto start_pos = sess->pos();
  auto prompt_size = prompt.size();
  std::vector<int> output(1);
  generate(sess, image, prompt, [&](size_t, size_t pos, int token, float) {
    auto eot = false;
    lua_pushvalue(L, stream_fn);
    if (pos - start_pos < prompt_size) {
      lua_pushnil(L);
    } else if (sess->inst()->eos(token)) {
      eot = true;
      lua_pushnil(L);
    } else {
      output.front() = token;
      std::string token_text;
      if (!sess->inst()->model().Tokenizer().Decode(output, &token_text)) {
        throw std::runtime_error("Tokenizer decoding failed. (stream_mode)");
      }
      lua_pushlstring(L, token_text.data(), token_text.size());
    }
    lua_pushinteger(L, pos - start_pos);
    lua_pushinteger(L, prompt_size);
    lua_call(L, 3, 1);
    auto res = lua_toboolean(L, -1);
    lua_pop(L, 1);
    if (!eot && res) {
      sess->set_pos(pos);
      return true;
    } else {
      return false;
    }
  });
  lua_pushboolean(L, 1);
  return 1;
}

int normal_mode(lua_State* L, cgemma::session* sess, const gcpp::ImageTokens* image, const std::vector<int>& prompt) {
  if (sess->inst()->model().GetModelConfig().wrapping == gcpp::PromptWrapping::PALIGEMMA) {
    sess->set_pos(0);
  }
  auto start_pos = sess->pos();
  auto prompt_size = prompt.size();
  std::vector<int> output;
  output.reserve(sess->args().max_generated_tokens);
  generate(sess, image, prompt, [&](size_t, size_t pos, int token, float) {
    if (pos - start_pos >= prompt_size) {
      if (sess->inst()->eos(token)) {
        return false;
      }
      output.push_back(token);
    }
    sess->set_pos(pos);
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
  if (sess->pos() >= sess->inst()->max_tokens()) {
    lua_pushnil(L);
    lua_pushliteral(L, "Session has ended.");
    return 2;
  }
  try {
    size_t len;
    auto image = cgemma::image_tokens::to(L, 2);
    auto offset = image ? 2 : 1;
    auto text = luaL_checklstring(L, 1 + offset, &len);
    auto prompt = image ? sess->tokenize(*image, text, len) : sess->tokenize(text, len);
    return lua_isfunction(L, 2 + offset) ? stream_mode(L, sess, image, prompt, 2 + offset) : normal_mode(L, sess, image, prompt);
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
  lua_pushboolean(L, sess->pos() < sess->inst()->max_tokens() ? 1 : 0);
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

template <class T>
class kv_cache_blob {
public:
  template <class U>
  kv_cache_blob(U sess, size_t resumed_pos = 0) {
    auto pos_size = sess->inst()->model().GetModelConfig().CachePosSize();
    if (pos_size > 0) {
      auto pos = std::min(resumed_pos ? resumed_pos : sess->pos(), sess->kv_cache().seq_len);
      ptrs_[static_cast<size_t>(kv_cache_field::kv_cache)] = sess->kv_cache().kv_cache.get();
      sizes_[static_cast<size_t>(kv_cache_field::kv_cache)] = pos_size * pos * sizeof(std::declval<gcpp::KVCache>().kv_cache[0]);
    } else {
      ptrs_[static_cast<size_t>(kv_cache_field::kv_cache)] = nullptr;
      sizes_[static_cast<size_t>(kv_cache_field::kv_cache)] = 0;
    }
#define GRIFFIN_CACHE(FIELD)                                                                \
  do {                                                                                      \
    if (sess->kv_cache().griffin_layers) {                                                  \
      auto span = sess->kv_cache().FIELD.Span();                                            \
      ptrs_[static_cast<size_t>(kv_cache_field::FIELD)] = span.ptr;                         \
      sizes_[static_cast<size_t>(kv_cache_field::FIELD)] = span.num * sizeof(span.ptr[0]);  \
    } else {                                                                                \
      ptrs_[static_cast<size_t>(kv_cache_field::FIELD)] = nullptr;                          \
      sizes_[static_cast<size_t>(kv_cache_field::FIELD)] = 0;                               \
    }                                                                                       \
  } while (false)
    GRIFFIN_CACHE(conv1d_cache);
    GRIFFIN_CACHE(rglru_cache);
#undef GRIFFIN_CACHE
  }

  template <kv_cache_field Field>
  T buffer() const { return ptrs_[static_cast<size_t>(Field)]; }

  template <kv_cache_field Field>
  size_t size() const { return sizes_[static_cast<size_t>(Field)]; }

  size_t total_size() const {
    return std::accumulate(sizes_.begin() + 1, sizes_.end(), sizes_.front());
  }

private:
  std::array<T, static_cast<size_t>(kv_cache_field::end)> ptrs_;
  std::array<size_t, static_cast<size_t>(kv_cache_field::end)> sizes_;
};

size_t dump_impl(char* buf, const cgemma::session* sess) {
  auto type = sess->inst()->model().GetModelConfig().model;
  uint16_t pos = sess->pos();
  kv_cache_blob<const void*> blob(sess);
  if (buf) {
    std::memcpy(buf, name, sizeof(name) - 1);
    buf[sizeof(name) - 1] = static_cast<char>(type);
    buf += sizeof(name);
    std::memcpy(buf, &pos, sizeof(pos));
    buf += sizeof(pos);
#define DUMP_CACHE(FIELD)                                                                         \
  do {                                                                                            \
    if (blob.buffer<kv_cache_field::FIELD>()) {                                                   \
      std::memcpy(buf, blob.buffer<kv_cache_field::FIELD>(), blob.size<kv_cache_field::FIELD>()); \
      buf += blob.size<kv_cache_field::FIELD>();                                                  \
    }                                                                                             \
  } while (false)
    DUMP_CACHE(kv_cache);
    DUMP_CACHE(conv1d_cache);
    DUMP_CACHE(rglru_cache);
#undef DUMP_CACHE
  }
  return sizeof(name) + sizeof(pos) + blob.total_size();
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
  if (type != sess->inst()->model().GetModelConfig().model) {
    throw std::invalid_argument("Invalid dump format: model type mismatch");
  }
  buf += sizeof(name);
  size_t pos = *reinterpret_cast<const uint16_t*>(buf);
  buf += sizeof(uint16_t);
  kv_cache_blob<void*> blob(sess, pos);
  if (n != sizeof(name) + sizeof(uint16_t) + blob.total_size()) {
    throw std::invalid_argument("Invalid dump format: KVCache length mismatch");
  }
  sess->set_pos(pos);
#define LOAD_CACHE(FIELD)                                                                         \
  do {                                                                                            \
    if (blob.buffer<kv_cache_field::FIELD>()) {                                                   \
      std::memcpy(blob.buffer<kv_cache_field::FIELD>(), buf, blob.size<kv_cache_field::FIELD>()); \
      buf += blob.size<kv_cache_field::FIELD>();                                                  \
    }                                                                                             \
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

int stats(lua_State* L) {
  cgemma::push_timing(L, cgemma::session::check(L, 1)->timing_info());
  return 1;
}

}

namespace cgemma {

session::session(instance* inst, int argc, char* argv[], bool no_wrapping)
  : inst_(inst)
  , args_(argc, argv)
  , no_wrapping_(no_wrapping)
  , kv_cache_(inst->model().GetModelConfig(), args_.prefill_tbatch_size) {
  // nop
}

std::vector<int> session::tokenize(const char* text, size_t len) const {
  auto prompt = tokenize_text(std::string(text, len));
  if (!no_wrapping_ && inst_->instruction_tuned()) {
    return inst_->model().ChatTemplate().Apply(pos_, prompt);
  } else {
    if (pos_ == 0) {
      prompt.insert(prompt.cbegin(), gcpp::BOS_ID);
    }
    return prompt;
  }
}

std::vector<int> session::tokenize(const gcpp::ImageTokens& image, const char* text, size_t len) const {
  auto text_part = tokenize_text(std::string(text, len));
  switch (inst_->model().GetModelConfig().wrapping) {
    case gcpp::PromptWrapping::PALIGEMMA:
      return inst_->model().ChatTemplate().WrapPali(text_part, image.Rows());
    case gcpp::PromptWrapping::GEMMA_VLM: {
      auto prompt = inst_->model().ChatTemplate().WrapVLM(text_part, image.Rows());
      return no_wrapping_ ? prompt : inst_->model().ChatTemplate().Apply(pos_, prompt);
    }
    default:
      throw std::invalid_argument("Current variant does not support vision prompt.");
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
    {"stats", stats},
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
  return static_cast<session*>(luaL_checkudata(L, index, name));
}

int session::create(lua_State* L) {
  auto nargs = lua_gettop(L);
  auto inst = instance::check(L, 1);
  constexpr const char* available_options[] = {
    "--max_generated_tokens",
    "--prefill_tbatch",
    "--decode_qbatch",
    "--temperature",
    "--top_k"
  };
  constexpr const int n = sizeof(available_options) / sizeof(available_options[0]);
  int argc = 1;
  char* argv[n * 2 + 1] = {const_cast<char*>("lua-cgemma")};
  bool no_wrapping = false;
  if (nargs >= 2) {
    luaL_checktype(L, 2, LUA_TTABLE);
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
    lua_getfield(L, 2, "no_wrapping");
    no_wrapping = lua_toboolean(L, -1) ? true : false;
    lua_pop(L, 1);
  }
  auto ud = lua_newuserdata(L, sizeof(session));
  try {
    auto sess = new(ud) session(inst, argc, argv, no_wrapping);
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

std::vector<int> session::tokenize_text(const std::string& text) const {
  std::vector<int> prompt;
  const auto max_prompt_tokens = inst_->max_tokens() - args_.max_generated_tokens;
  prompt.reserve(max_prompt_tokens > pos_ + 64 ? max_prompt_tokens - pos_ : 64);
  if (!inst_->model().Tokenizer().Encode(text, &prompt)) {
    throw std::runtime_error("Tokenizer encoding failed. (session::tokenize_text)");
  }
  if (!inst_->disabled_tokens().empty()) {
    std::replace_if(prompt.begin(), prompt.end(), [&](int token) {
      return inst_->disabled_tokens().find(token) != inst_->disabled_tokens().end();
    }, UNK_ID);
  }
  return prompt;
}

void push_timing(lua_State*L, const gcpp::TimingInfo& timing) {
  lua_newtable(L);
  lua_pushnumber(L, timing.prefill_duration);
  lua_setfield(L, -2, "prefill_duration");
  lua_pushinteger(L, timing.prefill_tokens);
  lua_setfield(L, -2, "prefill_tokens");
  lua_pushnumber(L, timing.time_to_first_token);
  lua_setfield(L, -2, "time_to_first_token");
  lua_pushnumber(L, timing.generate_duration);
  lua_setfield(L, -2, "generate_duration");
  lua_pushinteger(L, timing.tokens_generated);
  lua_setfield(L, -2, "tokens_generated");
  lua_pushnumber(L, timing.prefill_tokens / timing.prefill_duration);
  lua_setfield(L, -2, "prefill_tokens_per_second");
  lua_pushnumber(L, timing.tokens_generated / timing.generate_duration);
  lua_setfield(L, -2, "generate_tokens_per_second");
}

}
