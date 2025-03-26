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
  if (sess->inst()->model().Info().wrapping == gcpp::PromptWrapping::PALIGEMMA) {
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
  if (sess->inst()->model().Info().wrapping == gcpp::PromptWrapping::PALIGEMMA) {
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

class kv_cache_size_store {
public:
  kv_cache_size_store(const gcpp::ModelConfig& cfg, size_t pos) {
    store_[static_cast<size_t>(kv_cache_field::kv_cache)] = cfg.CachePosSize() * pos * sizeof(std::declval<gcpp::KVCache>().kv_cache[0]);
    auto griffin_layers = cfg.NumLayersOfType(gcpp::LayerAttentionType::kGriffinRecurrentBlock);
    decltype(std::declval<gcpp::LayerConfig>().conv1d_width) conv1d_width = 0;
    for (const auto& layer_cfg: cfg.layer_configs) {
      conv1d_width = std::max(conv1d_width, layer_cfg.conv1d_width);
    }
    store_[static_cast<size_t>(kv_cache_field::conv1d_cache)] = griffin_layers * (conv1d_width == 0 ? 0 : conv1d_width - 1) * cfg.model_dim * sizeof(std::declval<gcpp::KVCache>().conv1d_cache[0]);
    store_[static_cast<size_t>(kv_cache_field::rglru_cache)] = griffin_layers * cfg.model_dim * sizeof(std::declval<gcpp::KVCache>().rglru_cache[0]);
  }

  template <kv_cache_field Field>
  size_t get() const { return store_[static_cast<size_t>(Field)]; }

  size_t total() const {
    return std::accumulate(store_.begin() + 1, store_.end(), store_.front());
  }

private:
  std::array<size_t, static_cast<size_t>(kv_cache_field::end)> store_;
};

size_t dump_impl(char* buf, const cgemma::session* sess) {
  auto type = sess->inst()->model().Info().model;
  uint16_t pos = sess->pos();
  kv_cache_size_store size(sess->inst()->model().GetModelConfig(), std::min(sess->pos(), sess->kv_cache().seq_len));
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
  if (type != sess->inst()->model().Info().model) {
    throw std::invalid_argument("Invalid dump format: model type mismatch");
  }
  buf += sizeof(name);
  size_t pos = *reinterpret_cast<const uint16_t*>(buf);
  buf += sizeof(uint16_t);
  kv_cache_size_store size(sess->inst()->model().GetModelConfig(), std::min(pos, sess->kv_cache().seq_len));
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

int stats(lua_State* L) {
  cgemma::push_timing(L, cgemma::session::check(L, 1)->timing_info());
  return 1;
}

}

namespace cgemma {

session::session(instance* inst, int argc, char* argv[], bool no_wrapping)
  : inst_(inst)
  , args_(argc, argv)
  , no_wrapping_(no_wrapping) {
  if (auto err = args_.Validate()) {
    throw std::invalid_argument(err);
  }
  kv_cache_ = gcpp::KVCache::Create(inst->model().GetModelConfig(), args_.prefill_tbatch_size);
}

std::vector<int> session::tokenize(const char* text, size_t len) const {
  auto prompt = tokenize_text(std::string(text, len));
  if (!no_wrapping_ && inst_->instruction_tuned()) {
    return tokenize_wrap(prompt);
  } else {
    if (pos_ == 0) {
      prompt.insert(prompt.cbegin(), gcpp::BOS_ID);
    }
    return prompt;
  }
}

std::vector<int> session::tokenize(const gcpp::ImageTokens& image, const char* text, size_t len) const {
  auto text_part = tokenize_text(std::string(text, len));
  std::vector<int> prompt;
  switch (inst_->model().Info().wrapping) {
    case gcpp::PromptWrapping::PALIGEMMA: {
      std::vector<int> sep;
      if (!inst_->model().Tokenizer().Encode("\n", &sep)) {
        throw std::runtime_error("Tokenizer encoding failed. (session::tokenize)");
      }
      prompt.reserve(image.BatchSize() + 1 + text_part.size() + sep.size());
      prompt.resize(image.BatchSize(), PAD_ID);
      prompt.push_back(gcpp::BOS_ID);
      prompt.insert(prompt.cend(), text_part.cbegin(), text_part.cend());
      prompt.insert(prompt.cend(), sep.cbegin(), sep.cend());
      return prompt;
    }
    case gcpp::PromptWrapping::GEMMA_VLM: {
      std::vector<int> soi;
      soi.reserve(2);
      if (!inst_->model().Tokenizer().Encode("\n\n<start_of_image>", &soi)) {
        throw std::runtime_error("Tokenizer encoding failed. (session::tokenize)");
      }
      std::vector<int> eoi;
      eoi.reserve(2);
      if (!inst_->model().Tokenizer().Encode("<end_of_image>\n\n", &eoi)) {
        throw std::runtime_error("Tokenizer encoding failed. (session::tokenize)");
      }
      const auto prompt_size = text_part.size() + soi.size() + image.BatchSize() + eoi.size();
      if (no_wrapping_ && pos_ == 0) {
        prompt.reserve(1 + prompt_size);
        prompt.push_back(gcpp::BOS_ID);
      } else {
        prompt.reserve(prompt_size);
      }
      prompt.insert(prompt.cend(), text_part.cbegin(), text_part.cend());
      prompt.insert(prompt.cend(), soi.cbegin(), soi.cend());
      prompt.insert(prompt.cend(), image.BatchSize(), -2);
      prompt.insert(prompt.cend(), eoi.cbegin(), eoi.cend());
      return no_wrapping_ ? prompt : tokenize_wrap(prompt);
    }
    default:
      throw std::invalid_argument("Current variant does not support images.");
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

std::vector<int> session::tokenize_wrap(const std::vector<int>& input) const {
  std::vector<int> sot_user;
  sot_user.reserve(3);
  if (!inst_->model().Tokenizer().Encode("<start_of_turn>user\n", &sot_user)) {
    throw std::runtime_error("Tokenizer encoding failed. (session::tokenize_wrap)");
  }
  std::vector<int> sot_model;
  sot_model.reserve(3);
  if (!inst_->model().Tokenizer().Encode("<start_of_turn>model\n", &sot_model)) {
    throw std::runtime_error("Tokenizer encoding failed. (session::tokenize_wrap)");
  }
  std::vector<int> eot;
  eot.reserve(2);
  if (!inst_->model().Tokenizer().Encode("<end_of_turn>\n", &eot)) {
    throw std::runtime_error("Tokenizer encoding failed. (session::tokenize_wrap)");
  }
  std::vector<int> output;
  output.reserve(eot.size() + sot_user.size() + input.size() + eot.size() + sot_model.size());
  if (pos_ > 0) {
    output.insert(output.cend(), eot.cbegin(), eot.cend());
  } else {
    output.push_back(gcpp::BOS_ID);
  }
  output.insert(output.cend(), sot_user.cbegin(), sot_user.cend());
  output.insert(output.cend(), input.cbegin(), input.cend());
  output.insert(output.cend(), eot.cbegin(), eot.cend());
  output.insert(output.cend(), sot_model.cbegin(), sot_model.cend());
  return output;
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
