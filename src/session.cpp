#include "session.hpp"
#include "instance.hpp"
#include "image.hpp"
#include "utils/file_io.hpp"
#include <stdexcept>
#include <algorithm>
#include <utility>
#include <numeric>
#include <array>

namespace {

constexpr const char name[] = "cgemma.session";

void generate(cgemma::session* sess, const std::vector<int>& prompt, const gcpp::BatchStreamFunc& stream_token) {
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
  cfg.image_tokens = sess->image_tokens();
  size_t start_pos = sess->pos() > 0 ? sess->pos() + 1 : 0;
  if (cfg.image_tokens) {
    std::vector<int> image_prompt;
    image_prompt.reserve(cfg.image_tokens->BatchSize() + prompt.size());
    image_prompt.resize(cfg.image_tokens->BatchSize(), cgemma::PAD_ID);
    image_prompt.insert(image_prompt.cend(), prompt.cbegin(), prompt.cend());
    cfg.prefill_tbatch_size = image_prompt.size();
    sess->inst()->model().Generate(cfg, gcpp::PromptTokens(image_prompt.data(), image_prompt.size()), start_pos, image_prompt.size(), sess->kv_cache(), sess->timing_info());
  } else {
    sess->inst()->model().Generate(cfg, gcpp::PromptTokens(prompt.data(), prompt.size()), start_pos, sess->kv_cache(), sess->timing_info());
  }
}

int stream_mode(lua_State* L, cgemma::session* sess, const std::vector<int>& prompt) {
  if (sess->image_tokens()) {
    sess->set_pos(0);
  }
  auto start_pos = sess->pos();
  auto prompt_size = prompt.size();
  if (sess->image_tokens()) {
    prompt_size += sess->image_tokens()->BatchSize();
  }
  std::vector<int> output(1);
  generate(sess, prompt, [&](size_t, size_t pos, int token, float) {
    auto eot = false;
    lua_pushvalue(L, 3);
    if (pos - start_pos < prompt_size) {
      lua_pushnil(L);
    } else if (token == gcpp::EOS_ID || sess->inst()->model().Info().training == gcpp::ModelTraining::GEMMA_IT && token == cgemma::EOT_ID) {
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

int normal_mode(lua_State* L, cgemma::session* sess, const std::vector<int>& prompt) {
  if (sess->image_tokens()) {
    sess->set_pos(0);
  }
  auto start_pos = sess->pos();
  auto prompt_size = prompt.size();
  if (sess->image_tokens()) {
    prompt_size += sess->image_tokens()->BatchSize();
  }
  std::vector<int> output;
  output.reserve(sess->args().max_generated_tokens);
  generate(sess, prompt, [&](size_t, size_t pos, int token, float) {
    if (pos - start_pos >= prompt_size) {
      if (token == gcpp::EOS_ID || sess->inst()->model().Info().training == gcpp::ModelTraining::GEMMA_IT && token == cgemma::EOT_ID) {
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
    auto text = luaL_checklstring(L, 2, &len);
    auto prompt = sess->tokenize(text, len);
    return lua_isfunction(L, 3) ? stream_mode(L, sess, prompt) : normal_mode(L, sess, prompt);
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
    size_t conv1d_width = 0;
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
  auto img = sess->image_tokens();
  kv_cache_size_store size(sess->inst()->model().GetModelConfig(), std::min(sess->pos(), sess->kv_cache().seq_len));
  if (buf) {
    std::memcpy(buf, name, sizeof(name) - 1);
    buf[sizeof(name) - 1] = static_cast<char>(type);
    buf += sizeof(name);
    std::memcpy(buf, &pos, sizeof(pos));
    buf += sizeof(pos);
    if (img) {
      std::memcpy(buf, img->Const(), img->NumBytes());
      buf += img->NumBytes();
    }
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
  return sizeof(name) + sizeof(pos) + (img ? img->NumBytes() : 0) + size.total();
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
  auto img = sess->image_tokens();
  kv_cache_size_store size(sess->inst()->model().GetModelConfig(), std::min(pos, sess->kv_cache().seq_len));
  if (n != sizeof(name) + sizeof(uint16_t) + (img ? img->NumBytes() : 0) + size.total()) {
    throw std::invalid_argument("Invalid dump format: KVCache length mismatch");
  }
  sess->set_pos(pos);
  if (img) {
    std::memcpy(img->All(), buf, img->NumBytes());
    buf += img->NumBytes();
  }
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

session::session(instance* inst, int argc, char* argv[])
  : inst_(inst)
  , args_(argc, argv) {
  if (auto err = args_.Validate()) {
    throw std::invalid_argument(err);
  }
  const auto& cfg = inst->model().GetModelConfig();
  if (cfg.vit_seq_len > 0) {
    img_ = gcpp::ImageTokens(gcpp::Extents2D(cfg.vit_seq_len, cfg.model_dim));
  }
  kv_cache_ = gcpp::KVCache::Create(cfg, args_.prefill_tbatch_size);
}

std::vector<int> session::tokenize(const char* text, size_t len) const {
  constexpr const char user_sot[] = "<start_of_turn>user\n";
  constexpr const char model_sot[] = "<start_of_turn>model\n";
  constexpr const char eot[] = "<end_of_turn>\n";
  std::string s;
  if (inst_->model().Info().training == gcpp::ModelTraining::GEMMA_IT) {
    s.reserve(sizeof(eot) - 1
            + sizeof(user_sot) - 1
            + len
            + sizeof(eot) - 1
            + sizeof(model_sot) - 1);
    if (pos_ > 0) {
      s.append(eot, sizeof(eot) - 1);
    }
    s.append(user_sot, sizeof(user_sot) - 1);
    s.append(text, len);
    s.append(eot, sizeof(eot) - 1);
    s.append(model_sot, sizeof(model_sot) - 1);
  } else {
    s.append(text, len);
  }
  std::vector<int> prompt;
  const auto max_prompt_tokens = inst_->max_tokens() - args_.max_generated_tokens;
  prompt.reserve(max_prompt_tokens > pos_ + 64 ? max_prompt_tokens - pos_ : 64);
  if (!inst_->model().Tokenizer().Encode(s, &prompt)) {
    throw std::runtime_error("Tokenizer encoding failed. (session::tokenize)");
  }
  if (!inst_->disabled_tokens().empty()) {
    std::replace_if(prompt.begin(), prompt.end(), [&](int token) {
      return inst_->disabled_tokens().find(token) != inst_->disabled_tokens().end();
    }, UNK_ID);
  }
  if (pos_ == 0) {
    prompt.emplace(prompt.cbegin(), gcpp::BOS_ID);
  }
  if (inst_->model().Info().training == gcpp::ModelTraining::PALIGEMMA) {
    std::vector<int> sep;
    if (!inst_->model().Tokenizer().Encode("\n", &sep)) {
      throw std::runtime_error("Tokenizer encoding failed. (session::tokenize)");
    }
    prompt.insert(prompt.cend(), sep.cbegin(), sep.cend());
  }
  return prompt;
}

void session::embed(const gcpp::Image& img) {
  gcpp::RuntimeConfig cfg;
  args_.CopyTo(cfg);
  cfg.verbosity = 0;
  cfg.gen = &inst_->rnd();
  inst_->model().GenerateImageTokens(cfg, img, img_);
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
  if (!lua_isuserdata(L, index) || !luaL_checkudata(L, index, name)) {
    luaL_error(L, "Bad argument #%d, %s expected", index, name);
  }
  return static_cast<session*>(lua_touserdata(L, index));
}

int session::create(lua_State* L) {
  auto nargs = lua_gettop(L);
  auto inst = instance::check(L, 1);
  const gcpp::Image* img = image::to(L, 2);
  auto opt_idx = img ? 3 : 2;
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
  if (nargs >= opt_idx) {
    luaL_checktype(L, opt_idx, LUA_TTABLE);
    for (auto opt: available_options) {
      auto k = opt + 2;
      lua_getfield(L, opt_idx, k);
      auto v = lua_tostring(L, -1);
      if (v) {
        argv[argc++] = const_cast<char*>(opt);
        argv[argc++] = const_cast<char*>(v);
      }
      lua_pop(L, 1);
    }
  }
  auto ud = lua_newuserdata(L, sizeof(session));
  try {
    auto sess = new(ud) session(inst, argc, argv);
    if (sess->image_tokens() && img) {
      sess->embed(*img);
    }
    luaL_getmetatable(L, name);
    lua_setmetatable(L, -2);
    return 1;
  } catch (const std::exception& e) {
    lua_pushnil(L);
    lua_pushstring(L, e.what());
    return 2;
  }
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
