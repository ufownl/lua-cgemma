#include "batch.hpp"
#include "instance.hpp"
#include "session.hpp"
#include "image_tokens.hpp"
#include <tuple>
#include <stdexcept>

namespace {

class batch_data_holder {
public:
  explicit batch_data_holder(const std::vector<cgemma::session_context>& sess_ctxs)
    : sess_ctxs_(sess_ctxs) {
    prompts_.reserve(sess_ctxs.size());
    start_pos_.reserve(sess_ctxs.size());
    prefix_end_.reserve(sess_ctxs.size());
    kv_caches_.reserve(sess_ctxs.size());
    for (auto& ctx: sess_ctxs) {
      prompts_.emplace_back(ctx.prompt.data(), ctx.prompt.size());
      start_pos_.emplace_back(ctx.start_pos);
      prefix_end_.emplace_back(ctx.prefix_end);
      kv_caches_.emplace_back(std::move(ctx.sess->kv_cache()));
    }
  }

  ~batch_data_holder() {
    for (size_t i = 0; i < kv_caches_.size(); ++i) {
      sess_ctxs_[i].sess->kv_cache() = std::move(kv_caches_[i]);
    }
  }

  gcpp::QueriesPromptTokens prompts() {
    return gcpp::QueriesPromptTokens(prompts_.data(), prompts_.size());
  }

  gcpp::QueriesPos start_pos() {
    return gcpp::QueriesPos(start_pos_.data(), start_pos_.size());
  }

  gcpp::QueriesPos prefix_end() {
    return gcpp::QueriesPos(prefix_end_.data(), prefix_end_.size());
  }

  gcpp::KVCaches kv_caches() {
    return gcpp::KVCaches(kv_caches_.data(), kv_caches_.size());
  }

private:
  const std::vector<cgemma::session_context>& sess_ctxs_;
  std::vector<gcpp::PromptTokens> prompts_;
  std::vector<size_t> start_pos_;
  std::vector<size_t> prefix_end_;
  std::vector<gcpp::KVCache> kv_caches_;
};

int init_arg_state(lua_State* L, int narg, const gcpp::ImageTokens* image, std::vector<cgemma::session_context>& sess_ctxs) {
  auto sess = cgemma::session::check(L, narg);
  if (sess->inst()->model().GetModelConfig().wrapping == gcpp::PromptWrapping::PALIGEMMA) {
    sess->set_pos(0);
  } else if (sess->pos() >= sess->inst()->max_tokens()) {
    throw std::invalid_argument("Sessions in a batch must not be ended.");
  }
  if (!sess_ctxs.empty()) {
    if (sess_ctxs.front().sess->inst() != sess->inst()) {
      throw std::invalid_argument("Sessions in a batch must be created by the same cgemma instance.");
    }
    for (const auto& ctx: sess_ctxs) {
      if (ctx.sess == sess) {
        throw std::invalid_argument("Sessions in a batch must not be duplicated.");
      }
    }
  }
  sess_ctxs.emplace_back(sess);
  return 1;
}

std::tuple<const gcpp::ImageTokens*, std::vector<cgemma::session_context>> parse_args(lua_State* L) {
  constexpr decltype(init_arg_state)* const arg_states[] = {
    init_arg_state,
    [](lua_State* L, int narg, const gcpp::ImageTokens* image, std::vector<cgemma::session_context>& sess_ctxs) {
      size_t len;
      auto text = luaL_checklstring(L, narg, &len);
      auto& ctx = sess_ctxs.back();
      if (image) {
        ctx.prompt = ctx.sess->tokenize(*image, text, len);
        ctx.prefix_end = ctx.prompt.size();
      } else {
        ctx.prompt = ctx.sess->tokenize(text, len);
      }
      return 2;
    },
    [](lua_State* L, int narg, const gcpp::ImageTokens* image, std::vector<cgemma::session_context>& sess_ctxs) {
      auto& ctx = sess_ctxs.back();
      if (lua_isfunction(L, narg)) {
        ctx.output.resize(1);
        ctx.stream_fn = narg;
        return 0;
      } else {
        ctx.output.reserve(ctx.sess->args().max_generated_tokens);
        return init_arg_state(L, narg, image, sess_ctxs);
      }
    }
  };
  auto image = cgemma::image_tokens::to(L, 1);
  auto offset = image ? 1 : 0;
  auto nargs = lua_gettop(L);
  if (nargs < 2 + offset) {
    luaL_error(L, "Too few arguments, at least %d expected", 2 + offset);
  }
  std::vector<cgemma::session_context> sess_ctxs;
  sess_ctxs.reserve((nargs + offset) / 2);
  int arg_state = 0;
  for (auto i = 1 + offset; i <= nargs; ++i) {
    arg_state = arg_states[arg_state](L, i, image, sess_ctxs);
  }
  if (sess_ctxs.back().prompt.empty()) {
    luaL_error(L, "Too few arguments, %d expected", nargs + 1);
  }
  return {image, std::move(sess_ctxs)};
}

gcpp::RuntimeConfig parse_config(const std::vector<cgemma::session_context>& sess_ctxs) {
  gcpp::RuntimeConfig cfg;
  cfg.max_generated_tokens = 8192 - 1;
  cfg.prefill_tbatch_size = 4096;
  cfg.decode_qbatch_size = 4096;
  cfg.temperature = 0.0f;
  cfg.top_k = 1;
  for (const auto& ctx: sess_ctxs) {
    cfg.max_generated_tokens = std::min(cfg.max_generated_tokens, ctx.sess->args().max_generated_tokens);
    cfg.prefill_tbatch_size = std::min(cfg.prefill_tbatch_size, ctx.sess->args().prefill_tbatch_size);
    cfg.decode_qbatch_size = std::min(cfg.decode_qbatch_size, ctx.sess->args().decode_qbatch_size);
    cfg.temperature += ctx.sess->args().temperature;
    cfg.top_k = std::max(cfg.top_k, ctx.sess->args().top_k);
  }
  cfg.temperature /= sess_ctxs.size();
  return cfg;
}

gcpp::TimingInfo generate(cgemma::instance* inst, const std::vector<cgemma::session_context>& sess_ctxs, const gcpp::RuntimeConfig& cfg) {
  gcpp::TimingInfo timing;
  batch_data_holder bdh(sess_ctxs);
  inst->model().GenerateBatch(cfg, bdh.prompts(), bdh.start_pos(), bdh.prefix_end(), bdh.kv_caches(), timing);
  return timing;
}

constexpr const char name[] = "cgemma.batch_result";

int call(lua_State* L) {
  auto res = cgemma::batch_result::check(L, 1);
  auto sess = cgemma::session::check(L, 2);
  auto ctx = res->get(sess);
  if (!ctx) {
    lua_pushnil(L);
    lua_pushliteral(L, "No corresponding result.");
    return 2;
  }
  if (ctx->stream_fn > 0) {
    lua_pushboolean(L, 1);
  } else {
    std::string resp;
    if (!sess->inst()->model().Tokenizer().Decode(ctx->output, &resp)) {
      lua_pushnil(L);
      lua_pushliteral(L, "Tokenizer decoding failed.");
      return 2;
    }
    lua_pushlstring(L, resp.data(), resp.size());
  }
  return 1;
}

int destroy(lua_State* L) {
  cgemma::batch_result::check(L, 1)->~batch_result();
  return 0;
}

int stats(lua_State* L) {
  cgemma::push_timing(L, cgemma::batch_result::check(L, 1)->timing_info());
  return 1;
}

}

namespace cgemma {

int batch(lua_State* L) {
  try {
    const gcpp::ImageTokens* image;
    std::vector<cgemma::session_context> sess_ctxs;
    std::tie(image, sess_ctxs) = parse_args(L);
    auto cfg = parse_config(sess_ctxs);
    cfg.verbosity = 0;
    auto inst = sess_ctxs.front().sess->inst();
    cfg.gen = &inst->rnd();
    cfg.batch_stream_token = [&](size_t query_idx, size_t pos, int token, float) {
      auto& ctx = sess_ctxs[query_idx];
      if (ctx.stream_fn == 0) {
        if (pos - ctx.start_pos >= ctx.prompt.size()) {
          if (inst->eos(token)) {
            return false;
          }
          ctx.output.push_back(token);
        }
        ctx.sess->set_pos(pos);
        return true;
      } else {
        auto eot = false;
        lua_pushvalue(L, ctx.stream_fn);
        if (pos - ctx.start_pos < ctx.prompt.size()) {
          lua_pushnil(L);
        } else if (inst->eos(token)) {
          eot = true;
          lua_pushnil(L);
        } else {
          ctx.output.front() = token;
          std::string token_text;
          if (!inst->model().Tokenizer().Decode(ctx.output, &token_text)) {
            throw std::runtime_error("Tokenizer decoding failed. (batch stream_mode)");
          }
          lua_pushlstring(L, token_text.data(), token_text.size());
        }
        lua_pushinteger(L, pos - ctx.start_pos);
        lua_pushinteger(L, ctx.prompt.size());
        lua_call(L, 3, 1);
        auto res = lua_toboolean(L, -1);
        lua_pop(L, 1);
        if (!eot && res) {
          ctx.sess->set_pos(pos);
          return true;
        } else {
          return false;
        }
      }
    };
    if (!inst->disabled_tokens().empty()) {
      cfg.accept_token = [&](int token, float) {
        return inst->disabled_tokens().find(token) == inst->disabled_tokens().end();
      };
    }
    if (image) {
      cfg.prefill_tbatch_size = 0;
      for (const auto& ctx: sess_ctxs) {
        cfg.prefill_tbatch_size = std::max(cfg.prefill_tbatch_size, ctx.prefix_end);
      }
      cfg.image_tokens = image;
    }
    auto timing = generate(inst, sess_ctxs, cfg);
    batch_result result(std::move(sess_ctxs), std::move(timing));
    auto ud = lua_newuserdata(L, sizeof(batch_result));
    new(ud) batch_result(std::move(result));
    luaL_getmetatable(L, name);
    lua_setmetatable(L, -2);
    return 1;
  } catch (const std::exception& e) {
    lua_pushnil(L);
    lua_pushstring(L, e.what());
    return 2;
  }
}

session_context::session_context(session* s)
  : sess(s)
  , start_pos(s->pos()) {
  // nop
}

batch_result::batch_result(std::vector<session_context>&& sess_ctxs, gcpp::TimingInfo&& timing)
  : sess_ctxs_(std::move(sess_ctxs))
  , timing_(std::move(timing)) {
  for (const auto& ctx: sess_ctxs_) {
    sess2ctx_[ctx.sess] = &ctx;
  }
}

const session_context* batch_result::get(session* sess) const {
  try {
    return sess2ctx_.at(sess);
  } catch (const std::out_of_range&) {
    return nullptr;
  }
}

void batch_result::declare(lua_State* L) {
  constexpr const luaL_Reg metatable[] = {
    {"__call", call},
    {"__gc", destroy},
    {nullptr, nullptr}
  };
  constexpr const luaL_Reg methods[] = {
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

batch_result* batch_result::check(lua_State* L, int index) {
  return static_cast<batch_result*>(luaL_checkudata(L, index, name));
}

}
