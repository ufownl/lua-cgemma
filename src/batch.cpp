#include "batch.hpp"
#include "instance.hpp"
#include "session.hpp"
#include <stdexcept>

namespace {

class batch_data_holder {
public:
  explicit batch_data_holder(const std::vector<cgemma::session_context>& sess_ctxs)
    : sess_ctxs_(sess_ctxs) {
    prompts_.reserve(sess_ctxs.size());
    start_pos_.reserve(sess_ctxs.size());
    kv_caches_.reserve(sess_ctxs.size());
    for (auto& ctx: sess_ctxs) {
      prompts_.emplace_back(ctx.prompt.data(), ctx.prompt.size());
      start_pos_.emplace_back(ctx.start_pos);
      kv_caches_.emplace_back(std::move(ctx.sess->kv_cache()));
    }
  }

  ~batch_data_holder() {
    for (size_t i = 0; i < kv_caches_.size(); ++i) {
      sess_ctxs_[i].sess->kv_cache() = std::move(kv_caches_[i]);
    }
  }

  gcpp::MultiplePromptsTokens prompts() {
    return gcpp::MultiplePromptsTokens(prompts_.data(), prompts_.size());
  }

  gcpp::MultiplePositions start_pos() {
    return gcpp::MultiplePositions(start_pos_.data(), start_pos_.size());
  }

  gcpp::KVCaches kv_caches() {
    return gcpp::KVCaches(kv_caches_.data(), kv_caches_.size());
  }

private:
  const std::vector<cgemma::session_context>& sess_ctxs_;
  std::vector<gcpp::PromptTokens> prompts_;
  std::vector<size_t> start_pos_;
  std::vector<gcpp::KVCache> kv_caches_;
};

int init_arg_state(lua_State* L, int narg, std::vector<cgemma::session_context>& sess_ctxs) {
  auto sess = cgemma::session::check(L, narg);
  if (sess->pos() >= sess->args().max_tokens) {
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

std::vector<cgemma::session_context> parse_args(lua_State* L) {
  constexpr decltype(init_arg_state)* const arg_states[] = {
    init_arg_state,
    [](lua_State* L, int narg, std::vector<cgemma::session_context>& sess_ctxs) {
      size_t len;
      auto text = luaL_checklstring(L, narg, &len);
      auto& ctx = sess_ctxs.back();
      ctx.prompt = ctx.sess->tokenize(text, len);
      return 2;
    },
    [](lua_State* L, int narg, std::vector<cgemma::session_context>& sess_ctxs) {
      if (lua_isfunction(L, 3)) {
        sess_ctxs.back().stream_fn = narg;
        return 0;
      } else {
        auto& ctx = sess_ctxs.back();
        ctx.output.reserve(ctx.sess->args().max_generated_tokens);
        return init_arg_state(L, narg, sess_ctxs);
      }
    }
  };
  auto nargs = lua_gettop(L);
  if (nargs < 2) {
    luaL_error(L, "Too few arguments, at least 2 expected");
  }
  std::vector<cgemma::session_context> sess_ctxs;
  sess_ctxs.reserve(nargs / 2);
  int arg_state = 0;
  for (auto i = 1; i <= nargs; ++i) {
    arg_state = arg_states[arg_state](L, i, sess_ctxs);
  }
  if (sess_ctxs.back().prompt.empty()) {
    luaL_error(L, "Too few arguments, %d expected", nargs + 1);
  }
  size_t max_prompt_size = 0;
  for (const auto& ctx: sess_ctxs) {
    max_prompt_size = std::max(max_prompt_size, ctx.prompt.size());
  }
  for (auto& ctx: sess_ctxs) {
    auto padding_size = max_prompt_size - ctx.prompt.size();
    if (padding_size > 0) {
      std::vector<int> prompt;
      prompt.reserve(max_prompt_size);
      prompt.resize(padding_size, cgemma::PAD_ID);
      prompt.insert(prompt.end(), ctx.prompt.begin(), ctx.prompt.end());
      ctx.prompt = std::move(prompt);
    }
  }
  return sess_ctxs;
}

gcpp::RuntimeConfig parse_config(const std::vector<cgemma::session_context>& sess_ctxs) {
  gcpp::RuntimeConfig cfg;
  cfg.max_tokens = gcpp::kSeqLen;
  cfg.max_generated_tokens = cfg.max_tokens - 1;
  cfg.prefill_tbatch_size = 4096;
  cfg.decode_qbatch_size = 4096;
  cfg.temperature = 0.0f;
  for (const auto& ctx: sess_ctxs) {
    cfg.max_tokens = std::min(cfg.max_tokens, ctx.sess->args().max_tokens);
    cfg.max_generated_tokens = std::min(cfg.max_generated_tokens, ctx.sess->args().max_generated_tokens);
    cfg.prefill_tbatch_size = std::min(cfg.prefill_tbatch_size, ctx.sess->args().prefill_tbatch_size);
    cfg.decode_qbatch_size = std::min(cfg.decode_qbatch_size, ctx.sess->args().decode_qbatch_size);
    cfg.temperature += ctx.sess->args().temperature;
  }
  cfg.temperature /= sess_ctxs.size();
  return cfg;
}

gcpp::TimingInfo generate(cgemma::instance* inst, const std::vector<cgemma::session_context>& sess_ctxs, const gcpp::RuntimeConfig& cfg) {
  gcpp::TimingInfo timing;
  batch_data_holder bdh(sess_ctxs);
  inst->model().GenerateBatch(cfg, bdh.prompts(), bdh.start_pos(), bdh.kv_caches(), timing);
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
    auto sess_ctxs = parse_args(L);
    auto cfg = parse_config(sess_ctxs);
    cfg.verbosity = 0;
    auto inst = sess_ctxs.front().sess->inst();
    cfg.gen = &inst->rnd();
    cfg.batch_stream_token = [&](size_t query_idx, size_t pos, int token, float) {
      auto& ctx = sess_ctxs[query_idx];
      if (ctx.stream_fn == 0) {
        if (pos - ctx.start_pos >= ctx.prompt.size() && token != gcpp::EOS_ID) {
          if (inst->model().Info().training == gcpp::ModelTraining::GEMMA_IT && token == EOT_ID) {
            return false;
          }
          ctx.output.push_back(token);
        }
        ctx.sess->set_pos(pos);
        return true;
      } else {
        throw std::runtime_error("Not implemented.");
      }
    };
    if (!inst->disabled_tokens().empty()) {
      cfg.accept_token = [&](int token, float) {
        return inst->disabled_tokens().find(token) == inst->disabled_tokens().end();
      };
    }
    auto timing = generate(inst, sess_ctxs, cfg);
    auto ud = lua_newuserdata(L, sizeof(batch_result));
    new(ud) batch_result(std::move(sess_ctxs), timing);
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

batch_result::batch_result(std::vector<session_context>&& sess_ctxs, const gcpp::TimingInfo& timing)
  : sess_ctxs_(std::move(sess_ctxs))
  , timing_(timing) {
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
  if (!lua_isuserdata(L, index) || !luaL_checkudata(L, index, name)) {
    luaL_error(L, "Bad argument #%d, %s expected", index, name);
  }
  return static_cast<batch_result*>(lua_touserdata(L, index));
}

}
