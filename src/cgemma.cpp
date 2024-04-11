#include "cgemma.hpp"
#include "instance.hpp"
#include "session.hpp"
#include "scheduler.hpp"
#include <compression/compress.h>
#include <hwy/per_target.h>
#include <hwy/targets.h>
#include <iostream>
#include <iomanip>
#include <ctime>

namespace {

constexpr const char* banner = R""(
    __
   / /_  ______ _      _________ ____  ____ ___  ____ ___  ____ _
  / / / / / __ `/_____/ ___/ __ `/ _ \/ __ `__ \/ __ `__ \/ __ `/
 / / /_/ / /_/ /_____/ /__/ /_/ /  __/ / / / / / / / / / / /_/ /
/_/\__,_/\__,_/      \___/\__, /\___/_/ /_/ /_/_/ /_/ /_/\__,_/
                         /____/
)"";

int info(lua_State* L) {
  auto now = std::time(nullptr);
  std::cout
    << banner << std::endl
    << "Date & Time                                : " << std::put_time(std::localtime(&now), "%F %T") << std::endl
    << "Max Sequence Length                        : " << gcpp::kSeqLen << std::endl
    << "Top-K                                      : " << gcpp::kTopK << std::endl
    << "Weight Type                                : " << gcpp::TypeName(gcpp::GemmaWeightT()) << std::endl
    << "Embedder Input Type                        : " << gcpp::TypeName(gcpp::EmbedderInputT()) << std::endl
    << "Prefill Token Batch Size                   : " << gcpp::kPrefillBatchSize << std::endl
    << "Hardware Concurrency                       : " << std::thread::hardware_concurrency() << std::endl
    << "Instruction Set                            : " << hwy::TargetName(hwy::DispatchedTarget()) << " (" << hwy::VectorBytes() * 8 << " bits)" << std::endl
    << "Compiled Config                            : " << gcpp::CompiledConfig() << std::endl
    << std::endl;
  return 0;
}

template <gcpp::Model MODEL_T>
int compress_weights(lua_State* L) {
  auto w = luaL_checkstring(L, 1);
  auto cw = luaL_checkstring(L, 2);
  auto s = cgemma::scheduler::to(L, 3);
  try {
    std::unique_ptr<cgemma::scheduler> default_sched;
    if (!s) {
      default_sched = std::make_unique<cgemma::scheduler>();
      s = default_sched.get();
    }
    CompressWeights(MODEL_T, gcpp::Path(w), gcpp::Path(cw), s->pool());
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
    {"info", info},
    {"new", cgemma::instance::create},
    {"scheduler", cgemma::scheduler::create},
    {"compress_2b_weights", compress_weights<gcpp::Model::GEMMA_2B>},
    {"compress_7b_weights", compress_weights<gcpp::Model::GEMMA_7B>},
    {"compress_gr2b_weights", compress_weights<gcpp::Model::GRIFFIN_2B>},
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
