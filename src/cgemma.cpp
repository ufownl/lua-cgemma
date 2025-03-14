#include "cgemma.hpp"
#include "scheduler.hpp"
#include "instance.hpp"
#include "session.hpp"
#include "image_tokens.hpp"
#include "batch.hpp"
#include <hwy/timer.h>
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
  std::cout << banner << std::endl;
  auto now = std::time(nullptr);
  std::cout << "Date & Time              : " << std::put_time(std::localtime(&now), "%F %T") << std::endl;
  char cpu[100];
  if (hwy::platform::GetCpuString(cpu)) {
    std::cout << "CPU                      : " << cpu << std::endl;
  }
  std::cout << "Instruction Set          : " << hwy::TargetName(hwy::DispatchedTarget()) << " (" << hwy::VectorBytes() * 8 << " bits)" << std::endl;
  std::cout << "Hardware Concurrency     : " << std::thread::hardware_concurrency() << std::endl;
  std::cout << "Compiled Config          : " << gcpp::CompiledConfig() << std::endl;
  std::cout << std::endl;
  return 0;
}

}

int luaopen_cgemma(lua_State* L) {
  constexpr const luaL_Reg entries[] = {
    {"info", info},
    {"scheduler", cgemma::scheduler::create},
    {"new", cgemma::instance::create},
    {"batch", cgemma::batch},
    {nullptr, nullptr}
  };
  cgemma::scheduler::declare(L);
  cgemma::instance::declare(L);
  cgemma::session::declare(L);
  cgemma::image_tokens::declare(L);
  cgemma::batch_result::declare(L);
  lua_newtable(L);
  luaL_register(L, nullptr, entries);
  lua_pushliteral(L, "cgemma");
  lua_setfield(L, -2, "_NAME");
  lua_pushliteral(L, "1.0");
  lua_setfield(L, -2, "_VERSION");
  return 1;
}
