#ifndef CGEMMA_SCHEDULER_HPP
#define CGEMMA_SCHEDULER_HPP

#include <lua.hpp>
#include <util/app.h>
#include <util/threading.h>
#include <memory>

namespace cgemma {

class scheduler {
public:
  scheduler();
  scheduler(int argc, char* argv[]);

  gcpp::NestedPools& pools() { return *pools_; }

  static void declare(lua_State* L);
  static scheduler* to(lua_State* L, int index);
  static scheduler* check(lua_State* L, int index);
  static int create(lua_State* L);

private:
  gcpp::AppArgs args_;
  std::unique_ptr<gcpp::NestedPools> pools_;
};

}

#endif  // CGEMMA_SCHEDULER_HPP
