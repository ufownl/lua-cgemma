#include "laux.hpp"
#include <cstring>

namespace cgemma { namespace utils {

void* userdata(lua_State* L, int index, const char* name) {
  if (!lua_isuserdata(L, index) || !luaL_getmetafield(L, index, "_NAME")) {
    return nullptr;
  }
  auto r = std::strcmp(luaL_checkstring(L, -1), name);
  lua_pop(L, 1);
  return r == 0 ? lua_touserdata(L, index) : nullptr;
}

} }
