#ifndef CGEMMA_UTILS_LAUX_HPP
#define CGEMMA_UTILS_LAUX_HPP

#include <lua.hpp>

namespace cgemma { namespace utils {

void* userdata(lua_State* L, int index, const char* name);

} }

#endif  // CGEMMA_UTILS_LAUX_HPP
