#ifndef CGEMMA_IMAGE_HPP
#define CGEMMA_IMAGE_HPP

#include <lua.hpp>
#include <paligemma/image.h>

namespace cgemma { namespace image {

void declare(lua_State* L);
gcpp::Image* to(lua_State* L, int index);
gcpp::Image* check(lua_State* L, int index);
int create(lua_State* L);

} }

#endif  // CGEMMA_IMAGE_HPP
