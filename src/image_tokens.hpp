#ifndef CGEMMA_IMAGE_TOKENS_HPP
#define CGEMMA_IMAGE_TOKENS_HPP

#include <lua.hpp>
#include <gemma/gemma.h>

namespace cgemma { namespace image_tokens {

void declare(lua_State* L);
gcpp::ImageTokens* to(lua_State* L, int index);
gcpp::ImageTokens* check(lua_State* L, int index);
int create(lua_State* L);

} }

#endif  // CGEMMA_IMAGE_TOKENS_HPP
