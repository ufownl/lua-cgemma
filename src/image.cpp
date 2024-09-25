#include "image.hpp"
#include <stdexcept>

namespace {

constexpr const char name[] = "cgemma.image";

int destroy(lua_State* L) {
  cgemma::image::check(L, 1)->~Image();
  return 0;
}

}

namespace cgemma { namespace image {

void declare(lua_State* L) {
  constexpr const luaL_Reg metatable[] = {
    {"__gc", destroy},
    {nullptr, nullptr}
  };
  constexpr const luaL_Reg methods[] = {
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

gcpp::Image* check(lua_State* L, int index) {
  if (!lua_isuserdata(L, index) || !luaL_checkudata(L, index, name)) {
    luaL_error(L, "Bad argument #%d, %s expected", index, name);
  }
  return static_cast<gcpp::Image*>(lua_touserdata(L, index));
}

int create(lua_State* L) {
  size_t len;
  auto buf = luaL_checklstring(L, 1, &len);
  try {
    auto ud = lua_newuserdata(L, sizeof(gcpp::Image));
    auto img = new(ud) gcpp::Image;
    if (!img->ReadPPM(std::string(buf, len))) {
      throw std::runtime_error("Failed to read PPM image");
    }
    img->Resize();
    luaL_getmetatable(L, name);
    lua_setmetatable(L, -2);
    return 1;
  } catch (const std::exception& e) {
    lua_pushnil(L);
    lua_pushstring(L, e.what());
    return 2;
  }
}

} }
