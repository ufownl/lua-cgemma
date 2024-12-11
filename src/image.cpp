#include "image.hpp"
#include <string>
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

gcpp::Image* to(lua_State* L, int index) {
  return lua_isuserdata(L, index) && luaL_checkudata(L, index, name) ? static_cast<gcpp::Image*>(lua_touserdata(L, index)) : nullptr;
}

gcpp::Image* check(lua_State* L, int index) {
  auto ud = to(L, index);
  if (!ud) {
    luaL_error(L, "Bad argument #%d, %s expected", index, name);
  }
  return ud;
}

int create(lua_State* L) {
  auto nargs = lua_gettop(L);
  try {
    gcpp::Image img;
    if (nargs < 2) {
      size_t len;
      auto buf = luaL_checklstring(L, 1, &len);
      if (!img.ReadPPM(hwy::Span<const char>(buf, len))) {
        if (!img.ReadPPM(std::string(buf, len))) {
          throw std::runtime_error("Failed to read PPM image");
        }
      }
    } else {
      auto width = luaL_checkinteger(L, 1);
      auto height = luaL_checkinteger(L, 2);
      luaL_checktype(L, 3, LUA_TTABLE);
      std::vector<float> values;
      values.reserve(width * height * 3);
      lua_pushnil(L);
      while (lua_next(L, 3)) {
        values.push_back(luaL_checknumber(L, -1));
        lua_pop(L, 1);
      }
      if (values.size() < width * height * 3) {
        throw std::runtime_error("Not enough data");
      }
      img.Set(width, height, values.data());
    }
    auto ud = lua_newuserdata(L, sizeof(gcpp::Image));
    new(ud) gcpp::Image(std::move(img));
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
