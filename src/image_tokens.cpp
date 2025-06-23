#include "image_tokens.hpp"
#include "instance.hpp"
#include "utils/laux.hpp"

namespace {

constexpr const char name[] = "cgemma.image_tokens";

int destroy(lua_State* L) {
  cgemma::image_tokens::check(L, 1)->~MatStorageT();
  return 0;
}

}

namespace cgemma { namespace image_tokens {

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

gcpp::ImageTokens* to(lua_State* L, int index) {
  return static_cast<gcpp::ImageTokens*>(utils::userdata(L, index, name));
}

gcpp::ImageTokens* check(lua_State* L, int index) {
  return static_cast<gcpp::ImageTokens*>(luaL_checkudata(L, index, name));
}

int create(lua_State* L) {
  auto nargs = lua_gettop(L);
  auto inst = cgemma::instance::check(L, 1);
  try {
    gcpp::Image img;
    if (nargs < 3) {
      size_t len;
      auto buf = luaL_checklstring(L, 2, &len);
      if (!img.ReadPPM(hwy::Span<const char>(buf, len))) {
        if (!img.ReadPPM(std::string(buf, len))) {
          throw std::runtime_error("Failed to read PPM image");
        }
      }
    } else {
      auto width = luaL_checkinteger(L, 2);
      auto height = luaL_checkinteger(L, 3);
      luaL_checktype(L, 4, LUA_TTABLE);
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
    auto model_cfg = inst->model().GetModelConfig();
    img.Resize(model_cfg.vit_config.image_size, model_cfg.vit_config.image_size);
    gcpp::ImageTokens tks(
      "image_tokens",
      gcpp::Extents2D(model_cfg.vit_config.seq_len / (model_cfg.vit_config.pool_dim * model_cfg.vit_config.pool_dim), model_cfg.model_dim),
      gcpp::MatPadding::kOdd
    );
    gcpp::RuntimeConfig cfg;
    cfg.gen = &inst->rnd();
    cfg.verbosity = 0;
    inst->model().GenerateImageTokens(cfg, tks.Rows(), img, tks, inst->env());
    auto ud = lua_newuserdata(L, sizeof(gcpp::ImageTokens));
    new(ud) gcpp::ImageTokens(std::move(tks));
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
