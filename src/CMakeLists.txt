find_package(LuaJIT REQUIRED)
get_filename_component(LUA_LIBRARY_DIR ${LUA_LIBRARY} PATH)

aux_source_directory(. SOURCES)
aux_source_directory(utils UTILS_SOURCES)
add_library(cgemma MODULE ${SOURCES} ${UTILS_SOURCES})
target_include_directories(cgemma PRIVATE ${LUA_INCLUDE_DIR})
target_include_directories(cgemma PRIVATE ${gemma_SOURCE_DIR})
target_include_directories(cgemma PRIVATE ${sentencepiece_SOURCE_DIR})
target_link_libraries(cgemma PRIVATE ${LUA_LIBRARIES} libgemma)
set_target_properties(cgemma PROPERTIES
  PREFIX ""
  SUFFIX ".so"
  INSTALL_RPATH ${LUA_LIBRARY_DIR}
)

set(LUA_MODULE_DIR "${LUA_LIBRARY_DIR}/lua/5.1")
install(TARGETS cgemma DESTINATION ${LUA_MODULE_DIR})
