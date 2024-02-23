find_package(LuaJIT REQUIRED)

aux_source_directory(. SOURCES)
add_library(cgemma MODULE ${SOURCES})
target_include_directories(cgemma PRIVATE ${LUA_INCLUDE_DIR})
target_include_directories(cgemma PRIVATE ${gemma_SOURCE_DIR})
target_include_directories(cgemma PRIVATE ${sentencepiece_SOURCE_DIR})
target_link_libraries(cgemma PRIVATE ${LUA_LIBRARIES} libgemma)
set_target_properties(cgemma PROPERTIES PREFIX "" SUFFIX ".so")

get_filename_component(LUA_LIBRARY_DIR ${LUA_LIBRARY} PATH)
set(LUA_MODULE_DIR "${LUA_LIBRARY_DIR}/lua/5.1")
install(TARGETS cgemma DESTINATION ${LUA_MODULE_DIR})
install(TARGETS sentencepiece DESTINATION ${LUA_LIBRARY_DIR})
