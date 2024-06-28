set_target_properties(compress_weights PROPERTIES
  EXCLUDE_FROM_ALL FALSE
  PREFIX "gemma."
)
install(TARGETS compress_weights DESTINATION "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}")
