set_target_properties(migrate_weights PROPERTIES
  EXCLUDE_FROM_ALL FALSE
  PREFIX "gemma."
)
install(TARGETS migrate_weights DESTINATION "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}")
