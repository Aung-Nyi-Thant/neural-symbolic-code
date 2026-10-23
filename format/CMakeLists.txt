# Container format, profiles and the range coder. This is decode path too: same rules as core/.
add_library(hce_format STATIC
  container.cpp
  profiles.cpp)
target_link_libraries(hce_format PUBLIC hce_core PRIVATE hce_encoder hce_warnings)
