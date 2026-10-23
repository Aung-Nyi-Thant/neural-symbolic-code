# Decode-path kernels: integer-only, single-threaded, no allocation in per-bit code.
add_library(hce_core STATIC
  kernels/anchor_history.cpp
  kernels/cm_model.cpp
  kernels/crc32c.cpp
  kernels/tables.cpp)
target_include_directories(hce_core PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_features(hce_core PUBLIC cxx_std_20)
target_link_libraries(hce_core PRIVATE hce_warnings)
