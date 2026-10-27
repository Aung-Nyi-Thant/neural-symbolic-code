# ultracodec: the 4-tier engine (Tier 1 LZ77, Tier 2 LZ77 + rANS, Tier 3 lean
# CM + binary rANS, Tier 4 large CM). Self-contained: it shares no code with
# the HCE codec and can be built on its own (cmake -S ultracodec).
cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(ultracodec VERSION 1.0.0 LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  endif()
  find_package(Threads REQUIRED)
  option(ULTRA_BUILD_TESTS "Build the ultracodec tests" ON)
  enable_testing()
else()
  set(ULTRA_BUILD_TESTS ${HCE_BUILD_TESTS})
endif()

add_library(ultracodec STATIC
  src/common.cpp
  src/block_engine.cpp
  src/dedup.cpp
  src/engine.cpp
  src/rans_static.cpp
  src/tier1_hyperspeed.cpp
  src/tier2_consumer.cpp
  src/tier3_balanced.cpp
  src/tier4_enterprise.cpp)
target_include_directories(ultracodec PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(ultracodec PUBLIC cxx_std_20)
target_link_libraries(ultracodec PUBLIC Threads::Threads)
target_compile_options(ultracodec PRIVATE $<$<NOT:$<CONFIG:Debug>>:-O3>)
if(TARGET hce_warnings)
  target_link_libraries(ultracodec PRIVATE hce_warnings)
  set(ULTRA_WARNINGS hce_warnings)
else()
  target_compile_options(ultracodec PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion)
endif()

# File tool: ultra c|d|i ...
add_executable(ultra tools/ultra_cli.cpp)
target_link_libraries(ultra PRIVATE ultracodec ${ULTRA_WARNINGS})

# Round-trip benchmark: ultra_bench <tier> [--threads N] FILE...
add_executable(ultra_bench tools/ultra_bench.cpp)
target_link_libraries(ultra_bench PRIVATE ultracodec ${ULTRA_WARNINGS})
target_include_directories(ultra_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(ultra_bench PRIVATE $<$<NOT:$<CONFIG:Debug>>:-O2>)

if(ULTRA_BUILD_TESTS)
  enable_testing()
  add_executable(test_ultracodec tests/test_ultracodec.cpp)
  target_link_libraries(test_ultracodec PRIVATE ultracodec ${ULTRA_WARNINGS})
  target_include_directories(test_ultracodec PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME ultracodec COMMAND test_ultracodec)
  set(ULTRA_TEST_ENV
    "ASAN_OPTIONS=exitcode=86:detect_leaks=1"
    "UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1:exitcode=87"
    "TSAN_OPTIONS=halt_on_error=1:exitcode=88:second_deadlock_stack=1")
  if(HCE_SANITIZE MATCHES "thread" OR CMAKE_CROSSCOMPILING)
    list(APPEND ULTRA_TEST_ENV "ULTRA_FUZZ_ITERS=24")  # TSan, QEMU and Wine run the CM tiers 10-30x slower
  endif()
  if(WIN32 AND CMAKE_CROSSCOMPILING)
    list(APPEND ULTRA_TEST_ENV "WINEDEBUG=-all")
  endif()
  set_tests_properties(ultracodec PROPERTIES TIMEOUT 3600 ENVIRONMENT "${ULTRA_TEST_ENV}")
endif()
