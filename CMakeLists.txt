cmake_minimum_required(VERSION 3.20)
project(hce
  VERSION 2.1.0
  DESCRIPTION "Hybrid compression engine: neural CM codec in Container v3 FRAMED, encoder, decoder and CLI"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

option(HCE_WERROR "Treat compiler warnings as errors" ON)
option(HCE_BUILD_TESTS "Build the test suites" ON)
# The codec picks its SIMD kernels at run time (AVX2/SSE2 on x86-64, NEON on
# AArch64) and every path is bit-exact, so a portable build loses nothing but
# the compiler's own auto-vectorization of the scalar parts.
option(HCE_NATIVE "Tune for the build machine (-march=native); the binary may not run elsewhere" OFF)
set(HCE_SANITIZE "" CACHE STRING "Sanitizers to build with: address,undefined | thread | empty")

find_package(Threads REQUIRED)

# Warning policy shared by every first-party target.
add_library(hce_warnings INTERFACE)
target_compile_options(hce_warnings INTERFACE
  -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
  $<$<BOOL:${HCE_WERROR}>:-Werror>)

if(HCE_SANITIZE)
  add_compile_options(-fsanitize=${HCE_SANITIZE} -fno-omit-frame-pointer)
  add_link_options(-fsanitize=${HCE_SANITIZE})
  if(NOT HCE_SANITIZE MATCHES "thread")
    add_compile_options(-fno-sanitize-recover=all)
  endif()
endif()

if(HCE_NATIVE)
  add_compile_options(-march=native)
endif()

# Header-only core: framing (hc_framed.hpp) and stream interfaces.
add_library(hce_core INTERFACE)
target_include_directories(hce_core INTERFACE ${PROJECT_SOURCE_DIR})
target_compile_features(hce_core INTERFACE cxx_std_20)

# The neural CM codec: hot integer loops, optimized at -O3 with loop
# vectorization in every configuration but Debug.
add_library(hce_codec STATIC
  codec/neural_cm_codec.cpp
  codec/ncm_kernels.cpp)
target_link_libraries(hce_codec PUBLIC hce_core PRIVATE hce_warnings)
target_compile_options(hce_codec PRIVATE $<$<NOT:$<CONFIG:Debug>>:-O3 -ftree-vectorize>)

# Parallel encoder and decoder pipelines.
add_library(hce_pipeline STATIC
  encoder/parallel.cpp
  decoder/pipeline.cpp)
target_link_libraries(hce_pipeline
  PUBLIC hce_core hce_codec Threads::Threads
  PRIVATE hce_warnings)

# Asynchronous file I/O (aio/): io_uring on Linux when liburing is available,
# worker threads on macOS, Windows and every other system. The engine does
# not depend on it: its adapters implement the pipelines' InputStream /
# OutputStream. At run time, `automatic` falls back to threads wherever
# io_uring is compiled in but not permitted (seccomp, io_uring_disabled).
set(HCE_AIO_URING "AUTO" CACHE STRING "io_uring backend for aio/: AUTO, ON (require liburing) or OFF")
set_property(CACHE HCE_AIO_URING PROPERTY STRINGS AUTO ON OFF)

add_library(hce_aio STATIC
  aio/async_io.cpp
  aio/async_streams.cpp
  aio/native_file.cpp
  aio/thread_backend.cpp)
target_link_libraries(hce_aio PUBLIC hce_core Threads::Threads PRIVATE hce_warnings)

set(HCE_AIO_BACKENDS "threads")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT HCE_AIO_URING STREQUAL "OFF")
  find_package(PkgConfig QUIET)
  # A host liburing is useless to a cross build: there, look only when told to.
  if(PKG_CONFIG_FOUND AND (NOT CMAKE_CROSSCOMPILING OR HCE_AIO_URING STREQUAL "ON"))
    pkg_check_modules(LIBURING QUIET IMPORTED_TARGET liburing>=2.2)
  endif()
  if(LIBURING_FOUND)
    target_sources(hce_aio PRIVATE aio/uring_backend.cpp)
    target_compile_definitions(hce_aio PRIVATE HCE_AIO_HAVE_URING=1)
    target_link_libraries(hce_aio PRIVATE PkgConfig::LIBURING)
    set(HCE_AIO_BACKENDS "io_uring (liburing ${LIBURING_VERSION}), threads")
  elseif(HCE_AIO_URING STREQUAL "ON")
    message(FATAL_ERROR "HCE_AIO_URING=ON, but liburing >= 2.2 was not found (install liburing-dev)")
  endif()
endif()
message(STATUS "hce_aio backends: ${HCE_AIO_BACKENDS}")

# Command-line tool (POSIX).
add_executable(hce cli/hce.cpp)
target_link_libraries(hce PRIVATE hce_pipeline hce_warnings)

include(GNUInstallDirs)
install(TARGETS hce RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(HCE_BUILD_TESTS)
  enable_testing()

  add_executable(test_hc_framed test_hc_framed.cpp)
  target_link_libraries(test_hc_framed PRIVATE hce_core hce_warnings)
  add_test(NAME framing COMMAND test_hc_framed)

  add_executable(test_codec tests/test_codec.cpp)
  target_link_libraries(test_codec PRIVATE hce_pipeline hce_warnings)
  add_test(NAME codec COMMAND test_codec)

  add_executable(test_pipeline tests/test_pipeline.cpp)
  target_link_libraries(test_pipeline PRIVATE hce_pipeline hce_warnings)
  add_test(NAME pipeline COMMAND test_pipeline)

  add_executable(test_cli tests/test_cli.cpp)
  target_link_libraries(test_cli PRIVATE hce_core hce_warnings)
  add_test(NAME cli COMMAND test_cli $<TARGET_FILE:hce>)

  add_executable(test_aio tests/test_aio.cpp)
  target_link_libraries(test_aio PRIVATE hce_aio hce_pipeline hce_warnings)
  add_test(NAME aio COMMAND test_aio)

  # Sanitizer failures get exit codes of their own, so the CLI tests can never
  # mistake one for an hce exit status.
  set(HCE_TEST_ENV
    "ASAN_OPTIONS=exitcode=86:detect_leaks=1"
    "UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1:exitcode=87"
    "TSAN_OPTIONS=halt_on_error=1:exitcode=88:second_deadlock_stack=1")
  if(HCE_SANITIZE MATCHES "thread")
    list(APPEND HCE_TEST_ENV "HCE_FUZZ_ITERS=24")  # every byte costs ~10x under TSan
  endif()
  if(CMAKE_CROSSCOMPILING_EMULATOR)
    # The CLI test runs hce itself; without binfmt_misc it needs the emulator spelled out.
    list(JOIN CMAKE_CROSSCOMPILING_EMULATOR " " HCE_WRAPPER)
    list(APPEND HCE_TEST_ENV "HCE_TEST_WRAPPER=${HCE_WRAPPER}")
  endif()
  set_tests_properties(framing codec pipeline cli aio PROPERTIES TIMEOUT 3600 ENVIRONMENT "${HCE_TEST_ENV}")
endif()
