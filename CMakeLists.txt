cmake_minimum_required(VERSION 3.20)
project(hce
  VERSION 2.2.0
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
# The Python package: the extension module hce._hce (bindings/) and the hce
# package around it (python/hce). pip builds it through scikit-build-core
# (pyproject.toml); -DHCE_BUILD_PYTHON=ON builds it here and stages an
# importable package in <build>/python, which the python test runs against.
option(HCE_BUILD_PYTHON "Build the Python extension module (needs pybind11 >= 2.13)" OFF)
if(HCE_BUILD_PYTHON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # the static libraries end up inside a shared module
endif()

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

# Profile-guided optimization, off by default: GENERATE builds instrumented
# binaries that write profiles to HCE_PGO_DIR when run; USE rebuilds from
# those profiles (branch weights and code layout of the codec's bit loops).
# tools/pgo.sh runs the whole cycle on a training corpus. The output of a PGO
# build is byte-identical to any other build: only code placement changes.
set(HCE_PGO "OFF" CACHE STRING "Profile-guided optimization: OFF, GENERATE or USE")
set_property(CACHE HCE_PGO PROPERTY STRINGS OFF GENERATE USE)
set(HCE_PGO_DIR "${CMAKE_BINARY_DIR}/pgo-profile" CACHE PATH "Where GENERATE writes and USE reads profiles")
if(HCE_PGO STREQUAL "GENERATE")
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-fprofile-generate=${HCE_PGO_DIR})
    add_link_options(-fprofile-generate=${HCE_PGO_DIR})
  else()
    # Profiles are named after object paths relative to the build tree, so the
    # USE build finds them whatever its own directory is.
    add_compile_options(-fprofile-generate -fprofile-dir=${HCE_PGO_DIR} -fprofile-update=atomic
                        -fprofile-prefix-path=${CMAKE_BINARY_DIR})
    add_link_options(-fprofile-generate -fprofile-dir=${HCE_PGO_DIR} -fprofile-update=atomic)
  endif()
elseif(HCE_PGO STREQUAL "USE")
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Clang reads one merged file: llvm-profdata merge -o <dir>/hce.profdata <dir>/*.profraw
    add_compile_options(-fprofile-use=${HCE_PGO_DIR}/hce.profdata -Wno-profile-instr-unprofiled
                        -Wno-profile-instr-out-of-date)
  else()
    # Code the training run never reached is optimized as usual, not for size;
    # the test suites are not part of the training run and have no profiles.
    add_compile_options(-fprofile-use -fprofile-dir=${HCE_PGO_DIR} -fprofile-partial-training
                        -fprofile-prefix-path=${CMAKE_BINARY_DIR} -Wno-missing-profile)
  endif()
elseif(NOT HCE_PGO STREQUAL "OFF")
  message(FATAL_ERROR "HCE_PGO must be OFF, GENERATE or USE (got '${HCE_PGO}')")
endif()

# Header-only core: framing (hc_framed.hpp) and stream interfaces.
add_library(hce_core INTERFACE)
target_include_directories(hce_core INTERFACE ${PROJECT_SOURCE_DIR})
target_compile_features(hce_core INTERFACE cxx_std_20)

# The neural CM codec: hot integer loops, optimized at -O3 with loop
# vectorization in every configuration but Debug.
add_library(hce_codec STATIC
  codec/neural_cm_codec.cpp
  codec/ncm_run.cpp
  codec/ncm_pipe.cpp
  codec/ncm_kernels.cpp
  codec/ncm_tables.cpp)
target_link_libraries(hce_codec PUBLIC hce_core Threads::Threads PRIVATE hce_warnings)
target_compile_options(hce_codec PRIVATE $<$<NOT:$<CONFIG:Debug>>:-O3 -ftree-vectorize>)
# The 128 Ki-entry counter table is generated at compile time; Clang's default
# constant-evaluation budget (2^20 steps) is a little short for it.
set_source_files_properties(codec/ncm_tables.cpp PROPERTIES
  COMPILE_OPTIONS "$<$<CXX_COMPILER_ID:Clang,AppleClang>:-fconstexpr-steps=16777216>")

# Structural compression (structural/): the Phase 1 tokenizer, the frozen
# global model and the token-level codec. The pipelines instantiate for it.
add_library(hce_structural_codec STATIC
  structural/structural_parser.cpp
  structural/global_model.cpp
  structural/structural_cm_codec.cpp)
target_link_libraries(hce_structural_codec PUBLIC hce_core hce_codec PRIVATE hce_warnings)
target_compile_options(hce_structural_codec PRIVATE $<$<NOT:$<CONFIG:Debug>>:-O3>)

# Parallel encoder and decoder pipelines.
add_library(hce_pipeline STATIC
  encoder/parallel.cpp
  decoder/pipeline.cpp)
target_link_libraries(hce_pipeline
  PUBLIC hce_core hce_codec hce_structural_codec Threads::Threads
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
  aio/staged_file.cpp
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

# Two-pass structural archives: token spool and both passes over aio.
add_library(hce_structural STATIC structural/archive.cpp)
target_link_libraries(hce_structural PUBLIC hce_pipeline hce_aio PRIVATE hce_warnings)

# ultracodec (ultracodec/): the 4-tier engine (LZ77, LZ77 + rANS, lean CM,
# full CM). Shares no code with the HCE codec above.
option(HCE_BUILD_ULTRACODEC "Build the ultracodec 4-tier engine" ON)
if(HCE_BUILD_ULTRACODEC)
  add_subdirectory(ultracodec)
endif()

# Command-line tool (POSIX: file descriptors, signals, atomic renames). Windows
# builds (mingw-wine preset) cover the libraries and their suites.
if(NOT WIN32)
  add_executable(hce cli/hce.cpp)
  target_link_libraries(hce PRIVATE hce_pipeline hce_structural hce_warnings)

  if(SKBUILD)  # pip install: the tool lands in the environment's bin/ next to python
    install(TARGETS hce RUNTIME DESTINATION "${SKBUILD_SCRIPTS_DIR}")
  else()
    include(GNUInstallDirs)
    install(TARGETS hce RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endif()

if(HCE_BUILD_PYTHON)
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 2.13 CONFIG QUIET)
  if(NOT pybind11_FOUND)  # installed by pip: ask it where its CMake files are
    execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
                    OUTPUT_VARIABLE HCE_PYBIND11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE HCE_PYBIND11_RC ERROR_QUIET)
    if(NOT HCE_PYBIND11_RC EQUAL 0)
      message(FATAL_ERROR "HCE_BUILD_PYTHON=ON needs pybind11 >= 2.13: pip install pybind11==2.13.6")
    endif()
    find_package(pybind11 2.13 CONFIG REQUIRED PATHS "${HCE_PYBIND11_DIR}" NO_DEFAULT_PATH)
  endif()

  pybind11_add_module(hce_python MODULE NO_EXTRAS bindings/hce_python.cpp)  # no LTO: the engine libraries are not LTO objects
  set_target_properties(hce_python PROPERTIES
    OUTPUT_NAME _hce
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/hce")
  target_link_libraries(hce_python PRIVATE hce_structural hce_warnings)
  target_compile_definitions(hce_python PRIVATE HCE_VERSION="${PROJECT_VERSION}")
  # The pure-Python files beside the module: PYTHONPATH=<build>/python imports the package.
  file(GLOB HCE_PYTHON_FILES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/python/hce/*")
  add_custom_command(TARGET hce_python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${HCE_PYTHON_FILES} "$<TARGET_FILE_DIR:hce_python>"
    VERBATIM)
  if(SKBUILD)
    install(TARGETS hce_python LIBRARY DESTINATION hce)
  endif()
endif()

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

  set(HCE_TESTS framing codec pipeline aio structural)
  if(TARGET hce)
    add_executable(test_cli tests/test_cli.cpp)
    target_link_libraries(test_cli PRIVATE hce_core hce_warnings)
    add_test(NAME cli COMMAND test_cli $<TARGET_FILE:hce>)
    list(APPEND HCE_TESTS cli)
  endif()

  add_executable(test_aio tests/test_aio.cpp)
  target_link_libraries(test_aio PRIVATE hce_aio hce_pipeline hce_warnings)
  add_test(NAME aio COMMAND test_aio)

  add_executable(test_structural tests/test_structural.cpp)
  target_link_libraries(test_structural PRIVATE hce_structural hce_warnings)
  # The suite reads this repository's sources as a corpus. Under Wine, drive Z: is the host's root.
  if(WIN32 AND CMAKE_CROSSCOMPILING)
    set(HCE_CORPUS_DIR "Z:${PROJECT_SOURCE_DIR}")
  else()
    set(HCE_CORPUS_DIR "${PROJECT_SOURCE_DIR}")
  endif()
  target_compile_definitions(test_structural PRIVATE HCE_SOURCE_DIR="${HCE_CORPUS_DIR}")
  add_test(NAME structural COMMAND test_structural)

  # Sanitizer failures get exit codes of their own, so the CLI tests can never
  # mistake one for an hce exit status.
  set(HCE_TEST_ENV
    "ASAN_OPTIONS=exitcode=86:detect_leaks=1"
    "UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1:exitcode=87"
    "TSAN_OPTIONS=halt_on_error=1:exitcode=88:second_deadlock_stack=1")
  if(HCE_SANITIZE MATCHES "thread")
    list(APPEND HCE_TEST_ENV "HCE_FUZZ_ITERS=24")  # every byte costs ~10x under TSan
  endif()
  if(WIN32 AND CMAKE_CROSSCOMPILING)
    list(APPEND HCE_TEST_ENV "WINEDEBUG=-all")  # Wine's own diagnostics would drown the test output
  endif()
  if(CMAKE_CROSSCOMPILING_EMULATOR)
    # The CLI test runs hce itself; without binfmt_misc it needs the emulator spelled out.
    list(JOIN CMAKE_CROSSCOMPILING_EMULATOR " " HCE_WRAPPER)
    list(APPEND HCE_TEST_ENV "HCE_TEST_WRAPPER=${HCE_WRAPPER}")
  endif()
  set_tests_properties(${HCE_TESTS} PROPERTIES TIMEOUT 3600 ENVIRONMENT "${HCE_TEST_ENV}")

  # The pytest suite (tests/python), against the staged package and the CLI.
  if(HCE_BUILD_PYTHON)
    execute_process(COMMAND "${Python_EXECUTABLE}" -c "import pytest, hypothesis"
                    RESULT_VARIABLE HCE_PYTEST_RC OUTPUT_QUIET ERROR_QUIET)
    if(HCE_PYTEST_RC EQUAL 0)
      add_test(NAME python
               COMMAND "${Python_EXECUTABLE}" -m pytest "${PROJECT_SOURCE_DIR}/tests/python" -p no:cacheprovider
               WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
      set(HCE_PYTHON_ENV "PYTHONPATH=${CMAKE_BINARY_DIR}/python")
      if(TARGET hce)
        list(APPEND HCE_PYTHON_ENV "HCE_CLI=$<TARGET_FILE:hce>")
      endif()
      set_tests_properties(python PROPERTIES TIMEOUT 1800 ENVIRONMENT "${HCE_TEST_ENV};${HCE_PYTHON_ENV}")
    else()
      message(STATUS "pytest or hypothesis not importable: the python test is not registered "
                     "(pip install -c constraints.txt -e .[test], or see README)")
    endif()
  endif()
endif()
