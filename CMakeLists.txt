cmake_minimum_required(VERSION 3.20)
project(hce
  VERSION 0.1.0
  DESCRIPTION "Bit-exact lossless hybrid compression engine"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# The static libraries are also linked into the Python extension module.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(HCE_BUILD_TESTS "Build unit tests and the conformance harness" ON)
option(HCE_BUILD_PYTHON "Build the pybind11 module when pybind11 is available" ON)
option(HCE_WERROR "Treat compiler warnings as errors" OFF)
set(HCE_SANITIZE "" CACHE STRING "Comma-separated sanitizers, e.g. address,undefined")

# Warnings for every target we own; third-party headers are not affected.
add_library(hce_warnings INTERFACE)
if(MSVC)
  target_compile_options(hce_warnings INTERFACE /W4 /permissive- /utf-8 $<$<BOOL:${HCE_WERROR}>:/WX>)
else()
  target_compile_options(hce_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Wold-style-cast -Wcast-align
    $<$<BOOL:${HCE_WERROR}>:-Werror>)
endif()

if(HCE_SANITIZE)
  add_compile_options(-fsanitize=${HCE_SANITIZE} -fno-sanitize-recover=all -fno-omit-frame-pointer)
  add_link_options(-fsanitize=${HCE_SANITIZE})
endif()

add_subdirectory(core)
add_subdirectory(encoder)
add_subdirectory(format)

if(HCE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(conformance)
  add_subdirectory(tests)
  add_test(NAME decode_path_rules
    COMMAND ${CMAKE_COMMAND} -DROOT=${PROJECT_SOURCE_DIR} -P ${PROJECT_SOURCE_DIR}/cmake/CheckDecodePath.cmake)
endif()

if(HCE_BUILD_PYTHON)
  add_subdirectory(bindings)
endif()
