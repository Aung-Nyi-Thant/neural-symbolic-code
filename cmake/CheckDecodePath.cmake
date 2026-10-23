# Run as: cmake -DROOT=<source dir> -P CheckDecodePath.cmake
#
# Fails if any decode-path source (core/, format/) uses floating point, threads, clocks or <random>.
# `//` comments are stripped first, so prose may mention these words. Use only `//` comments in
# these directories.
file(GLOB_RECURSE sources
  "${ROOT}/core/*.hpp" "${ROOT}/core/*.cpp" "${ROOT}/format/*.hpp" "${ROOT}/format/*.cpp"
  "${ROOT}/encoder/*.hpp" "${ROOT}/encoder/*.cpp")

set(violations "")
foreach(path IN LISTS sources)
  file(READ "${path}" content)
  string(REGEX REPLACE "//[^\n]*" "" code " ${content} ")
  # The poison pragma in no_float.hpp names the forbidden types on purpose.
  string(REGEX REPLACE "#[ \t]*pragma[ \t]+GCC[ \t]+poison[^\n]*" "" code "${code}")
  if(code MATCHES "[^A-Za-z0-9_](float|double)[^A-Za-z0-9_]")
    string(APPEND violations "  ${path}: floating-point type `${CMAKE_MATCH_1}`\n")
  endif()
  if(code MATCHES "#[ \t]*include[ \t]*<(cmath|math\\.h|cfloat|float\\.h|thread|random|chrono|ctime|time\\.h)>")
    string(APPEND violations "  ${path}: forbidden header <${CMAKE_MATCH_1}>\n")
  endif()
  if(code MATCHES "std::(thread|jthread|async|random_device|mt19937)")
    string(APPEND violations "  ${path}: forbidden std::${CMAKE_MATCH_1}\n")
  endif()
endforeach()

list(LENGTH sources count)
if(violations)
  message(FATAL_ERROR "Decode-path rule violations:\n${violations}")
endif()
message(STATUS "Decode path clean: ${count} files checked")
