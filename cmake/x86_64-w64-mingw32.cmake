# Cross-compiles for 64-bit Windows with MinGW-w64 and runs the tests under Wine:
#   cmake --preset mingw-wine && cmake --build --preset mingw-wine && ctest --preset mingw-wine
# Needs g++-mingw-w64-x86-64-posix (the posix thread model, for std::thread and
# std::jthread) and Wine. The command-line tool and its test are POSIX-only and
# are not built here; the engine, the aio thread backend and their suites are.
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++-posix)
set(CMAKE_RC_COMPILER x86_64-w64-mingw32-windres)
# Static binaries: Wine then needs no MinGW runtime DLLs.
set(CMAKE_EXE_LINKER_FLAGS_INIT "-static")
set(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
# Debian and Ubuntu install the 64-bit loader as /usr/lib/wine/wine64, off PATH.
find_program(HCE_WINE NAMES wine64 wine PATHS /usr/lib/wine NO_CMAKE_FIND_ROOT_PATH)
if(HCE_WINE)
  set(CMAKE_CROSSCOMPILING_EMULATOR ${HCE_WINE})
endif()
