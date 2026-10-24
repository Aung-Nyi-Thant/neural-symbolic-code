# Cross-compiles for AArch64 Linux and runs the tests under QEMU user mode:
#   cmake --preset aarch64-qemu && cmake --build --preset aarch64-qemu && ctest --preset aarch64-qemu
# Needs g++-aarch64-linux-gnu and qemu-user. "-cpu max" exposes every
# extension QEMU implements (NEON, and SDOT/UDOT for armv8.2-a+dotprod).
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_CXX_FLAGS_INIT "-march=armv8.2-a+dotprod")
# Static binaries: QEMU then needs no AArch64 sysroot to run them.
set(CMAKE_EXE_LINKER_FLAGS_INIT "-static")
set(CMAKE_CROSSCOMPILING_EMULATOR qemu-aarch64 -cpu max)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
