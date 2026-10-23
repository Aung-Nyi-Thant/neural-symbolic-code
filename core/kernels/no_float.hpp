// Include this LAST in every decode-path .cpp file. From that point on, any use of `float` or
// `double` is a compile error (GCC/Clang). cmake/CheckDecodePath.cmake enforces the same rule for
// headers and for other compilers.
#pragma once

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC poison float double
#endif
