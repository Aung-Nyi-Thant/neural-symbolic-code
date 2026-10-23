// Minimal test harness: CHECK records a failure and continues; finish() sets the exit status.
#pragma once

#include <cstdio>
#include <exception>

namespace hce_test {

inline int failures = 0;

inline int finish(const char* suite) {
    if (failures == 0) {
        std::printf("%s: all checks passed\n", suite);
        return 0;
    }
    std::printf("%s: %d check(s) FAILED\n", suite, failures);
    return 1;
}

}  // namespace hce_test

#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++hce_test::failures;                                                     \
        }                                                                             \
    } while (0)

#define CHECK_THROWS(expr)                                                                   \
    do {                                                                                     \
        bool thrown_ = false;                                                                \
        try {                                                                                \
            (void)(expr);                                                                    \
        } catch (const std::exception&) {                                                    \
            thrown_ = true;                                                                  \
        }                                                                                    \
        if (!thrown_) {                                                                      \
            std::fprintf(stderr, "%s:%d: expected an exception: %s\n", __FILE__, __LINE__, #expr); \
            ++hce_test::failures;                                                            \
        }                                                                                    \
    } while (0)
