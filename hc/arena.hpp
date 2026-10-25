// hc/arena.hpp - the large, long-lived allocations: model tables and windows.
// Part of the header-only container library; hc_framed.hpp includes every part.
//
// The models probe tens of MiB of hash tables at random, several times per
// coded bit. With 4 KiB pages nearly every probe also misses the TLB, so on
// Linux an arena of 2 MiB or more is aligned to 2 MiB and offered to
// transparent huge pages (MADV_HUGEPAGE; a no-op where THP is off). Contents
// and sizes are unchanged: only the page size the kernel may back them with.
#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace hc {

struct ArenaDeleter {
  void operator()(std::byte* p) const noexcept {
#if defined(__linux__)
    std::free(p);
#else
    delete[] p;
#endif
  }
};
using ArenaPtr = std::unique_ptr<std::byte[], ArenaDeleter>;

// n bytes, uninitialized; nullptr when the allocation fails.
inline ArenaPtr allocate_arena(std::size_t n) noexcept {
#if defined(__linux__)
  constexpr std::size_t kHuge = std::size_t{2} << 20;
  if (n >= kHuge && n <= ~std::size_t{0} - kHuge) {
    const std::size_t rounded = (n + kHuge - 1) & ~(kHuge - 1);
    if (void* p = std::aligned_alloc(kHuge, rounded)) {
      (void)::madvise(p, rounded, MADV_HUGEPAGE);  // advisory: failure leaves 4 KiB pages
      return ArenaPtr(static_cast<std::byte*>(p));
    }
    return nullptr;
  }
  return ArenaPtr(static_cast<std::byte*>(std::malloc(n == 0 ? 1 : n)));
#else
  return ArenaPtr(new (std::nothrow) std::byte[n == 0 ? 1 : n]);
#endif
}

}  // namespace hc
