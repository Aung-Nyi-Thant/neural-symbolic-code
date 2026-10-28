#include "common.hpp"

#include <array>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#define ULTRA_MMAP 1
#endif
#if defined(__x86_64__) || defined(__i386__)
#include <nmmintrin.h>
#define ULTRA_X86 1
#endif

namespace ultracodec::detail {

namespace {

constexpr std::size_t kMapThreshold = std::size_t{1} << 20;

using CrcTables = std::array<std::array<u32, 256>, 8>;

constexpr CrcTables make_crc_tables() {
  CrcTables t{};
  for (u32 i = 0; i < 256; ++i) {
    u32 c = i;
    for (int k = 0; k < 8; ++k) c = (c & 1) ? (c >> 1) ^ 0x82F63B78u : c >> 1;
    t[0][i] = c;
  }
  for (u32 i = 0; i < 256; ++i)
    for (std::size_t s = 1; s < 8; ++s) t[s][i] = (t[s - 1][i] >> 8) ^ t[0][t[s - 1][i] & 0xFF];
  return t;
}
constexpr CrcTables kCrc = make_crc_tables();

u32 crc32c_soft(const u8* p, std::size_t n, u32 crc) noexcept {
  while (n >= 8) {
    u64 v = load64(p) ^ crc;
    crc = kCrc[7][v & 0xFF] ^ kCrc[6][(v >> 8) & 0xFF] ^ kCrc[5][(v >> 16) & 0xFF] ^
          kCrc[4][(v >> 24) & 0xFF] ^ kCrc[3][(v >> 32) & 0xFF] ^ kCrc[2][(v >> 40) & 0xFF] ^
          kCrc[1][(v >> 48) & 0xFF] ^ kCrc[0][v >> 56];
    p += 8;
    n -= 8;
  }
  while (n--) crc = (crc >> 8) ^ kCrc[0][(crc ^ *p++) & 0xFF];
  return crc;
}

#if defined(ULTRA_X86) && defined(__x86_64__)
__attribute__((target("sse4.2"))) u32 crc32c_hw(const u8* p, std::size_t n, u32 crc) noexcept {
  u64 c = crc;
  while (n >= 8) {
    c = _mm_crc32_u64(c, load64(p));
    p += 8;
    n -= 8;
  }
  u32 c32 = static_cast<u32>(c);
  while (n--) c32 = _mm_crc32_u8(c32, *p++);
  return c32;
}
const bool kHaveSse42 = __builtin_cpu_supports("sse4.2");
#endif

}  // namespace

u32 crc32c(const u8* data, std::size_t n, u32 crc) noexcept {
  crc = ~crc;
#if defined(ULTRA_X86) && defined(__x86_64__)
  if (kHaveSse42) return ~crc32c_hw(data, n, crc);
#endif
  return ~crc32c_soft(data, n, crc);
}

void* big_alloc(std::size_t bytes) {
#if defined(ULTRA_MMAP)
  if (bytes >= kMapThreshold) {
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) throw std::bad_alloc();
#if defined(MADV_HUGEPAGE)
    madvise(p, bytes, MADV_HUGEPAGE);
#endif
    return p;
  }
#endif
  std::size_t rounded = (bytes + 63) & ~std::size_t{63};
  void* p = ::operator new(rounded, std::align_val_t{64});
  std::memset(p, 0, rounded);
  return p;
}

void big_free(void* p, std::size_t bytes) noexcept {
#if defined(ULTRA_MMAP)
  if (bytes >= kMapThreshold) {
    munmap(p, bytes);
    return;
  }
#endif
  (void)bytes;
  ::operator delete(p, std::align_val_t{64});
}

void big_zero(void* p, std::size_t bytes) noexcept {
#if defined(ULTRA_MMAP) && defined(MADV_DONTNEED) && defined(__linux__)
  // Linux guarantees zero-filled pages on the next touch of a private
  // anonymous mapping after MADV_DONTNEED.
  if (bytes >= kMapThreshold && madvise(p, bytes, MADV_DONTNEED) == 0) return;
#endif
  std::memset(p, 0, bytes);
}

namespace {

#if defined(ULTRA_MMAP)
// An unnamed shared-memory file holding `bytes`, or -1.
int image_file(std::span<const u8> bytes) noexcept {
#if defined(__linux__)
  int fd = memfd_create("ultracodec-image", MFD_CLOEXEC);
#else
  static std::atomic<unsigned> serial{0};
  char name[32];
  std::snprintf(name, sizeof name, "/ultra-%ld-%u", static_cast<long>(getpid()), serial.fetch_add(1));
  int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
  if (fd >= 0) shm_unlink(name);
#endif
  if (fd < 0) return -1;
  // Filled through a mapping: macOS shared memory supports neither write() nor resizing twice.
  void* p = ftruncate(fd, static_cast<off_t>(bytes.size())) == 0
                ? mmap(nullptr, bytes.size(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
                : MAP_FAILED;
  if (p == MAP_FAILED) {
    close(fd);
    return -1;
  }
  std::memcpy(p, bytes.data(), bytes.size());
  munmap(p, bytes.size());
  return fd;
}

std::size_t page_bytes() noexcept {
  static const std::size_t page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
  return page;
}
#endif

}  // namespace

SharedImage::SharedImage(std::span<const u8> bytes) : size_(bytes.size()) {
#if defined(ULTRA_MMAP)
  if (size_ >= kMapThreshold && (fd_ = image_file(bytes)) >= 0) {
    void* p = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (p != MAP_FAILED) {
      data_ = static_cast<const u8*>(p);
      return;
    }
    close(fd_);
    fd_ = -1;
  }
#endif
  copy_.assign(bytes.begin(), bytes.end());
  data_ = copy_.data();
}

SharedImage::~SharedImage() {
#if defined(ULTRA_MMAP)
  if (fd_ >= 0) {
    munmap(const_cast<u8*>(data_), size_);
    close(fd_);
  }
#endif
}

void SharedImage::map_over(u8* dst, std::size_t bytes) const {
  if (bytes < size_) throw Error("ultracodec: shared image does not fit");
  std::size_t mapped = 0;
#if defined(ULTRA_MMAP)
  // Only big_alloc mappings are page-aligned blocks that a fixed mapping may replace. If the
  // mapping fails, dst is left as it was and is copied whole.
  if (fd_ >= 0 && bytes >= kMapThreshold && reinterpret_cast<std::uintptr_t>(dst) % page_bytes() == 0) {
    const std::size_t whole = size_ / page_bytes() * page_bytes();
    if (whole && mmap(dst, whole, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_FIXED, fd_, 0) != MAP_FAILED) mapped = whole;
  }
#endif
  // The partial last page is copied, so appending after the image dirties no shared page.
  if (size_ > mapped) std::memcpy(dst + mapped, data_ + mapped, size_ - mapped);
}

}  // namespace ultracodec::detail
