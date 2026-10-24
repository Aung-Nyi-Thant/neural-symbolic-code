// buffer_pool.hpp - equal, page-aligned I/O buffers in one allocation.
//
// Alignment to kIoAlign makes them valid for O_DIRECT / FILE_FLAG_NO_BUFFERING
// and lets io_uring register them as fixed buffers.
#pragma once

#include "aio/async_io.hpp"

#include <cstddef>
#include <new>
#include <vector>

namespace hce::aio {

class BufferPool {
 public:
  // Throws std::bad_alloc. `size` is rounded up to kIoAlign.
  BufferPool(std::size_t count, std::size_t size)
      : count_(count), size_((size + kIoAlign - 1) & ~(kIoAlign - 1)) {
    base_ = static_cast<std::byte*>(::operator new(count_ * size_, std::align_val_t{kIoAlign}));
    try {
      spans_.reserve(count_);
      for (std::size_t i = 0; i < count_; ++i) spans_.emplace_back(base_ + i * size_, size_);
    } catch (...) {
      ::operator delete(base_, std::align_val_t{kIoAlign});
      throw;
    }
  }
  BufferPool(const BufferPool&) = delete;
  BufferPool& operator=(const BufferPool&) = delete;
  ~BufferPool() { ::operator delete(base_, std::align_val_t{kIoAlign}); }

  std::size_t count() const noexcept { return count_; }
  std::size_t size() const noexcept { return size_; }
  std::byte* data(std::size_t i) const noexcept { return base_ + i * size_; }
  MutBytes buffer(std::size_t i) const noexcept { return spans_[i]; }
  std::span<const MutBytes> all() const noexcept { return spans_; }

 private:
  std::size_t count_, size_;
  std::byte* base_ = nullptr;
  std::vector<MutBytes> spans_;
};

}  // namespace hce::aio
