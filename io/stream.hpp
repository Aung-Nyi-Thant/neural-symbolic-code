// stream.hpp - byte stream interfaces for the hce pipelines.
//
// The pipelines are compiled once against these interfaces; the CLI provides
// POSIX file-descriptor implementations and the tests in-memory ones. Both
// satisfy hc::ByteSource / hc::ByteSink, so they also drive hc::pump().
#pragma once

#include "hc_framed.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

namespace hce {

class InputStream {
 public:
  // Reads up to buf.size() bytes, blocking until at least one is available.
  // Returns 0 at end of input, or on failure, in which case error() says why.
  virtual std::size_t read(hc::MutBytes buf) = 0;
  virtual hc::Errc error() const noexcept { return hc::Errc::ok; }
  // Makes a blocked or later read() return 0 promptly, with error() ==
  // cancelled. A pipeline that failed elsewhere calls it, from any thread.
  virtual void interrupt() noexcept {}

 protected:
  ~InputStream() = default;
};

class OutputStream {
 public:
  // Writes all of `buf`; false means the data could not be delivered.
  virtual bool write(hc::Bytes buf) = 0;
  // Makes a blocked or later write() return false promptly; any thread.
  virtual void interrupt() noexcept {}

 protected:
  ~OutputStream() = default;
};

static_assert(hc::ByteSource<InputStream>);
static_assert(hc::ByteSink<OutputStream>);

// In-memory implementations, for tests and embedding.
class MemoryInput final : public InputStream {
 public:
  explicit MemoryInput(hc::Bytes data, std::size_t max_read = ~std::size_t{0}) noexcept
      : data_(data), max_read_(max_read) {}
  std::size_t read(hc::MutBytes buf) override {
    const std::size_t n = std::min({buf.size(), data_.size() - pos_, max_read_});
    if (n) std::memcpy(buf.data(), data_.data() + pos_, n);
    pos_ += n;
    return n;
  }

 private:
  hc::Bytes data_;
  std::size_t max_read_;
  std::size_t pos_ = 0;
};

class MemoryOutput final : public OutputStream {
 public:
  bool write(hc::Bytes buf) override {
    data.insert(data.end(), buf.begin(), buf.end());
    return true;
  }
  std::vector<std::byte> data;
};

class NullOutput final : public OutputStream {
 public:
  bool write(hc::Bytes buf) override {
    bytes += buf.size();
    return true;
  }
  hc::u64 bytes = 0;
};

}  // namespace hce
