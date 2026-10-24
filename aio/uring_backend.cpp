// uring_backend.cpp - the Linux io_uring engine (liburing >= 2.2).
//
// Per operation it costs an SQE fill, no system call: submit() starts a whole
// batch with one io_uring_enter, and reap() collects completions in bulk.
// The file is registered (IOSQE_FIXED_FILE: no fget/fput per operation) and,
// when the caller registers its buffers, reads and writes use the fixed-buffer
// opcodes, so pages are pinned once instead of on every operation.
//
// Ring flags: COOP_TASKRUN and SUBMIT_ALL when the kernel has them. Not
// SINGLE_ISSUER / DEFER_TASKRUN: they bind the ring to one thread, while the
// stream adapters may be driven by different threads over their lifetime
// (a pipeline's writer thread, then the owner calling finish()).
#include "aio/backends.hpp"

#include <liburing.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <vector>

namespace hce::aio::detail {
namespace {

constexpr long long kWaitSliceNs = 20'000'000;  // interrupt() latency bound while waiting
constexpr unsigned kReapBatch = 64;

class UringEngine {
 public:
  static constexpr const char* kName = "io_uring";

  // Takes the file only when the ring exists; otherwise `sys_error` says why.
  UringEngine(NativeFile& file, const Options& opt, int& sys_error) noexcept : depth_(opt.queue_depth) {
    io_uring_params p{};
#ifdef IORING_SETUP_COOP_TASKRUN
    p.flags |= IORING_SETUP_COOP_TASKRUN;  // 5.19: no IPI to run completion work
#endif
#ifdef IORING_SETUP_SUBMIT_ALL
    p.flags |= IORING_SETUP_SUBMIT_ALL;  // 5.18: one bad SQE does not stall the batch
#endif
    int r = io_uring_queue_init_params(depth_, &ring_, &p);
    if (r == -EINVAL && p.flags != 0) {  // older kernel: plain ring
      p = {};
      r = io_uring_queue_init_params(depth_, &ring_, &p);
    }
    if (r < 0) {
      sys_error = -r;
      return;
    }
    ring_ok_ = true;
    file_ = std::move(file);
    const int fd = file_.handle();
    fixed_file_ = io_uring_register_files(&ring_, &fd, 1) == 0;
    sys_error = 0;
  }

  UringEngine(const UringEngine&) = delete;
  UringEngine& operator=(const UringEngine&) = delete;

  // The kernel writes into caller memory until each operation completes, and
  // ring teardown does not wait for that: collect every submitted operation.
  ~UringEngine() {
    if (!ring_ok_) return;
    while (submitted_ > 0) {
      io_uring_cqe* cqe = nullptr;
      const int r = io_uring_wait_cqe(&ring_, &cqe);
      if (r == -EINTR) continue;
      if (r < 0) break;
      if (cqe->user_data != LIBURING_UDATA_TIMEOUT) --submitted_;
      io_uring_cqe_seen(&ring_, cqe);
    }
    io_uring_queue_exit(&ring_);  // also unregisters the file and the buffers
  }

  unsigned depth() const noexcept { return depth_; }
  const NativeFile& file() const noexcept { return file_; }

  void register_buffers(std::span<const MutBytes> buffers) noexcept {
    if (!fixed_.empty() || buffers.empty() || buffers.size() > 1024) return;
    try {
      std::vector<iovec> v;
      v.reserve(buffers.size());
      for (const MutBytes b : buffers) v.push_back(iovec{b.data(), b.size()});
      // Fails under a low RLIMIT_MEMLOCK on kernels before 5.12: plain opcodes then.
      if (io_uring_register_buffers(&ring_, v.data(), static_cast<unsigned>(v.size())) == 0) fixed_ = std::move(v);
    } catch (...) {
    }
  }

  Errc queue(const Op& op) noexcept {
    if (outstanding_ >= depth_) return Errc::output_full;
    io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (sqe == nullptr) {  // SQ full of unsubmitted entries: start them
      if (submit() != Errc::ok) return Errc::io_error;
      sqe = io_uring_get_sqe(&ring_);
      if (sqe == nullptr) return Errc::output_full;
    }
    const int fd = fixed_file_ ? 0 : file_.handle();
    const auto len = static_cast<unsigned>(std::min(op.len, kMaxIoBytes));
    const int index = op.kind == OpKind::sync ? -1 : fixed_index(op.buf, len);
    switch (op.kind) {
      case OpKind::read:
        if (index >= 0) io_uring_prep_read_fixed(sqe, fd, op.buf, len, op.offset, index);
        else io_uring_prep_read(sqe, fd, op.buf, len, op.offset);
        break;
      case OpKind::write:
        if (index >= 0) io_uring_prep_write_fixed(sqe, fd, op.buf, len, op.offset, index);
        else io_uring_prep_write(sqe, fd, op.buf, len, op.offset);
        break;
      case OpKind::sync:
        io_uring_prep_fsync(sqe, fd, IORING_FSYNC_DATASYNC);
        break;
    }
    if (fixed_file_) sqe->flags |= IOSQE_FIXED_FILE;
    io_uring_sqe_set_data64(sqe, op.tag);
    ++outstanding_;
    ++unsubmitted_;
    return Errc::ok;
  }

  Errc submit() noexcept {
    while (unsubmitted_ > 0) {
      const int r = io_uring_submit(&ring_);
      if (r == -EINTR || r == -EAGAIN) continue;
      if (r <= 0) return Errc::io_error;
      const auto n = std::min(static_cast<unsigned>(r), unsubmitted_);
      unsubmitted_ -= n;
      submitted_ += n;
    }
    return Errc::ok;
  }

  std::size_t reap(std::span<Completion> out, std::size_t min) noexcept {
    if (unsubmitted_ > 0 && submit() != Errc::ok) return 0;
    min = std::min({min, out.size(), static_cast<std::size_t>(submitted_)});
    while (io_uring_cq_ready(&ring_) < min && !interrupted_.load(std::memory_order_acquire)) {
      __kernel_timespec ts{};
      ts.tv_nsec = kWaitSliceNs;
      io_uring_cqe* cqe = nullptr;
      const auto want = static_cast<unsigned>(min - io_uring_cq_ready(&ring_));
      const int r = io_uring_wait_cqes(&ring_, &cqe, want, &ts, nullptr);
      if (r < 0 && r != -ETIME && r != -EINTR) break;  // the ring failed: callers see too few
    }
    io_uring_cqe* cqes[kReapBatch];
    const unsigned got = io_uring_peek_batch_cqe(&ring_, cqes, std::min<unsigned>(kReapBatch, static_cast<unsigned>(out.size())));
    std::size_t n = 0;
    for (unsigned i = 0; i < got; ++i) {
      const io_uring_cqe* c = cqes[i];
      if (c->user_data == LIBURING_UDATA_TIMEOUT) continue;  // liburing's own, on kernels without EXT_ARG
      out[n++] = Completion{c->user_data, c->res > 0 ? static_cast<std::size_t>(c->res) : 0,
                            c->res < 0 ? -c->res : 0};
    }
    io_uring_cq_advance(&ring_, got);
    outstanding_ -= static_cast<unsigned>(n);
    submitted_ -= static_cast<unsigned>(n);
    return n;
  }

  void interrupt() noexcept { interrupted_.store(true, std::memory_order_release); }

 private:
  int fixed_index(const std::byte* p, std::size_t len) const noexcept {
    for (std::size_t i = 0; i < fixed_.size(); ++i) {
      const auto* base = static_cast<const std::byte*>(fixed_[i].iov_base);
      if (p >= base && p + len <= base + fixed_[i].iov_len) return static_cast<int>(i);
    }
    return -1;
  }

  io_uring ring_{};
  bool ring_ok_ = false;
  bool fixed_file_ = false;
  NativeFile file_;
  unsigned depth_;
  unsigned outstanding_ = 0;  // queued and not yet reaped
  unsigned unsubmitted_ = 0;  // queued, not yet given to the kernel
  unsigned submitted_ = 0;    // given to the kernel, not yet reaped
  std::vector<iovec> fixed_;  // registered buffers, by index
  std::atomic<bool> interrupted_{false};
};

template <class Facade>
std::unique_ptr<Facade> make(NativeFile& file, const Options& opt, int& sys_error) {
  auto io = std::make_unique<Facade>(file, opt, sys_error);
  if (sys_error != 0) io.reset();
  return io;
}

}  // namespace

bool uring_probe() noexcept {
  static const bool ok = [] {
    io_uring ring{};
    if (io_uring_queue_init(4, &ring, 0) < 0) return false;  // ENOSYS, EPERM (seccomp, sysctl), ENOMEM
    bool good = false;
    if (io_uring_probe* p = io_uring_get_probe_ring(&ring)) {  // 5.6+: IORING_OP_READ / WRITE exist
      good = io_uring_opcode_supported(p, IORING_OP_READ) && io_uring_opcode_supported(p, IORING_OP_WRITE) &&
             io_uring_opcode_supported(p, IORING_OP_FSYNC);
      io_uring_free_probe(p);
    }
    io_uring_queue_exit(&ring);
    return good;
  }();
  return ok;
}

std::unique_ptr<IAsyncReader> make_uring_reader(NativeFile& file, const Options& opt, int& sys_error) {
  return make<Reader<UringEngine>>(file, opt, sys_error);
}

std::unique_ptr<IAsyncWriter> make_uring_writer(NativeFile& file, const Options& opt, int& sys_error) {
  return make<Writer<UringEngine>>(file, opt, sys_error);
}

}  // namespace hce::aio::detail
