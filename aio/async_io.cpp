// async_io.cpp - backend selection and the factories.
#include "aio/async_io.hpp"

#include "aio/backends.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <new>
#include <string_view>
#include <system_error>

namespace hce::aio {

Options normalize(Options o) noexcept {
  o.queue_depth = std::clamp(o.queue_depth, 1u, 256u);
  o.chunk_bytes = std::clamp<std::size_t>(o.chunk_bytes, kIoAlign, std::size_t{64} << 20);
  o.chunk_bytes = (o.chunk_bytes + kIoAlign - 1) & ~(kIoAlign - 1);
  return o;
}

bool backend_available(Backend b) noexcept {
  switch (b) {
    case Backend::automatic:
    case Backend::threads: return true;
    case Backend::io_uring:
#if defined(HCE_AIO_HAVE_URING)
      return detail::uring_probe();
#else
      return false;
#endif
  }
  return false;
}

Backend resolve(Backend b) noexcept {
  if (b != Backend::automatic) return b;
  if (const char* env = std::getenv("HCE_AIO")) {
    const std::string_view want(env);
    if (want == "threads") return Backend::threads;
    if (want == "uring" || want == "io_uring") return Backend::io_uring;
  }
  return backend_available(Backend::io_uring) ? Backend::io_uring : Backend::threads;
}

const char* to_string(Backend b) noexcept {
  switch (b) {
    case Backend::automatic: return "automatic";
    case Backend::io_uring: return "io_uring";
    case Backend::threads: return "threads";
  }
  return "?";
}

namespace {

// One body for readers and writers: open, try io_uring, fall back to threads.
// An explicit Backend::io_uring fails instead of falling back.
template <class Io, class MakeUring, class MakeThreads>
std::unique_ptr<Io> open_file(const std::filesystem::path& path, const Options& requested, NativeFile::Mode mode,
                              OpenError* err, MakeUring make_uring, MakeThreads make_threads) noexcept {
  OpenError e;
  std::unique_ptr<Io> io;
  try {
    const Options opt = normalize(requested);
    NativeFile file = NativeFile::open(path, mode, opt.direct, e);
    if (file.is_open()) {
      const Backend want = resolve(opt.backend);
      if (want == Backend::io_uring) {
        int sys_error = 0;
        io = make_uring(file, opt, sys_error);
        if (!io && opt.backend == Backend::io_uring) e = {Errc::unsupported_feature, sys_error};
      }
      if (!io && e.status == Errc::ok) io = make_threads(std::move(file), opt);
    }
  } catch (const std::bad_alloc&) {
    e = {Errc::out_of_memory, 0};
  } catch (const std::system_error& x) {  // worker threads could not start
    e = {Errc::io_error, x.code().value()};
  }
  if (!io && e.status == Errc::ok) e = {Errc::io_error, 0};
  if (err) *err = io ? OpenError{} : e;
  return io;
}

}  // namespace

std::unique_ptr<IAsyncReader> open_reader(const std::filesystem::path& path, const Options& opt,
                                          OpenError* err) noexcept {
  return open_file<IAsyncReader>(
      path, opt, NativeFile::Mode::read, err,
      [](NativeFile& file, const Options& o, int& sys_error) -> std::unique_ptr<IAsyncReader> {
#if defined(HCE_AIO_HAVE_URING)
        return detail::make_uring_reader(file, o, sys_error);
#else
        (void)file, (void)o;
        sys_error = ENOSYS;
        return nullptr;
#endif
      },
      [](NativeFile&& file, const Options& o) { return detail::make_thread_reader(std::move(file), o); });
}

std::unique_ptr<IAsyncWriter> open_writer(const std::filesystem::path& path, const Options& opt,
                                          OpenError* err) noexcept {
  return open_file<IAsyncWriter>(
      path, opt, NativeFile::Mode::write, err,
      [](NativeFile& file, const Options& o, int& sys_error) -> std::unique_ptr<IAsyncWriter> {
#if defined(HCE_AIO_HAVE_URING)
        return detail::make_uring_writer(file, o, sys_error);
#else
        (void)file, (void)o;
        sys_error = ENOSYS;
        return nullptr;
#endif
      },
      [](NativeFile&& file, const Options& o) { return detail::make_thread_writer(std::move(file), o); });
}

}  // namespace hce::aio
