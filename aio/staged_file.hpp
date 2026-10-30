// staged_file.hpp - transactional file output: a destination changes only when
// the new content is complete.
//
// StagedFileOutput writes to a temporary file it creates exclusively beside
// the destination (same directory, so the same file system), and commit()
// publishes it with one rename. Until commit() succeeds the destination is
// untouched; a failure or a discard() removes the temporary file, and nothing
// else.
//
//   auto out = hce::aio::StagedFileOutput::create("app.log.hce", {}, &err);
//   ... out->write(...) ...
//   const auto c = out->commit(hce::aio::Publish::replace);
//   if (c.status != Errc::ok && !c.published) { /* app.log.hce as before */ }
//
// Aliases. The temporary file is a new directory entry, so the destination's
// old content stays readable until the rename: compressing a file onto itself
// cannot read its own output, but would replace its input, which callers
// should refuse first (same_file()). Publication replaces the destination
// name: a symbolic link there is replaced, never followed, and other hard
// links to the old file keep the old content.
#pragma once

#include "aio/async_streams.hpp"

#include <filesystem>
#include <memory>

namespace hce::aio {

enum class Publish : std::uint8_t {
  replace,     // the destination may exist: it is replaced atomically
  no_replace,  // fails with EEXIST (ERROR_FILE_EXISTS) if it exists; never overwrites, even in a race
};

struct CommitResult {
  Errc status = Errc::ok;
  int sys_error = 0;
  // True once the new file is at the destination. A failure with published
  // set is a durability failure (the directory could not be synced): the
  // data is in place and must not be deleted, but a crash could still lose
  // the new name.
  bool published = false;
  const char* stage = "";  // finish (write or sync), close, publish, sync-dir
};

// The publication step on its own: renames `from` to `to`, then makes the
// directory entry durable. Publish::no_replace uses renameat2(RENAME_NOREPLACE)
// on Linux, renamex_np(RENAME_EXCL) on macOS, link() + unlink() where neither
// exists, and MoveFileExW without MOVEFILE_REPLACE_EXISTING on Windows; when
// the file system offers none of them it fails with ENOTSUP and touches
// nothing: there is no check-then-rename fallback.
CommitResult publish_file(const std::filesystem::path& from, const std::filesystem::path& to, Publish how) noexcept;

// True when both paths name the same existing file (symbolic links followed,
// hard links included). False when either does not exist.
bool same_file(const std::filesystem::path& a, const std::filesystem::path& b) noexcept;

class StagedFileOutput final : public OutputStream {
 public:
  // Creates `<dir>/.<name>.hce-<random>.tmp` beside `dest`, exclusively. A
  // destination that is a directory fails here (io_error, EISDIR), one that
  // is a device, pipe or socket with unsupported_feature: a rename would
  // replace it. nullptr on failure; `err` says why.
  static std::unique_ptr<StagedFileOutput> create(const std::filesystem::path& dest, const Options& opt = {},
                                                  OpenError* err = nullptr);
  StagedFileOutput(const StagedFileOutput&) = delete;
  StagedFileOutput& operator=(const StagedFileOutput&) = delete;
  ~StagedFileOutput() { discard(); }

  bool write(Bytes buf) override { return out_ && out_->write(buf); }
  void interrupt() noexcept override {
    if (out_) out_->interrupt();
  }

  // Waits for every write, syncs the data to stable storage, closes the file
  // (checking close()'s error), takes on the old destination's permission
  // bits, publishes it and syncs the directory. Once.
  CommitResult commit(Publish how = Publish::replace) noexcept;
  // Removes the temporary file if commit() has not published it. Idempotent.
  void discard() noexcept;

  // Valid until commit() or discard().
  const char* backend() const noexcept { return out_ ? out_->backend() : ""; }
  Errc error() const noexcept { return out_ ? out_->error() : Errc::ok; }
  int sys_error() const noexcept { return out_ ? out_->sys_error() : 0; }
  const std::filesystem::path& temp_path() const noexcept { return tmp_; }
  const std::filesystem::path& path() const noexcept { return dest_; }

 private:
  StagedFileOutput(std::unique_ptr<AsyncFileOutput> out, std::filesystem::path tmp, std::filesystem::path dest)
      : out_(std::move(out)), tmp_(std::move(tmp)), dest_(std::move(dest)) {}

  std::unique_ptr<AsyncFileOutput> out_;
  std::filesystem::path tmp_, dest_;
  bool done_ = false;  // committed or discarded
};

namespace detail {
// Test hook: the no-replace methods publish_file() may use, as a bit set
// (1 renameat2/renamex_np, 2 link + unlink, 4 MoveFileExW). Default: all.
CommitResult publish_with(const std::filesystem::path& from, const std::filesystem::path& to, Publish how,
                          unsigned methods) noexcept;
inline constexpr unsigned kPublishRename = 1, kPublishLink = 2, kPublishMove = 4, kPublishAll = 7;
}  // namespace detail

}  // namespace hce::aio
