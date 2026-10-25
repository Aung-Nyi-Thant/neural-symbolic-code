// hce_python.cpp - the hce._hce extension module (pybind11).
//
// A thin layer over the engine's public C++ API: the byte codec through
// encode_parallel / decode_stream, the structural codec through
// structural::compress_stream / decompress_stream, files through the aio
// layer, and the Phase 1 tokenizer. The Python package (python/hce) adds the
// typed front end; this file owns argument checking, error mapping and the GIL.
//
// Conventions:
//   * Inputs are any C-contiguous bytes-like object. They are copied before
//     the GIL is released, so a caller mutating a bytearray in another thread
//     cannot race the codec.
//   * Engine failures raise the exception classes defined below, each with
//     .code (the engine's Errc name), .stage (where it failed, or None) and
//     .errno (the OS error, or 0). Out of memory raises MemoryError; bad
//     arguments raise ValueError or TypeError.
//   * Defaults match the hce command-line tool, so both write identical bytes.
#include "aio/async_streams.hpp"
#include "codec/neural_cm_codec.hpp"
#include "decoder/pipeline.hpp"
#include "encoder/parallel.hpp"
#include "io/stream.hpp"
#include "structural/archive.hpp"
#include "structural/structural_cm_codec.hpp"
#include "structural/token_stream.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#ifndef HCE_VERSION
#error "HCE_VERSION must be defined by the build (CMake passes PROJECT_VERSION)"
#endif

namespace py = pybind11;
namespace st = hce::structural;
namespace fs = std::filesystem;
using hc::Errc;
using hc::u64;
using Vec = std::vector<std::byte>;

namespace {

constexpr u64 kDefaultMaxMemory = u64{1} << 30;

// ------------------------------------------------------------ exceptions ---

// Created once at import; the module keeps them alive (and so does this
// table, deliberately never released: interpreter teardown order is undefined).
struct ExceptionTypes {
  PyObject* base = nullptr;
  PyObject* corrupt = nullptr;
  PyObject* limit = nullptr;
  PyObject* unsupported = nullptr;
  PyObject* io = nullptr;
};
ExceptionTypes g_exc;

PyObject* new_exception(py::module_& m, const char* name, PyObject* bases, const char* doc) {
  const std::string qualified = std::string("hce._hce.") + name;
  PyObject* t = PyErr_NewExceptionWithDoc(qualified.c_str(), doc, bases, nullptr);
  if (!t) throw py::error_already_set();
  m.add_object(name, py::reinterpret_borrow<py::object>(t));
  return t;
}

PyObject* exception_for(Errc e) noexcept {
  if (hc::is_limit(e)) return g_exc.limit;
  switch (e) {
    case Errc::unsupported_version:
    case Errc::unsupported_feature: return g_exc.unsupported;
    case Errc::io_error: return g_exc.io;
    case Errc::bad_magic:
    case Errc::header_corrupt:
    case Errc::payload_corrupt:
    case Errc::content_mismatch:
    case Errc::bad_reference:
    case Errc::sequence_gap:
    case Errc::truncated:
    case Errc::trailing_garbage: return g_exc.corrupt;
    default: return g_exc.base;
  }
}

[[noreturn]] void raise_as(PyObject* type, Errc e, const std::string& message, const char* stage, int sys_error) {
  py::object inst = py::reinterpret_borrow<py::object>(type)(message);
  inst.attr("code") = hc::to_string(e);
  inst.attr("stage") = (stage && *stage) ? py::object(py::str(stage)) : py::none();
  inst.attr("errno") = sys_error;
  PyErr_SetObject(type, inst.ptr());
  throw py::error_already_set();
}

[[noreturn]] void raise_errc(Errc e, const std::string& what, const char* stage = nullptr, int sys_error = 0) {
  if (e == Errc::out_of_memory) throw std::bad_alloc();
  std::string message = what + ": " + hc::to_string(e);
  if (stage && *stage) message += " (" + std::string(stage) + ")";
  if (sys_error) message += ": " + std::system_category().message(sys_error);
  raise_as(exception_for(e), e, message, stage, sys_error);
}

// ------------------------------------------------------------- arguments ---

Vec copy_buffer(const py::buffer& b) {
  const py::buffer_info info = b.request();
  if (info.ndim > 1) {
    for (py::ssize_t i = info.ndim - 1, stride = info.itemsize; i >= 0; stride *= info.shape[static_cast<std::size_t>(i)], --i)
      if (info.strides[static_cast<std::size_t>(i)] != stride) throw py::value_error("the buffer must be C-contiguous");
  } else if (info.ndim == 1 && info.strides[0] != info.itemsize) {
    throw py::value_error("the buffer must be C-contiguous");
  }
  const auto n = static_cast<std::size_t>(info.size * info.itemsize);
  const auto* p = static_cast<const std::byte*>(info.ptr);
  return Vec(p, p + n);
}

hc::u8 check_level(int level) {
  if (level < 0 || level > hce::kNeuralCMMaxLevel) throw py::value_error("level must be 0..4");
  return static_cast<hc::u8>(level);
}

hc::u32 check_profile(int profile) {
  if (profile < 1 || profile > hce::kNeuralCMMaxProfile) throw py::value_error("profile must be 1 or 2");
  return hce::kNeuralCMFamily | static_cast<hc::u32>(profile);
}

unsigned check_threads(int threads) {
  if (threads < 1 || threads > static_cast<int>(hce::kMaxThreads)) throw py::value_error("threads must be 1..256");
  return static_cast<unsigned>(threads);
}

u64 check_memory(u64 max_memory) {
  if (max_memory < (u64{16} << 20)) throw py::value_error("max_memory must be at least 16 MiB");
  return max_memory;
}

st::Language check_language(const std::string& name) {
  if (const auto l = st::parse_language(name)) return *l;
  throw py::value_error("unknown language '" + name +
                        "' (generic, c, cpp, python, javascript, typescript, java, go, rust)");
}

// "auto" resolves from the file name; anything else must be a language.
st::Language file_language(const std::string& name, const fs::path& source) {
  return name == "auto" ? st::language_for_path(source) : check_language(name);
}

enum class Format { unknown, framed, structural };

Format format_of(hc::Bytes head) noexcept {
  if (st::is_archive(head)) return Format::structural;
  if (head.size() >= hc::kMagic.size() && std::equal(hc::kMagic.begin(), hc::kMagic.end(), head.begin()))
    return Format::framed;
  return Format::unknown;
}

const char* to_string(Format f) noexcept {
  return f == Format::structural ? "structural" : f == Format::framed ? "framed" : "unknown";
}

Format file_format(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  std::byte head[4]{};
  if (!f) return Format::unknown;  // the open below reports why
  f.read(reinterpret_cast<char*>(head), sizeof head);
  return format_of(hc::Bytes(head, static_cast<std::size_t>(f.gcount())));
}

// --------------------------------------------------------------- outputs ---

// Fails a write that would pass `limit` total bytes, and remembers it did.
class BoundedOutput final : public hce::OutputStream {
 public:
  BoundedOutput(hce::OutputStream& out, u64 limit) noexcept : out_(out), limit_(limit) {}
  bool write(hc::Bytes buf) override {
    if (buf.size() > limit_ - count_) {
      exceeded_ = true;
      return false;
    }
    count_ += buf.size();
    return out_.write(buf);
  }
  bool exceeded() const noexcept { return exceeded_; }
  u64 count() const noexcept { return count_; }

 private:
  hce::OutputStream& out_;
  u64 limit_;
  u64 count_ = 0;
  bool exceeded_ = false;
};

// ------------------------------------------------------------ statistics ---

py::dict framed_stats(const hce::EncodeResult& r) {
  py::dict d;
  d["format"] = "framed";
  d["input_bytes"] = r.bytes_in;
  d["output_bytes"] = r.bytes_out;
  d["threads"] = r.threads;
  d["verified"] = false;
  d["frames"] = r.frames;
  d["blocks"] = r.blocks;
  d["stored_blocks"] = r.stored_blocks;
  return d;
}

py::dict framed_stats(const hce::DecodeResult& r) {
  py::dict d;
  d["format"] = "framed";
  d["input_bytes"] = r.bytes_in;
  d["output_bytes"] = r.bytes_out;
  d["threads"] = r.threads ? r.threads : 1u;
  d["verified"] = r.stats.content_verified && r.stats.damage_events == 0;
  d["frames"] = r.stats.frames;
  return d;
}

py::dict structural_stats(const st::ArchiveStats& s, bool decoding) {
  py::dict d;
  d["format"] = "structural";
  d["input_bytes"] = decoding ? s.archive_bytes : s.source_bytes;
  d["output_bytes"] = decoding ? s.source_bytes : s.archive_bytes;
  d["threads"] = std::max(s.threads, 1u);  // 0: the sequential decoder
  d["verified"] = decoding;
  d["language"] = st::to_string(s.language);
  d["tokens"] = s.tokens;
  d["symbols"] = s.symbols;
  d["dictionary_bytes"] = s.dictionary_bytes;
  d["model_bytes"] = s.model_bytes;
  d["stream_bytes"] = s.stream_bytes;
  return d;
}

// ------------------------------------------------------- the engine calls ---
// Each runs without the GIL and reports through its return value.

struct Encoded {
  hce::EncodeResult framed;
  st::ArchiveStats structural;
  Vec data;
};

Encoded encode(const Vec& input, hc::u8 level, hc::u32 prior_id, unsigned threads, bool structural,
               st::Language lang) {
  Encoded e;
  hce::MemoryInput in(input);
  hce::MemoryOutput out;
  if (structural) {
    st::ArchiveOptions o;
    o.level = level;
    o.threads = threads;
    o.language = lang;
    e.structural = st::compress_stream(in, out, o);
  } else {
    hce::EncodeOptions o;
    o.level = level;
    o.prior_id = prior_id;
    o.threads = threads;
    o.content_size = input.size();  // as the CLI records a regular file's size
    e.framed = hce::encode_parallel<hce::NeuralCMCodec>(in, out, o);
  }
  e.data = std::move(out.data);
  return e;
}

struct Decoded {
  Format format = Format::unknown;
  hce::DecodeResult framed;
  st::ArchiveStats structural;
  bool exceeded = false;
};

Decoded decode(hce::InputStream& in, Format format, hce::OutputStream& sink, unsigned threads, u64 max_memory,
               u64 max_output) {
  Decoded d;
  d.format = format;
  BoundedOutput out(sink, max_output);
  if (format == Format::structural) {
    st::ArchiveOptions o;
    o.threads = threads;
    o.max_memory = max_memory;
    d.structural = st::decompress_stream(in, out, o);
  } else {
    hce::DecodeOptions o;
    o.threads = threads;
    o.limits.max_memory_bytes = max_memory;
    o.limits.max_output_bytes = max_output;
    d.framed = hce::decode_stream<hce::NeuralCMCodec>(in, out, o);
  }
  d.exceeded = out.exceeded();
  return d;
}

// Raises unless `d` is a complete, verified decode; returns its statistics.
py::dict check_decoded(const Decoded& d, const char* what) {
  if (d.exceeded) raise_errc(Errc::limit_output, std::string(what) + ": output larger than max_output");
  if (d.format == Format::structural) {
    if (d.structural.status != Errc::ok) raise_errc(d.structural.status, what, d.structural.stage, d.structural.sys_error);
    return structural_stats(d.structural, true);
  }
  const hce::DecodeResult& r = d.framed;
  if (r.status != Errc::ok) raise_errc(r.status, what);
  if (!r.stats.content_verified || r.stats.damage_events != 0)
    raise_errc(Errc::payload_corrupt, std::string(what) + ": the stream did not verify");
  return framed_stats(r);
}

// A file that could not be opened: not a regular file (a directory, a pipe, a
// device) or an OS error. Both are HceIOError: the format is not at fault.
[[noreturn]] void raise_open_error(const char* fn, const fs::path& p, Errc e, int sys_error) {
  if (e == Errc::out_of_memory) throw std::bad_alloc();
  const std::string why = e == Errc::unsupported_feature ? std::string("not a regular file")
                          : sys_error                    ? std::system_category().message(sys_error)
                                                         : std::string(hc::to_string(e));
  raise_as(g_exc.io, Errc::io_error, std::string(fn) + ": cannot open " + p.string() + ": " + why, "open", sys_error);
}

void remove_quietly(const fs::path& p) noexcept {
  std::error_code ec;
  fs::remove(p, ec);
}

// ------------------------------------------------------- module functions ---

py::bytes py_compress(const py::buffer& data, int level, int profile, int threads, bool structural,
                      const std::string& language) {
  const hc::u8 lvl = check_level(level);
  const hc::u32 prior = check_profile(profile);
  const unsigned t = check_threads(threads);
  if (language == "auto") throw py::value_error("language 'auto' needs a file name: use compress_file(), or name the language");
  const st::Language lang = check_language(language);
  const Vec input = copy_buffer(data);
  Encoded e;
  {
    py::gil_scoped_release nogil;
    e = encode(input, lvl, prior, t, structural, lang);
  }
  if (structural && e.structural.status != Errc::ok)
    raise_errc(e.structural.status, "compress", e.structural.stage, e.structural.sys_error);
  if (!structural && e.framed.status != Errc::ok) raise_errc(e.framed.status, "compress");
  return py::bytes(reinterpret_cast<const char*>(e.data.data()), e.data.size());
}

py::bytes py_decompress(const py::buffer& data, int threads, u64 max_memory, std::optional<u64> max_output) {
  const unsigned t = check_threads(threads);
  const u64 mem = check_memory(max_memory);
  const Vec input = copy_buffer(data);
  const Format f = format_of(input);
  if (f == Format::unknown)
    raise_errc(input.empty() ? Errc::truncated : Errc::bad_magic, "decompress: not an hce stream or archive");
  hce::MemoryOutput out;
  Decoded d;
  {
    py::gil_scoped_release nogil;
    hce::MemoryInput in(input);
    d = decode(in, f, out, t, mem, max_output.value_or(~u64{0}));
  }
  check_decoded(d, "decompress");
  return py::bytes(reinterpret_cast<const char*>(out.data.data()), out.data.size());
}

py::dict py_verify(const py::buffer& data, int threads, u64 max_memory) {
  const unsigned t = check_threads(threads);
  const u64 mem = check_memory(max_memory);
  const Vec input = copy_buffer(data);
  const Format f = format_of(input);
  if (f == Format::unknown)
    raise_errc(input.empty() ? Errc::truncated : Errc::bad_magic, "verify: not an hce stream or archive");
  hce::NullOutput out;
  Decoded d;
  {
    py::gil_scoped_release nogil;
    hce::MemoryInput in(input);
    d = decode(in, f, out, t, mem, ~u64{0});
  }
  return check_decoded(d, "verify");
}

py::dict py_compress_file(const fs::path& source, const fs::path& dest, int level, int profile, int threads,
                          bool structural, const std::string& language) {
  const hc::u8 lvl = check_level(level);
  const hc::u32 prior = check_profile(profile);
  const unsigned t = check_threads(threads);
  const st::Language lang = file_language(language, source);
  if (structural) {
    st::ArchiveOptions o;
    o.level = lvl;
    o.threads = t;
    o.language = lang;
    st::ArchiveStats s;
    {
      py::gil_scoped_release nogil;
      s = st::compress_file(source, dest, o);
    }
    if (s.status != Errc::ok) raise_errc(s.status, "compress_file " + source.string(), s.stage, s.sys_error);
    return structural_stats(s, false);
  }
  hce::EncodeResult r;
  hc::Errc open_status = Errc::ok, finish_status = Errc::ok;
  int open_errno = 0;
  const fs::path* open_path = &source;
  {
    py::gil_scoped_release nogil;
    hce::aio::OpenError oe;
    auto in = hce::aio::AsyncFileInput::open(source, {}, &oe);
    std::unique_ptr<hce::aio::AsyncFileOutput> out;
    if (in) out = hce::aio::AsyncFileOutput::create(dest, {}, &oe);
    if (!in || !out) {
      open_status = oe.status;
      open_errno = oe.sys_error;
      open_path = in ? &dest : &source;
    } else {
      hce::EncodeOptions o;
      o.level = lvl;
      o.prior_id = prior;
      o.threads = t;
      o.content_size = in->size();
      r = hce::encode_parallel<hce::NeuralCMCodec>(*in, *out, o);
      if (r.status == Errc::ok) finish_status = out->finish(true);
      out.reset();
      if (r.status != Errc::ok || finish_status != Errc::ok) remove_quietly(dest);
    }
  }
  if (open_status != Errc::ok) raise_open_error("compress_file", *open_path, open_status, open_errno);
  if (r.status != Errc::ok) raise_errc(r.status, "compress_file " + source.string());
  if (finish_status != Errc::ok) raise_errc(finish_status, "compress_file " + dest.string(), "finish");
  return framed_stats(r);
}

py::dict py_decompress_file(const fs::path& source, const fs::path& dest, int threads, u64 max_memory) {
  const unsigned t = check_threads(threads);
  const u64 mem = check_memory(max_memory);
  hc::Errc open_status = Errc::ok, finish_status = Errc::ok;
  int open_errno = 0;
  const fs::path* open_path = &source;
  Decoded d;
  {
    py::gil_scoped_release nogil;
    hce::aio::OpenError oe;
    auto in = hce::aio::AsyncFileInput::open(source, {}, &oe);
    std::unique_ptr<hce::aio::AsyncFileOutput> out;
    if (in) out = hce::aio::AsyncFileOutput::create(dest, {}, &oe);
    if (!in || !out) {
      open_status = oe.status;
      open_errno = oe.sys_error;
      open_path = in ? &dest : &source;
    } else {
      // The format, from the first bytes, read through a separate handle: `in` still starts at 0.
      d.format = file_format(source);
      if (d.format != Format::unknown) {
        d = decode(*in, d.format, *out, t, mem, ~u64{0});
        const bool ok = !d.exceeded && (d.format == Format::structural
                                            ? d.structural.status == Errc::ok
                                            : d.framed.status == Errc::ok && d.framed.stats.content_verified &&
                                                  d.framed.stats.damage_events == 0);
        if (ok) finish_status = out->finish(true);
        if (!ok || finish_status != Errc::ok) {
          out.reset();
          remove_quietly(dest);
        }
      } else {
        out.reset();
        remove_quietly(dest);
      }
    }
  }
  if (open_status != Errc::ok) raise_open_error("decompress_file", *open_path, open_status, open_errno);
  if (d.format == Format::unknown) raise_errc(Errc::bad_magic, "decompress_file " + source.string() + ": not an hce stream or archive");
  py::dict stats = check_decoded(d, ("decompress_file " + source.string()).c_str());
  if (finish_status != Errc::ok) raise_errc(finish_status, "decompress_file " + dest.string(), "finish");
  return stats;
}

py::object py_detect_format(const py::buffer& data) {
  const Vec head = [&] {
    Vec v = copy_buffer(data);
    if (v.size() > 4) v.resize(4);
    return v;
  }();
  const Format f = format_of(head);
  return f == Format::unknown ? py::none() : py::object(py::str(to_string(f)));
}

py::list py_tokenize(const py::object& source, const std::string& language) {
  std::string text;
  if (PyUnicode_Check(source.ptr())) {
    text = source.cast<std::string>();  // UTF-8
  } else {
    const Vec v = copy_buffer(source.cast<py::buffer>());
    text.assign(reinterpret_cast<const char*>(v.data()), v.size());
  }
  const st::Language lang = check_language(language);
  if (text.size() > st::kMaxSourceBytes) throw py::value_error("source larger than 1 GiB: tokenize it in pieces");
  std::vector<st::StructuralToken> tokens;
  {
    py::gil_scoped_release nogil;
    tokens = st::tokenize(text, lang);
  }
  static constexpr const char* kNames[] = {"KEYWORD", "IDENTIFIER", "LITERAL", "OPERATOR", "STRUCTURE", "INDENTATION"};
  py::list out;
  for (const st::StructuralToken& t : tokens)
    out.append(py::make_tuple(kNames[static_cast<int>(t.type)], py::bytes(t.value), t.line_number, t.checked_integer_id));
  return out;
}

py::dict py_info() {
  py::dict d;
  d["version"] = HCE_VERSION;
  d["neural_cm_codec_id"] = hce::kNeuralCMCodecId;
  d["neural_cm_codec_ids"] = py::make_tuple(hce::kNeuralCMCodecId, hce::kNeuralCMCodecId2);
  d["max_profile"] = static_cast<int>(hce::kNeuralCMMaxProfile);
  d["structural_codec_id"] = hce::kStructuralCMCodecId;
  d["max_level"] = static_cast<int>(hce::kNeuralCMMaxLevel);
  d["simd_path"] = hce::NeuralCMCodec::simd_path();
  d["aio_backend"] = hce::aio::to_string(hce::aio::resolve(hce::aio::Backend::automatic));
  py::list langs;
  for (std::uint8_t l = 0; l < st::kLanguages; ++l) langs.append(st::to_string(static_cast<st::Language>(l)));
  d["languages"] = langs;
  return d;
}

}  // namespace

PYBIND11_MODULE(_hce, m) {
  m.doc() = "HCE engine bindings. Use the hce package, not this module.";

  g_exc.base = new_exception(m, "HceError", PyExc_Exception, "Base class of every error the engine reports.");
  {
    py::tuple b = py::make_tuple(py::handle(g_exc.base), py::handle(PyExc_ValueError));
    g_exc.corrupt = new_exception(m, "CorruptInputError", b.ptr(), "The input is damaged, truncated or not hce data.");
  }
  g_exc.limit = new_exception(m, "LimitExceededError", g_exc.base, "A resource limit (memory, output size) was exceeded.");
  g_exc.unsupported = new_exception(m, "UnsupportedFormatError", g_exc.base, "A format version, codec or feature this build does not support.");
  {
    py::tuple b = py::make_tuple(py::handle(g_exc.base), py::handle(PyExc_OSError));
    g_exc.io = new_exception(m, "HceIOError", b.ptr(), "A file could not be opened, read or written.");
  }

  m.attr("__version__") = HCE_VERSION;
  m.attr("DEFAULT_MAX_MEMORY") = kDefaultMaxMemory;

  m.def("compress", &py_compress, py::arg("data"), py::kw_only(), py::arg("level") = 2,
        py::arg("profile") = static_cast<int>(hce::kNeuralCMProfile2), py::arg("threads") = 1,
        py::arg("structural") = false, py::arg("language") = "generic");
  m.def("decompress", &py_decompress, py::arg("data"), py::kw_only(), py::arg("threads") = 1,
        py::arg("max_memory") = kDefaultMaxMemory, py::arg("max_output") = py::none());
  m.def("verify", &py_verify, py::arg("data"), py::kw_only(), py::arg("threads") = 1,
        py::arg("max_memory") = kDefaultMaxMemory);
  m.def("compress_file", &py_compress_file, py::arg("source"), py::arg("dest"), py::kw_only(), py::arg("level") = 2,
        py::arg("profile") = static_cast<int>(hce::kNeuralCMProfile2), py::arg("threads") = 1,
        py::arg("structural") = false, py::arg("language") = "auto");
  m.def("decompress_file", &py_decompress_file, py::arg("source"), py::arg("dest"), py::kw_only(),
        py::arg("threads") = 1, py::arg("max_memory") = kDefaultMaxMemory);
  m.def("detect_format", &py_detect_format, py::arg("data"));
  m.def("tokenize", &py_tokenize, py::arg("source"), py::kw_only(), py::arg("language") = "generic");
  m.def("info", &py_info);
}
