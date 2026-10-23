// Python bindings for orchestration, testing and analysis. The codec runs with the GIL released;
// non-bytes inputs (bytearray, memoryview, ...) are first copied into an immutable bytes object, so
// other Python threads cannot mutate the buffer while it is being read.
#include <pybind11/pybind11.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/kernels/crc32c.hpp"
#include "format/container.hpp"
#include "format/profiles.hpp"

namespace py = pybind11;

namespace {

class Input {
public:
    explicit Input(const py::object& obj)
        : bytes_(py::isinstance<py::bytes>(obj) ? py::reinterpret_borrow<py::bytes>(obj)
                                                : py::bytes(py::module_::import("builtins").attr("bytes")(obj))) {
        char* data = nullptr;
        Py_ssize_t size = 0;
        if (PyBytes_AsStringAndSize(bytes_.ptr(), &data, &size) != 0) throw py::error_already_set();
        view_ = {reinterpret_cast<const uint8_t*>(data), static_cast<size_t>(size)};
    }

    std::span<const uint8_t> view() const { return view_; }

private:
    py::bytes bytes_;
    std::span<const uint8_t> view_;
};

py::bytes to_bytes(const std::vector<uint8_t>& v) {
    return py::bytes(reinterpret_cast<const char*>(v.data()), v.size());
}

py::dict stats_dict(const hce::CodecStats& s) {
    py::dict d;
    d["trace"] = s.trace;
    d["decisions"] = s.decisions;
    d["frames_stored"] = s.frames_stored;
    d["frames_modeled"] = s.frames_modeled;
    d["copies"] = s.copies;
    d["copied_bytes"] = s.copied_bytes;
    return d;
}

const hce::Profile& profile_or_raise(int id) {
    const hce::Profile* p = (id >= 0 && id <= 255) ? hce::find_profile(static_cast<uint8_t>(id)) : nullptr;
    if (p == nullptr) throw py::value_error("unknown profile id " + std::to_string(id));
    return *p;
}

}  // namespace

PYBIND11_MODULE(hce, m) {
    m.doc() = "Bit-exact lossless hybrid compression engine (container format v1).";
    py::register_exception<hce::FormatError>(m, "FormatError", PyExc_ValueError);
    m.attr("FORMAT_VERSION") = hce::kFormatVersion;
    m.attr("DEFAULT_PROFILE") = hce::kDefaultProfile;

    m.def(
        "compress",
        [](const py::object& data, int profile, bool stats) -> py::object {
            const uint8_t id = profile_or_raise(profile).id;
            const Input in(data);
            hce::CodecStats s;
            std::vector<uint8_t> out;
            {
                py::gil_scoped_release release;
                out = hce::compress(in.view(), id, &s);
            }
            if (stats) return py::make_tuple(to_bytes(out), stats_dict(s));
            return to_bytes(out);
        },
        py::arg("data"), py::arg("profile") = hce::kDefaultProfile, py::arg("stats") = false,
        "Compress a bytes-like object. With stats=True, return (blob, stats).");

    m.def(
        "decompress",
        [](const py::object& data, uint64_t max_output, bool stats) -> py::object {
            const Input in(data);
            hce::DecodeLimits limits;
            limits.max_output = max_output;
            hce::CodecStats s;
            std::vector<uint8_t> out;
            {
                py::gil_scoped_release release;
                out = hce::decompress(in.view(), limits, &s);
            }
            if (stats) return py::make_tuple(to_bytes(out), stats_dict(s));
            return to_bytes(out);
        },
        py::arg("data"), py::arg("max_output") = hce::DecodeLimits{}.max_output, py::arg("stats") = false,
        "Decompress a stream; raises FormatError (a ValueError) for any invalid or corrupted input.");

    m.def(
        "model_id", [](int profile) { return hce::model_id(profile_or_raise(profile)); }, py::arg("profile"),
        "Model id written to the header for a profile.");

    m.def(
        "profiles",
        [] {
            py::list out;
            for (const hce::Profile& p : hce::profiles()) {
                py::dict d;
                d["id"] = p.id;
                d["name"] = p.name;
                d["modeled"] = p.modeled;
                d["frame_size"] = p.frame_size;
                d["model_id"] = hce::model_id(p);
                d["copy"] = p.model.anchors.enabled;
                if (p.modeled) {
                    py::list orders;
                    for (size_t i = 0; i < p.model.num_orders; ++i) orders.append(p.model.orders[i]);
                    d["orders"] = orders;
                    d["table_bytes"] = (uint64_t{1} << p.model.table_bits) * 16;
                }
                out.append(d);
            }
            return out;
        },
        "Describe the available profiles.");

    m.def(
        "crc32c", [](const py::object& data, uint32_t crc) { return hce::crc32c(Input(data).view(), crc); },
        py::arg("data"), py::arg("crc") = 0, "CRC-32C (Castagnoli) of a bytes-like object.");
}
