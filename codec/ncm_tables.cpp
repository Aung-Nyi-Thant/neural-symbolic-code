// ncm_tables.cpp - the counter transition table (see ncm_tables.hpp).
#include "codec/ncm_tables.hpp"

namespace hce::ncm {

namespace {
template <std::uint16_t (*Rule)(std::uint16_t, int) noexcept>
consteval std::array<std::uint16_t, 2 * 65536> make_node_next() {
  std::array<std::uint16_t, 2 * 65536> t{};
  for (int y = 0; y < 2; ++y)
    for (int v = 0; v < 65536; ++v)
      t[static_cast<std::size_t>(y << 16 | v)] = Rule(static_cast<std::uint16_t>(v), y);
  return t;
}
}  // namespace

alignas(64) constinit const std::array<std::uint16_t, 2 * 65536> kNodeNext = make_node_next<node_next>();
alignas(64) constinit const std::array<std::uint16_t, 2 * 65536> kNodeNext2 = make_node_next<node_next2>();

}  // namespace hce::ncm
