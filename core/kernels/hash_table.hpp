// Bucketed hash table of bit-history slots.
#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "core/kernels/tables.hpp"

namespace hce {

// One 16-byte slot per (context, nibble): byte 0 is a check byte, bytes 1..15 hold the bit-history
// states of the nibble's 15-node binary tree (node 1 = root). Four slots form a 64-byte,
// cache-line-aligned bucket, so a lookup touches exactly one cache line.
class HashTable {
public:
    static constexpr uint32_t kSlotBytes = 16;

    // 2^slot_bits slots; slot_bits must be in [3, 30] (checked by CmModel).
    explicit HashTable(uint32_t slot_bits)
        : buckets_(size_t{1} << (slot_bits - 2)), bucket_bits_(slot_bits - 2) {}

    // Slot index for context hash h. On a miss, the bucket's slot whose root node has seen the
    // fewest bits (ties: lowest index) is cleared and claimed. Determined by table contents only.
    uint32_t find(uint64_t h, const Tables& t) noexcept {
        const auto bucket = static_cast<uint32_t>(h >> (64 - bucket_bits_));
        const auto check = static_cast<uint8_t>(h);
        uint8_t* b = buckets_[bucket].bytes.data();
        uint32_t victim = 0;
        uint32_t victim_priority = UINT32_MAX;
        for (uint32_t i = 0; i < 4; ++i) {
            const uint8_t* s = b + i * kSlotBytes;
            if (s[0] == check) return bucket * 4 + i;
            const uint32_t priority = uint32_t{t.n0[s[1]]} + t.n1[s[1]];
            if (priority < victim_priority) {
                victim_priority = priority;
                victim = i;
            }
        }
        uint8_t* s = b + victim * kSlotBytes;
        std::memset(s, 0, kSlotBytes);
        s[0] = check;
        return bucket * 4 + victim;
    }

    uint8_t* slot(uint32_t index) noexcept {
        return buckets_[index >> 2].bytes.data() + (index & 3u) * kSlotBytes;
    }

private:
    struct alignas(64) Bucket {
        std::array<uint8_t, 64> bytes;
    };
    std::vector<Bucket> buckets_;  // value-initialized: every slot starts as the empty history
    uint32_t bucket_bits_;
};

}  // namespace hce
