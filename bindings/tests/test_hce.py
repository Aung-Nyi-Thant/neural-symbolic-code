"""Tests for the hce Python bindings.

Run through ctest, or directly:  PYTHONPATH=build/python python -m unittest discover -s bindings/tests
"""

import random
import unittest

import hce


def sample_text(n: int, seed: int) -> bytes:
    rng = random.Random(seed)
    words = [b"alpha", b"beta", b"gamma", b"delta", b"stream", b"frame", b"model", b"bit"]
    out = bytearray()
    while len(out) < n:
        out += rng.choice(words) + (b". " if rng.randrange(10) == 0 else b" ")
    return bytes(out[:n])


class RoundTrip(unittest.TestCase):
    def test_every_profile(self):
        data = sample_text(50_000, 1)
        for profile in hce.profiles():
            self.assertEqual(hce.decompress(hce.compress(data, profile["id"])), data)
        self.assertLess(len(hce.compress(data, 1)), len(data) // 3)

    def test_empty_and_tiny_inputs(self):
        for data in (b"", b"x", b"abc", bytes(range(256))):
            self.assertEqual(hce.decompress(hce.compress(data, 1)), data)

    def test_buffer_types(self):
        data = sample_text(5_000, 2)
        for view in (bytearray(data), memoryview(data)):
            self.assertEqual(hce.decompress(hce.compress(view, 1)), data)

    def test_encoder_and_decoder_traces_match(self):
        data = sample_text(30_000, 3)
        blob, enc = hce.compress(data, 1, stats=True)
        out, dec = hce.decompress(blob, stats=True)
        self.assertEqual(out, data)
        self.assertEqual(enc["trace"], dec["trace"])
        self.assertEqual(enc["decisions"], 8 * len(data))

    def test_deterministic_output(self):
        data = sample_text(40_000, 4)
        self.assertEqual(hce.compress(data, 2), hce.compress(data, 2))

    def test_incompressible_input_is_stored(self):
        data = random.Random(5).randbytes(100_000)
        blob, stats = hce.compress(data, 1, stats=True)
        self.assertEqual(stats["frames_modeled"], 0)
        self.assertLessEqual(len(blob), len(data) + 28 + 2 * 13)
        self.assertEqual(hce.decompress(blob), data)

    def test_hybrid_copy_and_trace(self):
        block = random.Random(81).randbytes(65_536)
        data = block * 4
        blob, enc = hce.compress(data, 3, stats=True)
        out, dec = hce.decompress(blob, stats=True)
        self.assertEqual(out, data)
        self.assertEqual(enc, dec)
        self.assertGreater(enc["copied_bytes"], 150_000)
        self.assertGreater(enc["copies"], 0)
        self.assertLess(len(blob), len(data) // 2)


class Errors(unittest.TestCase):
    def test_corruption_raises_format_error(self):
        blob = bytearray(hce.compress(sample_text(10_000, 6), 1))
        blob[len(blob) // 2] ^= 0x10
        with self.assertRaises(hce.FormatError):
            hce.decompress(bytes(blob))
        self.assertTrue(issubclass(hce.FormatError, ValueError))

    def test_truncation_raises(self):
        blob = hce.compress(sample_text(10_000, 7), 1)
        with self.assertRaises(hce.FormatError):
            hce.decompress(blob[:-1])

    def test_unknown_profile(self):
        with self.assertRaises(ValueError):
            hce.compress(b"data", 99)

    def test_output_limit(self):
        data = sample_text(10_000, 8)
        blob = hce.compress(data, 1)
        with self.assertRaises(hce.FormatError):
            hce.decompress(blob, max_output=len(data) - 1)
        self.assertEqual(hce.decompress(blob, max_output=len(data)), data)


class Introspection(unittest.TestCase):
    def test_profiles_and_model_ids(self):
        profiles = hce.profiles()
        ids = {p["id"] for p in profiles}
        self.assertIn(hce.DEFAULT_PROFILE, ids)
        model_ids = [hce.model_id(p["id"]) for p in profiles]
        self.assertEqual(model_ids, [p["model_id"] for p in profiles])
        self.assertEqual(len(set(model_ids)), len(model_ids))

    def test_crc32c(self):
        self.assertEqual(hce.crc32c(b"123456789"), 0xE3069283)
        self.assertEqual(hce.crc32c(b"456789", hce.crc32c(b"123")), 0xE3069283)


if __name__ == "__main__":
    unittest.main()
