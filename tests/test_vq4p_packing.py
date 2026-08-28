#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""The VQ4P index packing, written twice, checked against itself once.

`tools/convert.py` packs a real container's `index_bits 6` records;
`tools/make_test_container.py` packs the synthetic one `tests/run.sh` builds
so the VQ4P arm has anything at all to run on. The second says in its own
docstring that it is "byte-for-byte the packing tools/convert.py's
block_indices_packed writes". Nothing checked that (issue #56).

Reading the two is not enough, and the arm they feed cannot notice. It
compares the engine against itself — chunked against sequential, SIMD
against the CPU baseline, cache against no cache — and every one of those
runs the *same* unpack over the same bytes. A generator that packed
differently from the converter would produce a container the engine decodes
into some other set of indices, both backends would decode it the same
wrong way, and all four checks would pass on a layout no converter writes.

So the two are run against each other here, on random indices, for shapes
that exercise the padding as well as the exact case. torch is a dependency
of `convert.py` and never of the engine, so this runs under `uv` like the
other oracles and skips cleanly without it.

The constants are compared too, and first. `VEC_DIM` and `IDX_BLOCK` decide
the blocking rather than the bit packing, and a disagreement there would
move every byte while both files still looked internally consistent.

  uv run --with torch python -m unittest tests.test_vq4p_packing
"""

import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

import make_test_container as gen                            # noqa: E402

STAGES6 = 4          # index_bits 6 is four stages of 64 entries, and only that


def load_convert():
    """tools/convert.py, or None when torch is not installed."""
    try:
        import convert
    except ImportError:
        return None
    return convert


class TestPackingAgreesWithTheConverter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.convert = load_convert()
        if cls.convert is None:
            raise unittest.SkipTest(
                "torch not installed; convert.py cannot be imported "
                "(run under `uv run --with torch`)")
        import torch
        cls.torch = torch

    def test_the_blocking_constants_agree(self):
        """A different VEC_DIM or IDX_BLOCK moves every byte."""
        self.assertEqual(self.convert.VEC_DIM, gen.VEC_DIM)
        self.assertEqual(self.convert.IDX_BLOCK, gen.IDX_BLOCK)

    def test_packed_bytes_are_identical(self):
        torch = self.torch
        # (M, N): rows and columns. M = 64 is exactly one index block, 100
        # needs padding to two, 17 is a block mostly padding. N is a
        # multiple of VEC_DIM because a vector position is VEC_DIM wide.
        for M, N in ((64, 32), (100, 64), (17, 16), (128, 8)):
            with self.subTest(M=M, N=N):
                nvr = N // gen.VEC_DIM
                g = torch.Generator().manual_seed(M * 1000 + N)
                idx = torch.randint(0, 64, (STAGES6, M * nvr),
                                    dtype=torch.uint8, generator=g)

                want = self.convert.block_indices_packed(idx, M, N, STAGES6)
                want = bytes(want.reshape(-1).tolist())

                rows = [[int(v) for v in idx[s].tolist()]
                        for s in range(STAGES6)]
                got = gen.block_indices_packed6(rows, M, N)

                self.assertEqual(len(want), len(got),
                                 f"{M}x{N}: packed lengths differ")
                self.assertEqual(want, got,
                                 f"{M}x{N}: the two packings disagree")

    def test_a_changed_index_changes_the_bytes(self):
        """The comparison is not vacuously true on all-zero input."""
        rows = [[0] * 8 for _ in range(STAGES6)]
        base = gen.block_indices_packed6(rows, 8, 8)
        rows[2][3] = 63
        self.assertNotEqual(base, gen.block_indices_packed6(rows, 8, 8))


if __name__ == "__main__":
    unittest.main()
