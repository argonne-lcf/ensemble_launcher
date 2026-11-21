"""`comm.blob.gather` copies selected records the same way on both of its paths."""

import random

import numpy as np
import pytest

from ensemble_launcher.comm import blob

pytestmark = pytest.mark.core


def _column(lengths, rng):
    recs = [bytes(rng.getrandbits(8) for _ in range(n)) for n in lengths]
    return recs, b"".join(recs), blob.offsets_from_lengths(np.array(lengths, dtype=np.int64))


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("record_bytes", [0, 3, 40, 300])  # short runs: byte index; long: join
def test_gather_matches_slicing(seed, record_bytes):
    rng = random.Random(seed)
    n = rng.randint(1, 200)
    recs, data, off = _column([rng.randint(0, record_bytes) for _ in range(n)], rng)
    rows = rng.choice([
        list(range(n)),                                   # one run
        sorted(rng.sample(range(n), rng.randint(0, n))),  # scattered, in order
        [rng.randrange(n) for _ in range(rng.randint(0, 2 * n))],  # any order, repeats
    ])  # fmt: skip
    for src in (data, memoryview(data)):
        got, got_off = blob.gather(src, off, np.array(rows, dtype=np.int64))
        assert got == b"".join(recs[r] for r in rows)
        assert [got[got_off[i] : got_off[i + 1]] for i in range(len(rows))] == [recs[r] for r in rows]


def test_both_paths_are_taken(monkeypatch):
    recs, data, off = _column([10] * 50, random.Random(0))
    rows = np.arange(0, 50, 2)
    want = b"".join(recs[r] for r in rows)
    for cutoff in (1, 10**9):  # join / byte index
        monkeypatch.setattr(blob, "GATHER_JOIN_MIN_RUN_BYTES", cutoff)
        assert blob.gather(data, off, rows)[0] == want
