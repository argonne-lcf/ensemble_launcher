"""Columns of variable-length records: one blob plus an ``int64[n+1]`` offsets array.

The task and result messages keep every task's (or result's) bytes in a few such columns
instead of one Python object per item, so a node can route, split and join thousands of them
without creating objects the cyclic GC would have to walk. Everything here works on bytes,
memoryviews and numpy arrays -- none of them GC-tracked.

Record ``i`` of a column is ``blob[off[i]:off[i+1]]``. ``blob`` may be ``bytes`` or any
buffer (a received zmq frame, an ``mmap`` of a spilled segment).
"""

from typing import Iterable, List, Sequence, Tuple

import numpy as np

UID_BYTES = 16

EMPTY_OFFSETS = np.zeros(1, dtype=np.int64)


def offsets_from_lengths(lengths: Sequence[int]) -> np.ndarray:
    off = np.zeros(len(lengths) + 1, dtype=np.int64)
    if len(lengths):
        np.cumsum(lengths, out=off[1:])
    return off


def encode_column(records: Iterable[bytes]) -> Tuple[bytes, np.ndarray]:
    """Join records into one blob and the offsets that split it again."""
    records = list(records)
    return b"".join(records), offsets_from_lengths([len(r) for r in records])


def offsets_from_frame(frame) -> np.ndarray:
    """The offsets array of a received frame, as a zero-copy read-only view."""
    return np.frombuffer(frame, dtype=np.int64)


def record(blob, off: np.ndarray, i: int) -> memoryview:
    return memoryview(blob)[int(off[i]) : int(off[i + 1])]

# Average run length (bytes) from which `gather` copies run by run instead of byte by byte. A run
# costs ~70-100 ns as a slice in ``b"".join``; the byte-indexed copy costs ~1-1.5 ns per byte plus
# its index arrays. They cross at ~48-64 B per run (measured for 100-5000 runs).
GATHER_JOIN_MIN_RUN_BYTES = 64


def gather(blob, off: np.ndarray, rows: np.ndarray) -> Tuple[bytes, np.ndarray]:
    """The records at ``rows`` (in that order), as a new blob and offsets.

    Adjacent records are copied as one run. Long runs (e.g. task payloads) are joined as
    memoryview slices, one memcpy each; many short runs (e.g. small results) are copied
    through a byte index in numpy, which has no per-run Python cost.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) == 0:
        return b"", EMPTY_OFFSETS.copy()
    starts = off[rows]
    ends = off[rows + 1]
    new_off = offsets_from_lengths(ends - starts)
    breaks = np.flatnonzero(starts[1:] != ends[:-1]) + 1
    run_starts = starts[np.concatenate(([0], breaks))]
    run_ends = ends[np.concatenate((breaks - 1, [len(rows) - 1]))]
    total = int(new_off[-1])

    if total >= GATHER_JOIN_MIN_RUN_BYTES * len(run_starts):
        mv = memoryview(blob)
        return b"".join([mv[s:e] for s, e in zip(run_starts.tolist(), run_ends.tolist())]), new_off

    lengths = run_ends - run_starts
    run_out = np.empty(len(lengths), dtype=np.int64)
    run_out[0] = 0
    np.cumsum(lengths[:-1], out=run_out[1:])
    idx = np.arange(total, dtype=np.int64) + np.repeat(run_starts - run_out, lengths)
    return np.frombuffer(blob, dtype=np.uint8)[idx].tobytes(), new_off


def concat(columns: List[Tuple[object, np.ndarray]]) -> Tuple[bytes, np.ndarray]:
    """Join several ``(blob, off)`` columns into one."""
    if not columns:
        return b"", EMPTY_OFFSETS.copy()
    parts, offs, base = [], [np.zeros(1, dtype=np.int64)], 0
    for blob, off in columns:
        lo, hi = int(off[0]), int(off[-1])
        parts.append(memoryview(blob)[lo:hi])
        offs.append(off[1:] - lo + base)
        base += hi - lo
    return b"".join(parts), np.concatenate(offs)


# --------------------------------------------------------------------------- #
#  Fixed-width columns (uids, flags)                                          #
# --------------------------------------------------------------------------- #


def split_fixed(col, width: int) -> List[bytes]:
    """Every ``width``-byte item of a fixed-width column, as ``bytes``."""
    b = bytes(col)
    return [b[i : i + width] for i in range(0, len(b), width)]


def gather_fixed(col, width: int, rows: np.ndarray) -> bytes:
    if len(rows) == 0:
        return b""
    arr = np.frombuffer(col, dtype=np.uint8).reshape(-1, width)
    return arr[np.asarray(rows, dtype=np.int64)].tobytes()
