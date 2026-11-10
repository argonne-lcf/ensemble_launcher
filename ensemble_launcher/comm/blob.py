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


def gather(blob, off: np.ndarray, rows: np.ndarray) -> Tuple[bytes, np.ndarray]:
    """The records at ``rows`` (in that order), as a new blob and offsets.

    Adjacent records are copied as one slice, so selecting a contiguous run of rows is a single
    copy regardless of how many rows it covers.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) == 0:
        return b"", EMPTY_OFFSETS.copy()
    starts = off[rows]
    ends = off[rows + 1]
    new_off = offsets_from_lengths(ends - starts)
    # Start a new slice wherever a record does not begin where the previous one ended.
    breaks = np.flatnonzero(starts[1:] != ends[:-1]) + 1
    run_starts = starts[np.concatenate(([0], breaks))]
    run_ends = ends[np.concatenate((breaks - 1, [len(rows) - 1]))]
    mv = memoryview(blob)
    return b"".join(mv[s:e] for s, e in zip(run_starts.tolist(), run_ends.tolist())), new_off


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
