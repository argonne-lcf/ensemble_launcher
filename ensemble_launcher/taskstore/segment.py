"""`SegmentTaskStore`: received TaskUpdates kept as they arrived, hot in memory up to a budget,
the rest offloaded to node-local files and read back through ``mmap``.

Per task the store holds one index entry (``uid -> seg << 32 | row``: bytes key, int value) and,
once set, one status entry (an enum singleton). Neither is a container, so the cyclic GC sees a
couple of dicts rather than objects per task. A segment's bytes are never decoded as a whole:
`TaskMeta`s and `Task`s are built one row at a time, on demand.
"""

import asyncio
import logging
import mmap
import os
import shutil
import struct
import tempfile
from itertools import count
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np

from ensemble_launcher.ensemble import Task, TaskMeta, TaskStatus

from .base import TaskStore, register_task_store

_ROW_BITS = 32
_ROW_MASK = (1 << _ROW_BITS) - 1

# Spill file: five u64 lengths, then the columns. The two offset columns come first so they sit
# 8-byte aligned right after the 40-byte header.
_FILE_ORDER = ("_meta_off", "_deep_off", "_uids", "_meta", "_deep")
_HEADER = struct.Struct("!5Q")


class _Segment:
    __slots__ = ("update", "nbytes", "delivered", "released", "undelivered", "remaining",
                 "hot", "spilling", "unspillable", "path", "mm")  # fmt: skip

    def __init__(self, update):
        n = update.ntasks
        self.update = update
        self.nbytes = update.nbytes
        self.delivered = np.zeros(n, dtype=bool)
        self.released = np.zeros(n, dtype=bool)
        self.undelivered = n
        self.remaining = n
        self.hot = True
        self.spilling = False
        self.unspillable = False
        self.path: Optional[str] = None
        self.mm: Optional[mmap.mmap] = None


@register_task_store("segment")
class SegmentTaskStore(TaskStore):
    def __init__(
        self,
        node_id: str = "node",
        hot_bytes: int = 1 << 30,
        spill_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.node_id = node_id
        self.hot_bytes_limit = hot_bytes
        self._spill_root = spill_dir or tempfile.gettempdir()
        self._dir: Optional[str] = None
        self.logger = logger or logging.getLogger(__name__)
        self._segments: Dict[int, _Segment] = {}
        self._seg_ids = count()
        self._index: Dict[bytes, int] = {}
        self._status: Dict[bytes, TaskStatus] = {}
        self.hot_bytes = 0
        self._pending_spills: set = set()

    # ------------------------------------------------------------------ #
    #  Mapping                                                           #
    # ------------------------------------------------------------------ #

    def _locate(self, uid: bytes):
        loc = self._index[uid]
        return self._segments[loc >> _ROW_BITS], loc & _ROW_MASK

    def __getitem__(self, uid: bytes) -> TaskMeta:
        seg, row = self._locate(uid)
        return seg.update.meta(row)

    def __contains__(self, uid) -> bool:
        return uid in self._index

    def __iter__(self) -> Iterator[bytes]:
        return iter(list(self._index))

    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------ #
    #  TaskStore                                                         #
    # ------------------------------------------------------------------ #

    def add(self, update) -> List[bytes]:
        uids = update.uids()
        fresh = [row for row, uid in enumerate(uids) if uid not in self._index]
        if not fresh:
            return []
        if len(fresh) < len(uids):  # keep only the rows we don't have yet
            update = update.select(fresh)
            uids = [uids[r] for r in fresh]
        seg_no = next(self._seg_ids)
        self._segments[seg_no] = _Segment(update)
        base = seg_no << _ROW_BITS
        index = self._index
        for row, uid in enumerate(uids):
            index[uid] = base | row
        self.hot_bytes += self._segments[seg_no].nbytes
        self._maybe_spill()
        return uids

    def task(self, uid: bytes) -> Task:
        seg, row = self._locate(uid)
        return seg.update.task(row)

    def build_update(self, uids: Iterable[bytes], **msg_fields):
        from ensemble_launcher.comm.messages import TaskUpdate

        # Consecutive uids from the same segment become one select, so order is kept exactly
        # and a queue filled from one incoming update is a single gather.
        runs: List[tuple] = []
        for uid in uids:
            loc = self._index[uid]
            seg_no, row = loc >> _ROW_BITS, loc & _ROW_MASK
            if runs and runs[-1][0] == seg_no:
                runs[-1][1].append(row)
            else:
                runs.append((seg_no, [row]))
        parts = [self._segments[s].update.select(rows) for s, rows in runs]
        return TaskUpdate.concat(parts, **msg_fields)

    def get_status(self, uid, default=None):
        return self._status.get(uid, default)

    def set_status(self, uids: Iterable[bytes], status: TaskStatus) -> None:
        st = self._status
        for uid in uids:
            st[uid] = status

    def mark_delivered(self, uids: Iterable[bytes]) -> None:
        for seg_no, rows in self._rows_by_segment(uids).items():
            seg = self._segments[seg_no]
            new = rows[~seg.delivered[rows]]
            seg.delivered[new] = True
            seg.undelivered -= len(new)
        self._maybe_spill()

    def release(self, uids: Iterable[bytes]) -> None:
        # A released task leaves the index at once (so ``uid in store`` means "still to run",
        # which recovery relies on); its segment's bytes go when the last of its tasks does.
        uids = list(uids)
        by_seg = self._rows_by_segment(uids)
        for uid in uids:
            self._index.pop(uid, None)
        for seg_no, rows in by_seg.items():
            seg = self._segments[seg_no]
            new = rows[~seg.released[rows]]
            seg.released[new] = True
            seg.remaining -= len(new)
            if seg.remaining == 0:
                self._drop(seg_no)

    def close(self) -> None:
        for seg_no in list(self._segments):
            self._drop(seg_no, delete_now=True)
        self._index.clear()
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None

    # ------------------------------------------------------------------ #
    #  Internals                                                         #
    # ------------------------------------------------------------------ #

    def _rows_by_segment(self, uids: Iterable[bytes]) -> Dict[int, np.ndarray]:
        out: Dict[int, List[int]] = {}
        index = self._index
        for uid in uids:
            loc = index.get(uid)
            if loc is not None:
                out.setdefault(loc >> _ROW_BITS, []).append(loc & _ROW_MASK)
        return {s: np.asarray(rows, dtype=np.int64) for s, rows in out.items()}

    def _drop(self, seg_no: int, delete_now: bool = False) -> None:
        seg = self._segments.pop(seg_no)
        for uid in seg.update.uids():
            self._index.pop(uid, None)
        if seg.hot:
            self.hot_bytes -= seg.nbytes
        seg.update = None
        self._unmap(seg, delete_now)

    def _unmap(self, seg: _Segment, delete_now: bool) -> None:
        if seg.mm is not None:
            try:
                seg.mm.close()
            except BufferError:
                pass  # a view is still alive somewhere; the mapping goes when it does
            seg.mm = None
        if seg.path is not None and not seg.spilling:
            self._delete_file(seg.path, delete_now)
            seg.path = None

    def _delete_file(self, path: str, now: bool) -> None:
        loop = _running_loop()
        if now or loop is None:
            _remove(path)
        else:
            loop.run_in_executor(None, _remove, path)

    # --- spilling ------------------------------------------------------ #

    def _pick_victim(self) -> Optional[int]:
        """Oldest hot segment already fully delivered, else the newest hot one."""
        candidates = [
            (seg_no, seg)
            for seg_no, seg in self._segments.items()
            if seg.hot and not seg.spilling and not seg.unspillable
        ]
        for seg_no, seg in candidates:  # dicts keep insertion order: oldest first
            if seg.undelivered == 0:
                return seg_no
        return candidates[-1][0] if candidates else None

    def _maybe_spill(self) -> None:
        in_flight = sum(s.nbytes for s in self._segments.values() if s.spilling)
        while self.hot_bytes - in_flight > self.hot_bytes_limit:
            seg_no = self._pick_victim()
            if seg_no is None:
                return
            seg = self._segments[seg_no]
            in_flight += seg.nbytes
            self._spill(seg_no, seg)

    def _spill(self, seg_no: int, seg: _Segment) -> None:
        if self._dir is None:
            try:
                self._dir = tempfile.mkdtemp(prefix=f"el-taskstore-{self.node_id}-", dir=self._spill_root)
            except OSError as e:
                self.logger.warning(f"Task store: cannot create spill dir in {self._spill_root}: {e}")
                seg.unspillable = True
                return
        seg.spilling = True
        seg.path = os.path.join(self._dir, f"seg-{seg_no}.bin")
        frames = [_as_bytes(getattr(seg.update, name)) for name in _FILE_ORDER]
        loop = _running_loop()
        if loop is None:
            self._spill_done(seg_no, seg, _write_and_map, (seg.path, frames))
            return
        fut = loop.run_in_executor(None, _write_and_map, seg.path, frames)
        self._pending_spills.add(fut)
        fut.add_done_callback(lambda f: self._spill_finished(seg_no, seg, f))

    def _spill_done(self, seg_no, seg, fn, args) -> None:
        try:
            mm = fn(*args)
        except OSError as e:
            self._spill_failed(seg, e)
            return
        self._adopt_mapping(seg_no, seg, mm)

    def _spill_finished(self, seg_no: int, seg: _Segment, fut) -> None:
        self._pending_spills.discard(fut)
        if fut.cancelled():
            seg.spilling = False
            return
        exc = fut.exception()
        if exc is not None:
            self._spill_failed(seg, exc)
            return
        self._adopt_mapping(seg_no, seg, fut.result())

    def _spill_failed(self, seg: _Segment, exc: BaseException) -> None:
        self.logger.warning(
            f"Task store: could not offload a segment to {seg.path} ({exc}); keeping it in memory"
        )
        seg.spilling = False
        seg.unspillable = True
        if seg.path:
            _remove(seg.path)
            seg.path = None

    def _adopt_mapping(self, seg_no: int, seg: _Segment, mm: mmap.mmap) -> None:
        seg.spilling = False
        if self._segments.get(seg_no) is not seg:  # released while the write was running
            seg.mm = mm
            self._unmap(seg, delete_now=False)
            return
        seg.mm = mm
        seg.update = _mapped_update(mm)
        seg.hot = False
        self.hot_bytes -= seg.nbytes


# --------------------------------------------------------------------------- #
#  File I/O (runs on an executor thread)                                     #
# --------------------------------------------------------------------------- #


def _running_loop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _as_bytes(col) -> bytes:
    return col.tobytes() if isinstance(col, np.ndarray) else col


def _write_and_map(path: str, frames: list) -> mmap.mmap:
    with open(path, "wb") as f:
        f.write(_HEADER.pack(*(len(fr) for fr in frames)))
        for fr in frames:
            f.write(fr)
    with open(path, "rb") as f:
        return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)


def _mapped_update(mm: mmap.mmap):
    """A TaskUpdate whose columns are views into a spill file's mapping."""
    from ensemble_launcher.comm.messages import TaskUpdate

    lengths = _HEADER.unpack_from(mm, 0)
    mv = memoryview(mm)
    cols, pos = {}, _HEADER.size
    for name, n in zip(_FILE_ORDER, lengths):
        if name.endswith("_off"):
            cols[name] = np.frombuffer(mm, dtype=np.int64, count=n // 8, offset=pos)
        else:
            cols[name] = mv[pos : pos + n]
        pos += n
    update = TaskUpdate()
    for name, col in cols.items():
        setattr(update, name, col)
    return update


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
