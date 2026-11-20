"""`SegmentTaskStore`: received TaskUpdates kept as they arrived, hot in memory up to a budget,
the rest offloaded to node-local files and read back through ``mmap``.

A task's handle is ``seg << 32 | row``: the segment its update became and its row there, so a
handle finds its bytes with no lookup. Per task the store holds one inverse-map entry
(``uid -> handle``: bytes key, int value) until the task is released, plus its row in three
per-segment columns: ``assignment`` (int32), ``client`` (int32) and ``delivered`` (bool). Those
live on the segment, not in the (possibly memory-mapped, read-only) update. The store is not a
container of per-task objects, so the cyclic GC sees a few dicts and arrays. A segment's bytes
are never decoded as a whole: `TaskMeta`s and `Task`s are built one row at a time, on demand.
"""

import asyncio
import logging
import mmap
import os
import shutil
import struct
import tempfile
from itertools import count, repeat
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cloudpickle
import numpy as np

from ensemble_launcher.ensemble import Task, TaskMeta
from ensemble_launcher.ensemble.ensemble import META_DTYPE, decode_meta_array

from .base import NO_CLIENT, RELEASED, UNASSIGNED, TaskStore, register_task_store

_ROW_BITS = 32
_ROW_MASK = (1 << _ROW_BITS) - 1
_UID_BYTES = 16
_EMPTY_IDX = np.empty(0, dtype=np.int64)

# Spill file: four u64 lengths, then the columns. The offset column comes first so it sits
# 8-byte aligned right after the 32-byte header.
_FILE_ORDER = ("_deep_off", "_uids", "_meta", "_deep")
_HEADER = struct.Struct("!4Q")

# Below this many uids, `lookup` uses a list comprehension: numpy's fixed cost per call (~2.5 us)
# outweighs its per-uid saving until ~50-60 uids. Workers look up one task at a time.
_SMALL_BATCH = 48


def _as_idx(idx) -> np.ndarray:
    return idx if isinstance(idx, np.ndarray) and idx.dtype == np.int64 else np.asarray(idx, dtype=np.int64).reshape(-1)


def _group_locs(locs: np.ndarray):
    """Split handles (``seg << _ROW_BITS | row``) by segment: ``([(seg_no, rows)], order)``.
    Rows keep their order within a segment. ``order`` is the stable argsort by segment
    (positions in ``locs``, group after group), or None when there is at most one segment, so
    no reordering happened."""
    if len(locs) == 0:
        return [], None
    segs = locs >> _ROW_BITS
    rows = locs & _ROW_MASK
    if (segs == segs[0]).all():
        return [(int(segs[0]), rows)], None
    order = np.argsort(segs, kind="stable")
    bounds = np.flatnonzero(np.diff(segs[order])) + 1
    return [(int(segs[g[0]]), rows[g]) for g in np.split(order, bounds)], order


class _Segment:
    __slots__ = ("update", "nbytes", "delivered", "assignment", "client", "undelivered",
                 "remaining", "hot", "spilling", "unspillable", "path", "mm")  # fmt: skip

    def __init__(self, update, client: int):
        n = update.ntasks
        self.update = update
        self.nbytes = update.nbytes
        self.delivered = np.zeros(n, dtype=bool)
        self.assignment = np.full(n, UNASSIGNED, dtype=np.int32)  # RELEASED once finished
        self.client = np.full(n, client, dtype=np.int32)
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
        self._index: Dict[bytes, int] = {}  # uid -> handle, live tasks only
        self._counts: Dict[int, int] = {}  # live tasks per assignment tag
        self._client_names: List[str] = []
        self._client_codes: Dict[str, int] = {}
        self.hot_bytes = 0
        self._pending_spills: set = set()

    # ------------------------------------------------------------------ #
    #  Handles                                                           #
    # ------------------------------------------------------------------ #

    def _live(self, h) -> Tuple[_Segment, int]:
        """The segment and row of a live handle; KeyError otherwise."""
        h = int(h)
        seg = self._segments.get(h >> _ROW_BITS)
        row = h & _ROW_MASK
        if seg is None or row >= len(seg.assignment) or seg.assignment[row] == RELEASED:
            raise KeyError(h)
        return seg, row

    def _by_segment(self, idx: np.ndarray):
        """``([(seg_no, rows)], inv)`` for working on ``idx`` one segment at a time, whatever
        their order. Concatenating per-segment results in list order and indexing with ``inv``
        restores the order of ``idx``; ``inv`` is None when no reordering is needed."""
        groups, order = _group_locs(idx)
        if order is None:
            return groups, None
        inv = np.empty(len(idx), dtype=np.int64)
        inv[order] = np.arange(len(idx), dtype=np.int64)
        return groups, inv

    def _gather(self, idx, column: str, missing) -> np.ndarray:
        """A per-segment column (``assignment`` / ``client``) at ``idx``; ``missing`` for
        handles whose segment is gone."""
        idx = _as_idx(idx)
        if len(idx) == 1:  # the worker's case
            h = int(idx[0])
            seg = self._segments.get(h >> _ROW_BITS)
            v = missing if seg is None else getattr(seg, column)[h & _ROW_MASK]
            return np.array([v], dtype=np.int32)
        groups, inv = self._by_segment(idx)
        parts = []
        for s, rows in groups:
            seg = self._segments.get(s)
            parts.append(np.full(len(rows), missing, np.int32) if seg is None else getattr(seg, column)[rows])
        if not parts:
            return np.empty(0, dtype=np.int32)
        out = parts[0] if len(parts) == 1 else np.concatenate(parts)
        return out if inv is None else out[inv]

    def _count(self, tags: np.ndarray, sign: int) -> None:
        counts = self._counts
        if len(tags) == 1:
            t = int(tags[0])
            counts[t] = counts.get(t, 0) + sign
            return
        vals, ns = np.unique(tags, return_counts=True)
        for t, n in zip(vals.tolist(), ns.tolist()):
            counts[t] = counts.get(t, 0) + sign * n

    # ------------------------------------------------------------------ #
    #  Mapping                                                           #
    # ------------------------------------------------------------------ #

    def __getitem__(self, h) -> TaskMeta:
        seg, row = self._live(h)
        return seg.update.meta(row)

    def __contains__(self, h) -> bool:
        try:
            self._live(h)
        except (KeyError, TypeError, ValueError):
            return False
        return True

    def __iter__(self) -> Iterator[int]:
        return iter(self.all().tolist())

    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------ #
    #  Tasks in and out                                                  #
    # ------------------------------------------------------------------ #

    def _client_code(self, client_id: Optional[str]) -> int:
        if client_id is None:
            return NO_CLIENT
        code = self._client_codes.get(client_id)
        if code is None:
            code = self._client_codes[client_id] = len(self._client_names)
            self._client_names.append(client_id)
        return code

    def add(self, update, client_id: Optional[str] = None) -> np.ndarray:
        uids = update.uids()
        index = self._index
        fresh = [row for row, uid in enumerate(uids) if uid not in index]
        if not fresh:
            return _EMPTY_IDX.copy()
        if len(fresh) < len(uids):  # keep only the rows we don't have yet
            update = update.select(fresh)
            uids = [uids[r] for r in fresh]
        seg_no = next(self._seg_ids)
        seg = self._segments[seg_no] = _Segment(update, self._client_code(client_id))
        base = seg_no << _ROW_BITS
        for row, uid in enumerate(uids):
            index[uid] = base | row
        self._counts[UNASSIGNED] = self._counts.get(UNASSIGNED, 0) + len(uids)
        self.hot_bytes += seg.nbytes
        self._maybe_spill()
        return base + np.arange(len(uids), dtype=np.int64)

    def lookup(self, uids: Sequence[bytes]) -> np.ndarray:
        get = self._index.get
        if len(uids) < _SMALL_BATCH:
            return np.array([get(uid, -1) for uid in uids], dtype=np.int64)
        return np.fromiter(map(get, uids, repeat(-1)), dtype=np.int64, count=len(uids))

    def _release_one(self, h: int) -> None:
        seg_no, row = h >> _ROW_BITS, h & _ROW_MASK
        seg = self._segments.get(seg_no)
        if seg is None or row >= len(seg.assignment):
            return
        old = int(seg.assignment[row])
        if old == RELEASED:
            return
        seg.assignment[row] = RELEASED
        self._counts[old] -= 1
        self._index.pop(seg.update.uid_at(row), None)
        seg.remaining -= 1
        if seg.remaining == 0:
            self._drop(seg_no, forget=False)

    def release(self, idx, uids: Optional[Sequence[bytes]] = None) -> None:
        # A released task leaves the index at once (so ``h in store`` / `lookup` forget it);
        # its segment's bytes go when the last of its tasks does. Every task's uid is popped
        # from the index, and that pop is also what makes a repeat release nothing: only the
        # first pop of a uid returns its handle.
        pop = self._index.pop
        if uids is not None:  # the caller has them (results): no need to read them back
            idx = np.fromiter(map(pop, uids, repeat(-1)), dtype=np.int64, count=len(uids))
            idx = idx[idx >= 0]
        elif isinstance(idx, (tuple, list)) and len(idx) == 1:  # a worker: one task at a time
            self._release_one(int(idx[0]))
            return
        else:
            kept = []
            for s, rows in _group_locs(_as_idx(idx))[0]:
                seg = self._segments.get(s)
                if seg is None:
                    continue
                base, uid_at = s << _ROW_BITS, seg.update.uid_at
                kept.extend(base | row for row in rows.tolist()
                            if seg.assignment[row] != RELEASED and pop(uid_at(row), None) is not None)  # fmt: skip
            idx = np.asarray(kept, dtype=np.int64)
        # ``idx`` now holds live, distinct handles whose uids have left the index.
        for s, rows in _group_locs(idx)[0]:
            seg = self._segments[s]
            self._count(seg.assignment[rows], -1)
            seg.assignment[rows] = RELEASED
            seg.remaining -= len(rows)
            if seg.remaining == 0:
                self._drop(s, forget=False)  # its uids all left the index above or before

    def mark_delivered(self, idx) -> None:
        if isinstance(idx, (tuple, list)) and len(idx) == 1:  # a worker: one task at a time
            h = int(idx[0])
            seg = self._segments.get(h >> _ROW_BITS)
            row = h & _ROW_MASK
            if seg is not None and not seg.delivered[row]:
                seg.delivered[row] = True
                seg.undelivered -= 1
                if seg.undelivered == 0:
                    self._maybe_spill()
            return
        for s, rows in _group_locs(_as_idx(idx))[0]:
            seg = self._segments.get(s)
            if seg is None:
                continue
            new = rows[~seg.delivered[rows]]
            seg.delivered[new] = True
            seg.undelivered -= len(new)
        self._maybe_spill()

    def close(self) -> None:
        for seg_no in list(self._segments):
            self._drop(seg_no, delete_now=True)
        self._index.clear()
        self._counts.clear()
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None

    # ------------------------------------------------------------------ #
    #  Reading tasks                                                     #
    # ------------------------------------------------------------------ #

    def task(self, h) -> Task:
        seg, row = self._live(h)
        return seg.update.task(row)

    def all(self) -> np.ndarray:
        parts = []
        for s, seg in self._segments.items():
            rows = np.flatnonzero(seg.assignment != RELEASED)
            if len(rows):
                parts.append((s << _ROW_BITS) + rows.astype(np.int64))
        return np.concatenate(parts) if parts else _EMPTY_IDX.copy()

    def uids(self, idx) -> List[bytes]:
        out = []
        for h in _as_idx(idx).tolist():
            seg, row = self._live(h)
            out.append(seg.update.uid_at(row))
        return out

    def meta_columns(self, idx) -> np.ndarray:
        groups, inv = self._by_segment(_as_idx(idx))
        if not groups:
            return np.empty(0, dtype=META_DTYPE)
        parts = [self._segments[s].update.meta_columns(rows) for s, rows in groups]
        meta = parts[0] if len(parts) == 1 else np.concatenate(parts)
        return meta if inv is None else meta[inv]

    def build_update(self, idx, **msg_fields):
        from ensemble_launcher.comm.messages import TaskUpdate

        groups, inv = self._by_segment(_as_idx(idx))
        if len(groups) == 1:  # the common case: no concat, so the bytes are copied once
            ((s, rows),) = groups
            return self._segments[s].update.select(rows, **msg_fields)
        parts = [self._segments[s].update.select(rows) for s, rows in groups]
        update = TaskUpdate.concat(parts, **msg_fields)
        return update if inv is None else update.select(inv, **msg_fields)

    def build_failed_results(self, idx, sender: str, reason: str):
        from ensemble_launcher.comm import blob
        from ensemble_launcher.comm.messages import IResultBatch

        idx = _as_idx(idx)
        idx = idx[self.assigned(idx) != RELEASED]
        groups, _ = _group_locs(idx)
        uid_parts, task_ids, handles = [], [], []
        for s, rows in groups:  # results need no particular order: keep segment order
            update = self._segments[s].update
            uid_parts.append(blob.gather_fixed(update._uids, _UID_BYTES, rows))
            for raw in update.meta_columns(rows)["task_id"].tolist():
                task_ids.append(raw.decode("ascii"))
            handles.append((s << _ROW_BITS) + rows)
        batch = IResultBatch(sender=sender)
        batch._uids = b"".join(uid_parts)
        batch._success = bytes(len(task_ids))
        batch._payload, batch._payload_off = blob.encode_column(
            cloudpickle.dumps((tid, None, reason)) for tid in task_ids
        )
        return batch, (np.concatenate(handles) if handles else _EMPTY_IDX.copy())

    # ------------------------------------------------------------------ #
    #  Assignment                                                        #
    # ------------------------------------------------------------------ #

    def assigned(self, idx) -> np.ndarray:
        return self._gather(idx, "assignment", RELEASED)

    def assign(self, idx, tag: int) -> np.ndarray:
        idx = _as_idx(idx)
        if len(idx) == 0:
            return idx
        moved = idx[self.assigned(idx) == UNASSIGNED]
        if len(moved):
            for s, rows in _group_locs(moved)[0]:
                self._segments[s].assignment[rows] = tag
            self._counts[UNASSIGNED] -= len(moved)
            self._counts[tag] = self._counts.get(tag, 0) + len(moved)
        return moved

    def assigned_to(self, tag: int) -> np.ndarray:
        if not self._counts.get(tag):
            return _EMPTY_IDX.copy()
        parts = []
        for s, seg in self._segments.items():
            rows = np.flatnonzero(seg.assignment == tag)
            if len(rows):
                parts.append((s << _ROW_BITS) + rows.astype(np.int64))
        return np.concatenate(parts) if parts else _EMPTY_IDX.copy()

    def reassign(self, from_tag: int, to_tag: int) -> np.ndarray:
        idx = self.assigned_to(from_tag)
        if len(idx) and from_tag != to_tag:
            for s, rows in _group_locs(idx)[0]:
                self._segments[s].assignment[rows] = to_tag
            self._counts[from_tag] -= len(idx)
            self._counts[to_tag] = self._counts.get(to_tag, 0) + len(idx)
        return idx

    def count(self, tag: int) -> int:
        return self._counts.get(tag, 0)

    def groups(self) -> List[Tuple[int, np.ndarray]]:
        handles, tags = [], []
        for s, seg in self._segments.items():
            rows = np.flatnonzero(seg.assignment != RELEASED)
            if len(rows):
                handles.append((s << _ROW_BITS) + rows.astype(np.int64))
                tags.append(seg.assignment[rows])
        if not handles:
            return []
        handles, tags = np.concatenate(handles), np.concatenate(tags)
        order = np.argsort(tags, kind="stable")
        tags, handles = tags[order], handles[order]
        bounds = np.flatnonzero(np.diff(tags)) + 1
        return [(int(t[0]), h) for t, h in zip(np.split(tags, bounds), np.split(handles, bounds))]

    # ------------------------------------------------------------------ #
    #  Clients                                                           #
    # ------------------------------------------------------------------ #

    def clients(self, idx) -> np.ndarray:
        return self._gather(idx, "client", NO_CLIENT)

    def client_name(self, code: int) -> Optional[str]:
        return None if code < 0 else self._client_names[code]

    def has_clients(self) -> bool:
        return bool(self._client_names)

    # ------------------------------------------------------------------ #
    #  Internals                                                         #
    # ------------------------------------------------------------------ #

    def _drop(self, seg_no: int, delete_now: bool = False, forget: bool = True) -> None:
        """Drop a segment's bytes. ``forget=False`` when every one of its uids is already out
        of the index (the last of them was just released), to skip a pop per row."""
        seg = self._segments.pop(seg_no)
        if forget:
            pop = self._index.pop
            for uid in seg.update.uids():
                pop(uid, None)
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
        elif name == "_meta":
            cols[name] = decode_meta_array(mm, offset=pos, count=n // META_DTYPE.itemsize)
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
