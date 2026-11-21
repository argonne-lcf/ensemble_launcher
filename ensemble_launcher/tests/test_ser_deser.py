import os

import pytest

from ensemble_launcher.ensemble import Task, TaskStatus

pytestmark = pytest.mark.core


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #


def _make_str_task(**overrides):
    defaults = dict(
        task_id="t-str",
        nnodes=2,
        ppn=4,
        executable="echo hello",
        args=(1, 2),
        kwargs={"x": 3},
        env={"FOO": "bar"},
        cpu_affinity=[0, 1, 2, 3],
        gpu_affinity=["0", "1"],
        executor_name="async_mpi",
        tag="test-tag",
        run_dir="/tmp/test",
        stdout_file="out.log",
        stderr_file="err.log",
    )
    defaults.update(overrides)
    return Task(**defaults)


def _my_fn(x):
    return x * 2


def _make_callable_task(**overrides):
    defaults = dict(
        task_id="t-callable",
        nnodes=1,
        ppn=1,
        executable=_my_fn,
        args=(5,),
        kwargs={"extra": "val"},
        env={"GPU": "0"},
    )
    defaults.update(overrides)
    return Task(**defaults)


# ================================================================== #
#  Message classes — to_bytes / from_bytes                          #
# ================================================================== #

from ensemble_launcher.comm.messages import (
    IResultBatch,
    Message,
    NodeRequest,
    NodeUpdate,
    Ready,
    Result,
    ResultBatch,
    Status,
    Stop,
    StopType,
    TaskRequest,
    TaskUpdate,
    _MSG_REGISTRY,
)


def _make_result(**overrides):
    defaults = dict(
        sender="w0",
        receiver="main",
        data={"accuracy": 0.95, "loss": 0.1},
        task_id="task-42",
        success=True,
        exception=None,
    )
    defaults.update(overrides)
    return Result(**defaults)


def _make_result_batch(n=3):
    results = [
        _make_result(task_id=f"task-{i}", data={"val": i}) for i in range(n)
    ]
    return ResultBatch.from_results(results, sender="w0", receiver="main")


def _make_task_update():
    t1 = Task(
        task_id="added-1",
        nnodes=1,
        ppn=2,
        executable="echo added",
        args=(1,),
        kwargs={"k": "v"},
    )
    t2 = Task(
        task_id="deleted-1",
        nnodes=1,
        ppn=1,
        executable=_my_fn,
        args=(10,),
    )
    return TaskUpdate.from_tasks(
        [t1], sender="main", receiver="w0", deleted_uids=[t2.uid]
    )


# ------------------------------------------------------------------ #
#  Message base (metadata-only) — pack/unpack are no-ops              #
# ------------------------------------------------------------------ #


class TestMessageBase:
    def test_pack_unpack_noop(self):
        m = Message(sender="a", receiver="b")
        m.pack()
        assert m.is_packed
        assert m.sender == "a"
        m.unpack()
        assert not m.is_packed
        assert m.sender == "a"

    def test_to_bytes_from_bytes_roundtrip(self):
        m = Message(sender="a", receiver="b", message_id="m1")
        wire = m.to_bytes()
        m2 = Message.from_bytes(wire)
        assert isinstance(m2, Message)
        m2.unpack()
        assert m2.sender == "a"
        assert m2.receiver == "b"
        assert m2.message_id == "m1"


# ------------------------------------------------------------------ #
#  TaskUpdate: blob-backed task encoding                               #
# ------------------------------------------------------------------ #

from ensemble_launcher.comm.messages import TaskUpdate as _TU  # noqa: E402
from ensemble_launcher.ensemble import TaskMeta  # noqa: E402
from ensemble_launcher.ensemble.ensemble import META_FIELDS  # noqa: E402


def _wire(msg):
    """A message as the receiver sees it: encoded to frames and decoded again."""
    from ensemble_launcher.comm.messages import Message

    return Message.from_byte_array(msg.to_byte_array())


def _all_fields(t):
    return {f: getattr(t, f) for f in Task.model_fields}


class TestTaskUid:
    def test_uid_is_16_random_bytes(self):
        a, b = _make_str_task(), _make_str_task()
        assert isinstance(a.uid, bytes) and len(a.uid) == 16
        assert a.uid != b.uid  # same task_id, different tasks

    def test_uid_not_in_meta_record(self):
        assert "uid" not in META_FIELDS
        assert not {"executable", "args", "kwargs", "env", "result"} & set(META_FIELDS)


class TestTaskUpdateRoundtrip:
    @pytest.mark.parametrize("make", [_make_str_task, _make_callable_task])
    def test_every_field_survives(self, make):
        t = make(status=TaskStatus.READY, estimated_runtime=2.5, driver_only=True)
        u = _wire(_TU.from_tasks([t], sender="main"))
        assert u.sender == "main" and u.ntasks == 1
        back = u.task(0)
        got, want = _all_fields(back), _all_fields(t)
        if callable(want["executable"]):
            assert got.pop("executable")(4) == want.pop("executable")(4)
        assert got == want

    def test_executable_by_reference(self):
        t = _make_callable_task(serialize_executable_by_value=False)
        back = _wire(_TU.from_tasks([t])).task(0)
        assert back.executable is _my_fn  # pickled by reference, resolves to the same function
        assert back.args == (5,) and back.kwargs == {"extra": "val"}

    def test_from_tasks_leaves_tasks_untouched(self):
        t = _make_callable_task()
        before = _all_fields(t)
        _TU.from_tasks([t]).to_byte_array()
        assert _all_fields(t) == before
        assert t.executable is _my_fn

    def test_meta_has_every_non_deep_field(self):
        t = _make_str_task()
        m = _wire(_TU.from_tasks([t])).meta(0)
        assert isinstance(m, TaskMeta)
        assert m.uid == t.uid
        for f in META_FIELDS:
            assert getattr(m, f) == getattr(t, f), f
        assert not hasattr(m, "executable")
        assert m.get_resource_requirements() == t.get_resource_requirements()

    def test_uids_read_from_column(self):
        tasks = [_make_str_task(task_id=f"t{i}") for i in range(5)]
        u = _wire(_TU.from_tasks(tasks))
        assert u.uids() == [t.uid for t in tasks]
        assert [u.uid_at(i) for i in range(5)] == [t.uid for t in tasks]

    def test_empty_update(self):
        u = _wire(_TU(sender="main"))
        assert u.ntasks == 0 and u.uids() == [] and u.to_tasks() == []

    def test_deleted_uids_travel(self):
        u = _wire(_TU(deleted_uids=[b"x" * 16]))
        assert u.deleted_uids == [b"x" * 16]

    def test_to_bytes_from_bytes(self):
        from ensemble_launcher.comm.messages import Message

        tasks = [_make_str_task(task_id=f"t{i}", args=(i,)) for i in range(3)]
        u = Message.from_bytes(_TU.from_tasks(tasks, message_id="m1").to_bytes())
        assert type(u) is _TU and u.message_id == "m1"
        assert [t.args for t in u.to_tasks()] == [(0,), (1,), (2,)]


class TestTaskUpdateSelectConcat:
    def _tasks(self, n=6):
        return [_make_callable_task(task_id=f"t{i}", args=(i,)) for i in range(n)]

    def test_select_keeps_given_order(self):
        tasks = self._tasks()
        u = _wire(_TU.from_tasks(tasks))
        s = _wire(u.select([4, 0, 1, 2], sender="m"))
        assert s.sender == "m"
        assert s.uids() == [tasks[i].uid for i in (4, 0, 1, 2)]
        assert [t.args for t in s.to_tasks()] == [(4,), (0,), (1,), (2,)]

    def test_select_contiguous_and_empty(self):
        tasks = self._tasks()
        u = _wire(_TU.from_tasks(tasks))
        assert [t.task_id for t in u.select([2, 3, 4]).to_tasks()] == ["t2", "t3", "t4"]
        assert u.select([]).ntasks == 0

    def test_concat_of_selections(self):
        tasks = self._tasks()
        u = _wire(_TU.from_tasks(tasks))
        c = _wire(_TU.concat([u.select([5, 3]), u.select([0]), _TU()]))
        assert c.uids() == [tasks[i].uid for i in (5, 3, 0)]
        assert [c.meta(i).task_id for i in range(3)] == ["t5", "t3", "t0"]
        assert c.task(2).executable(3) == 6


# ------------------------------------------------------------------ #
#  Result batches: blob-backed result encoding                         #
# ------------------------------------------------------------------ #


def _results(n=4):
    return [
        Result(uid=bytes([i]) * 16, task_id=f"task-{i}", data={"val": i},
               success=i % 2 == 0, exception=None if i % 2 == 0 else f"err {i}")
        for i in range(n)
    ]  # fmt: skip


class TestResultBatches:
    @pytest.mark.parametrize("cls", [ResultBatch, IResultBatch])
    def test_roundtrip(self, cls):
        rs = _results()
        b = _wire(cls.from_results(rs, sender="w0"))
        assert type(b) is cls and b.sender == "w0" and b.nresults == 4
        assert list(b.rows()) == [(r.uid, r.success) for r in rs]
        got = [(r.uid, r.task_id, r.data, r.success, r.exception) for r in b.data]
        assert got == [(r.uid, r.task_id, r.data, r.success, r.exception) for r in rs]

    def test_select_and_concat(self):
        rs = _results(5)
        b = _wire(IResultBatch.from_results(rs))
        c = _wire(ResultBatch.concat([b.select([3, 1]), IResultBatch.from_results([]), b.select([0])]))
        assert type(c) is ResultBatch
        assert [r.task_id for r in c.data] == ["task-3", "task-1", "task-0"]

    def test_to_dict_keyed_by_user_task_id(self):
        d = _wire(ResultBatch.from_results(_results(2))).to_dict()
        assert set(d) == {"task-0", "task-1"} and d["task-1"]["exception"] == "err 1"

    def test_large_numpy_payload(self):
        import numpy as np

        arr = np.arange(1_000_000, dtype=np.float64)
        b = _wire(ResultBatch.from_results([Result(uid=bytes(16), task_id="big", data=arr)]))
        assert np.array_equal(b.data[0].data, arr)

    def test_to_bytes_from_bytes(self):
        b = Message.from_bytes(IResultBatch.from_results(_results(3), message_id="r1").to_bytes())
        assert type(b) is IResultBatch and b.message_id == "r1" and b.nresults == 3

    def test_empty(self):
        b = _wire(ResultBatch.from_results([]))
        assert b.nresults == 0 and b.data == [] and list(b.rows()) == []


# ------------------------------------------------------------------ #
#  Metadata-only message types                                         #
# ------------------------------------------------------------------ #


class TestMetadataOnlyMessages:
    def test_status_roundtrip(self):
        s = Status(
            sender="w0",
            receiver="main",
            nrunning_tasks=5,
            nfailed_tasks=1,
            nsuccessful_tasks=10,
            nfree_cores=8,
            nfree_gpus=2,
            nremaining_tasks=20,
            task_throughput=3.14,
            tag="gpu-pool",
        )
        wire = s.to_bytes()
        s2 = Message.from_bytes(wire)
        assert isinstance(s2, Status)
        s2.unpack()
        assert s2.nrunning_tasks == 5
        assert s2.nfailed_tasks == 1
        assert s2.nsuccessful_tasks == 10
        assert s2.nfree_cores == 8
        assert s2.nfree_gpus == 2
        assert s2.nremaining_tasks == 20
        assert s2.task_throughput == 3.14
        assert s2.tag == "gpu-pool"

    def test_stop_roundtrip(self):
        s = Stop(type=StopType.TERMINATE)
        wire = s.to_bytes()
        s2 = Message.from_bytes(wire)
        assert isinstance(s2, Stop)
        s2.unpack()
        assert s2.type == StopType.TERMINATE

    def test_stop_kill(self):
        s = Stop(type=StopType.KILL)
        wire = s.to_bytes()
        s2 = Message.from_bytes(wire)
        s2.unpack()
        assert s2.type == StopType.KILL

    def test_ready_roundtrip(self):
        r = Ready(sender="w0", receiver="main")
        wire = r.to_bytes()
        r2 = Message.from_bytes(wire)
        assert isinstance(r2, Ready)
        r2.unpack()
        assert r2.sender == "w0"

    def test_task_request_roundtrip(self):
        tr = TaskRequest(sender="w0", receiver="main", ntasks=5)
        wire = tr.to_bytes()
        tr2 = Message.from_bytes(wire)
        assert isinstance(tr2, TaskRequest)
        tr2.unpack()
        assert tr2.ntasks == 5

    def test_node_request_roundtrip(self):
        nr = NodeRequest(sender="main", receiver="w0")
        wire = nr.to_bytes()
        nr2 = Message.from_bytes(wire)
        assert isinstance(nr2, NodeRequest)
        nr2.unpack()
        assert nr2.sender == "main"


# ------------------------------------------------------------------ #
#  Registry dispatch                                                   #
# ------------------------------------------------------------------ #


class TestFromBytesDispatch:
    def test_all_types_in_registry(self):
        from ensemble_launcher.comm.messages import all_messages
        for cls in all_messages:
            assert cls in _MSG_REGISTRY.values(), f"{cls.__name__} missing from registry"

    def test_dispatch_returns_correct_subclass(self):
        cases = [
            (Message(sender="a"), Message),
            (Status(sender="a", nrunning_tasks=1), Status),
            (_make_result(), Result),
            (_make_result_batch(1), ResultBatch),
            (IResultBatch.from_results([_make_result()]), IResultBatch),
            (_make_task_update(), TaskUpdate),
            (Ready(sender="a"), Ready),
            (Stop(type=StopType.KILL), Stop),
            (TaskRequest(ntasks=3), TaskRequest),
            (NodeRequest(), NodeRequest),
        ]
        for msg, expected_cls in cases:
            wire = msg.to_bytes()
            restored = Message.from_bytes(wire)
            assert type(restored) is expected_cls, (
                f"Expected {expected_cls.__name__}, got {type(restored).__name__}"
            )

    def test_unknown_type_id_raises(self):
        import struct
        bad_wire = struct.pack("!HI", 999, 0)
        with pytest.raises(ValueError, match="Unknown message type ID"):
            Message.from_bytes(bad_wire)
