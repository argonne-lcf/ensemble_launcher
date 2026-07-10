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


# ------------------------------------------------------------------ #
#  pack / unpack                                                       #
# ------------------------------------------------------------------ #


class TestPack:
    def test_pack_sets_packed_flag(self):
        t = _make_str_task()
        assert not t.is_packed
        t.pack()
        assert t.is_packed

    def test_pack_nulls_deep_fields(self):
        t = _make_str_task()
        t.pack()
        assert t.executable is None
        assert t.args is None
        assert t.kwargs is None
        assert t.env is None
        assert t.result is None

    def test_pack_preserves_shallow_fields(self):
        t = _make_str_task()
        t.pack()
        assert t.task_id == "t-str"
        assert t.nnodes == 2
        assert t.ppn == 4
        assert t.cpu_affinity == [0, 1, 2, 3]
        assert t.gpu_affinity == ["0", "1"]
        assert t.executor_name == "async_mpi"
        assert t.tag == "test-tag"
        assert t.run_dir == "/tmp/test"
        assert t.stdout_file == "out.log"
        assert t.stderr_file == "err.log"

    def test_pack_stores_raw_deep(self):
        t = _make_str_task()
        t.pack()
        assert t._raw_deep is not None
        assert isinstance(t._raw_deep, bytes)
        assert len(t._raw_deep) > 0

    def test_pack_is_idempotent(self):
        t = _make_str_task()
        t.pack()
        blob1 = t._raw_deep
        t.pack()
        assert t._raw_deep is blob1

    def test_pack_returns_self(self):
        t = _make_str_task()
        ret = t.pack()
        assert ret is t

    def test_pack_callable(self):
        t = _make_callable_task()
        t.pack()
        assert t.is_packed
        assert t.task_id == "t-callable"
        assert t.executable is None


class TestUnpack:
    def test_unpack_restores_str_executable(self):
        t = _make_str_task()
        t.pack()
        t.unpack()
        assert not t.is_packed
        assert t.executable == "echo hello"
        assert t.args == (1, 2)
        assert t.kwargs == {"x": 3}
        assert t.env == {"FOO": "bar"}

    def test_unpack_restores_callable_executable(self):
        t = _make_callable_task()
        t.pack()
        t.unpack()
        assert not t.is_packed
        assert callable(t.executable)
        assert t.executable(10) == 20
        assert t.args == (5,)
        assert t.kwargs == {"extra": "val"}

    def test_unpack_clears_raw_deep(self):
        t = _make_str_task()
        t.pack()
        t.unpack()
        assert t._raw_deep is None

    def test_unpack_noop_when_already_unpacked(self):
        t = _make_str_task()
        t.unpack()
        assert t.executable == "echo hello"

    def test_unpack_restores_result_field(self):
        t = _make_str_task(result={"accuracy": 0.95})
        t.pack()
        assert t.result is None
        t.unpack()
        assert t.result == {"accuracy": 0.95}

    def test_pack_unpack_roundtrip_preserves_all_fields(self):
        t = _make_str_task(
            status=TaskStatus.RUNNING,
            estimated_runtime=42.0,
            exception="something broke",
            result=b"raw bytes result",
            start_time=100.0,
            end_time=200.0,
        )
        t.pack()
        t.unpack()
        assert t.task_id == "t-str"
        assert t.nnodes == 2
        assert t.ppn == 4
        assert t.executable == "echo hello"
        assert t.args == (1, 2)
        assert t.kwargs == {"x": 3}
        assert t.env == {"FOO": "bar"}
        assert t.status == TaskStatus.RUNNING
        assert t.estimated_runtime == 42.0
        assert t.exception == "something broke"
        assert t.result == b"raw bytes result"
        assert t.cpu_affinity == [0, 1, 2, 3]
        assert t.gpu_affinity == ["0", "1"]
        assert t.run_dir == "/tmp/test"
        assert t.start_time == 100.0
        assert t.end_time == 200.0
        assert t.executor_name == "async_mpi"
        assert t.tag == "test-tag"
        assert t.stdout_file == "out.log"
        assert t.stderr_file == "err.log"


class TestCheckUnpacked:
    def test_check_raises_when_packed(self):
        t = _make_str_task()
        t.pack()
        with pytest.raises(RuntimeError, match="call unpack"):
            t._check_unpacked("executable")

    def test_check_passes_when_unpacked(self):
        t = _make_str_task()
        t._check_unpacked("executable")


# ------------------------------------------------------------------ #
#  to_bytes / from_bytes                                               #
# ------------------------------------------------------------------ #


class TestToFromBytes:
    def test_from_bytes_returns_packed(self):
        t = _make_str_task()
        wire = t.to_bytes()
        t2 = Task.from_bytes(wire)
        assert t2.is_packed

    def test_roundtrip_shallow_fields(self):
        t = _make_str_task(
            status=TaskStatus.READY,
            estimated_runtime=3.14,
            exception="err msg",
            ngpus_per_process=0.5,
            start_time=10.0,
            end_time=20.0,
        )
        wire = t.to_bytes()
        t2 = Task.from_bytes(wire)
        assert t2.task_id == "t-str"
        assert t2.nnodes == 2
        assert t2.ppn == 4
        assert t2.ngpus_per_process == 0.5
        assert t2.status == TaskStatus.READY
        assert t2.estimated_runtime == 3.14
        assert t2.exception == "err msg"
        assert t2.cpu_affinity == [0, 1, 2, 3]
        assert t2.gpu_affinity == ["0", "1"]
        assert t2.executor_name == "async_mpi"
        assert t2.tag == "test-tag"
        assert t2.run_dir == "/tmp/test"
        assert t2.start_time == 10.0
        assert t2.end_time == 20.0
        assert t2.stdout_file == "out.log"
        assert t2.stderr_file == "err.log"

    def test_roundtrip_deep_fields_str_executable(self):
        t = _make_str_task()
        wire = t.to_bytes()
        t2 = Task.from_bytes(wire)
        t2.unpack()
        assert t2.executable == "echo hello"
        assert t2.args == (1, 2)
        assert t2.kwargs == {"x": 3}
        assert t2.env == {"FOO": "bar"}

    def test_roundtrip_deep_fields_callable_executable(self):
        t = _make_callable_task()
        wire = t.to_bytes()
        t2 = Task.from_bytes(wire)
        t2.unpack()
        assert callable(t2.executable)
        assert t2.executable(7) == 14
        assert t2.args == (5,)

    def test_to_bytes_packs_unpacked_task(self):
        t = _make_str_task()
        assert not t.is_packed
        _ = t.to_bytes()
        assert t.is_packed

    def test_to_bytes_works_on_already_packed_task(self):
        t = _make_str_task()
        t.pack()
        wire = t.to_bytes()
        t2 = Task.from_bytes(wire)
        t2.unpack()
        assert t2.executable == "echo hello"

    def test_roundtrip_with_none_optional_fields(self):
        t = Task(
            task_id="minimal",
            nnodes=1,
            ppn=1,
            executable="ls",
        )
        wire = t.to_bytes()
        t2 = Task.from_bytes(wire)
        assert t2.task_id == "minimal"
        assert t2.run_dir is None
        assert t2.executor_name is None
        assert t2.tag is None
        assert t2.exception is None
        assert t2.start_time is None
        assert t2.end_time is None

    def test_roundtrip_with_large_payload(self):
        large_result = os.urandom(1_000_000)
        t = _make_str_task(result=large_result)
        wire = t.to_bytes()
        t2 = Task.from_bytes(wire)
        assert t2.is_packed
        t2.unpack()
        assert t2.result == large_result

    def test_roundtrip_with_complex_args(self):
        import numpy as np

        arr = np.arange(100)
        t = _make_callable_task(args=(arr, [1, 2, 3]), kwargs={"matrix": arr.reshape(10, 10)})
        wire = t.to_bytes()
        t2 = Task.from_bytes(wire)
        t2.unpack()
        assert (t2.args[0] == arr).all()
        assert t2.args[1] == [1, 2, 3]
        assert (t2.kwargs["matrix"] == arr.reshape(10, 10)).all()

    def test_multiple_roundtrips(self):
        t = _make_str_task()
        for _ in range(5):
            wire = t.to_bytes()
            t = Task.from_bytes(wire)
            t.unpack()
        assert t.executable == "echo hello"
        assert t.task_id == "t-str"


# ------------------------------------------------------------------ #
#  Edge cases                                                          #
# ------------------------------------------------------------------ #


class TestEdgeCases:
    def test_empty_collections(self):
        t = Task(
            task_id="empty",
            nnodes=1,
            ppn=1,
            executable="ls",
            args=(),
            kwargs={},
            env={},
            cpu_affinity=[],
            gpu_affinity=[],
        )
        wire = t.to_bytes()
        t2 = Task.from_bytes(wire)
        assert t2.cpu_affinity == []
        assert t2.gpu_affinity == []
        t2.unpack()
        assert t2.args == ()
        assert t2.kwargs == {}
        assert t2.env == {}

    def test_gpu_affinity_mixed_types(self):
        t = _make_str_task(gpu_affinity=[0, "1.0", 2, "3.1"])
        wire = t.to_bytes()
        t2 = Task.from_bytes(wire)
        assert t2.gpu_affinity == [0, "1.0", 2, "3.1"]

    def test_pack_unpack_multiple_cycles(self):
        t = _make_callable_task()
        for _ in range(10):
            t.pack()
            t.unpack()
        assert t.executable(3) == 6
        assert not t.is_packed


# ================================================================== #
#  Message classes — pack / unpack / to_bytes / from_bytes            #
# ================================================================== #

from ensemble_launcher.comm.messages import (
    IResultBatch,
    Message,
    NodeRequest,
    NodeUpdate,
    Ready,
    Result,
    ResultAck,
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
    return ResultBatch(sender="w0", receiver="main", data=results)


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
    return TaskUpdate(
        sender="main",
        receiver="w0",
        added_tasks=[t1],
        deleted_tasks=[t2],
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
#  Result                                                              #
# ------------------------------------------------------------------ #


class TestResultPackUnpack:
    def test_pack_nulls_data(self):
        r = _make_result()
        r.pack()
        assert r.is_packed
        assert r.data is None
        assert r.task_id == "task-42"
        assert r.success is True

    def test_unpack_restores_data(self):
        r = _make_result()
        r.pack()
        r.unpack()
        assert not r.is_packed
        assert r.data == {"accuracy": 0.95, "loss": 0.1}

    def test_pack_idempotent(self):
        r = _make_result()
        r.pack()
        blob1 = r._raw_deep
        r.pack()
        assert r._raw_deep is blob1

    def test_unpack_noop_when_unpacked(self):
        r = _make_result()
        r.unpack()
        assert r.data == {"accuracy": 0.95, "loss": 0.1}

    def test_pack_with_large_data(self):
        big = os.urandom(500_000)
        r = _make_result(data=big)
        r.pack()
        r.unpack()
        assert r.data == big

    def test_pack_with_none_data(self):
        r = _make_result(data=None)
        r.pack()
        r.unpack()
        assert r.data is None

    def test_pack_with_callable_data(self):
        r = _make_result(data=_my_fn)
        r.pack()
        r.unpack()
        assert r.data(7) == 14


class TestResultToFromBytes:
    def test_roundtrip_basic(self):
        r = _make_result()
        wire = r.to_bytes()
        r2 = Message.from_bytes(wire)
        assert isinstance(r2, Result)
        assert r2.is_packed
        assert r2.task_id == "task-42"
        assert r2.success is True
        r2.unpack()
        assert r2.data == {"accuracy": 0.95, "loss": 0.1}

    def test_to_bytes_packs_unpacked_result(self):
        r = _make_result()
        assert not r.is_packed
        _ = r.to_bytes()
        assert r.is_packed

    def test_roundtrip_with_exception(self):
        r = _make_result(success=False, exception="traceback here", data=None)
        wire = r.to_bytes()
        r2 = Message.from_bytes(wire)
        assert r2.success is False
        assert r2.exception == "traceback here"
        r2.unpack()
        assert r2.data is None

    def test_multiple_roundtrips(self):
        r = _make_result()
        for _ in range(5):
            wire = r.to_bytes()
            r = Message.from_bytes(wire)
            r.unpack()
        assert r.data == {"accuracy": 0.95, "loss": 0.1}
        assert r.task_id == "task-42"


# ------------------------------------------------------------------ #
#  ResultBatch                                                         #
# ------------------------------------------------------------------ #


class TestResultBatchPackUnpack:
    def test_pack_keeps_list_live(self):
        rb = _make_result_batch(3)
        rb.pack()
        assert rb.is_packed
        assert len(rb.data) == 3
        for r in rb.data:
            assert r.is_packed

    def test_unpack_restores_all_results(self):
        rb = _make_result_batch(3)
        rb.pack()
        rb.unpack()
        assert not rb.is_packed
        assert len(rb.data) == 3
        for i, r in enumerate(rb.data):
            assert not r.is_packed
            assert r.task_id == f"task-{i}"
            assert r.success is True
            assert r.data == {"val": i}

    def test_empty_batch(self):
        rb = ResultBatch(sender="w0", receiver="main", data=[])
        rb.pack()
        rb.unpack()
        assert rb.data == []


class TestResultBatchToFromBytes:
    def test_roundtrip(self):
        rb = _make_result_batch(4)
        wire = rb.to_bytes()
        rb2 = Message.from_bytes(wire)
        assert isinstance(rb2, ResultBatch)
        assert rb2.is_packed
        assert len(rb2.data) == 4
        assert rb2.data[2].is_packed
        assert rb2.data[2].task_id == "task-2"
        rb2.unpack()
        assert rb2.data[2].data == {"val": 2}


# ------------------------------------------------------------------ #
#  IResultBatch                                                        #
# ------------------------------------------------------------------ #


class TestIResultBatchPackUnpack:
    def test_pack_unpack_roundtrip(self):
        results = [_make_result(task_id=f"ir-{i}", data=i * 10) for i in range(3)]
        irb = IResultBatch(sender="w0", receiver="main", data=results)
        irb.pack()
        assert irb.is_packed
        irb.unpack()
        assert len(irb.data) == 3
        assert not irb.data[1].is_packed
        assert irb.data[1].task_id == "ir-1"
        assert irb.data[1].data == 10

    def test_to_from_bytes(self):
        results = [_make_result(task_id=f"ir-{i}", data={"x": i}) for i in range(2)]
        irb = IResultBatch(sender="w0", receiver="main", data=results)
        wire = irb.to_bytes()
        irb2 = Message.from_bytes(wire)
        assert isinstance(irb2, IResultBatch)
        assert irb2.is_packed
        assert irb2.data[0].is_packed
        assert irb2.data[0].task_id == "ir-0"
        irb2.unpack()
        assert irb2.data[0].data == {"x": 0}


# ------------------------------------------------------------------ #
#  TaskUpdate                                                          #
# ------------------------------------------------------------------ #


class TestTaskUpdatePackUnpack:
    def test_pack_keeps_lists_live(self):
        tu = _make_task_update()
        tu.pack()
        assert tu.is_packed
        assert len(tu.added_tasks) == 1
        assert len(tu.deleted_tasks) == 1
        for t in tu.added_tasks + tu.deleted_tasks:
            assert t.is_packed

    def test_unpack_restores_tasks(self):
        tu = _make_task_update()
        tu.pack()
        tu.unpack()
        assert not tu.is_packed
        assert len(tu.added_tasks) == 1
        assert len(tu.deleted_tasks) == 1
        assert not tu.added_tasks[0].is_packed
        assert tu.added_tasks[0].task_id == "added-1"
        assert tu.added_tasks[0].nnodes == 1
        assert tu.added_tasks[0].ppn == 2
        assert tu.added_tasks[0].executable == "echo added"
        assert not tu.deleted_tasks[0].is_packed
        assert tu.deleted_tasks[0].task_id == "deleted-1"
        assert callable(tu.deleted_tasks[0].executable)
        assert tu.deleted_tasks[0].executable(4) == 8


class TestTaskUpdateToFromBytes:
    def test_roundtrip(self):
        tu = _make_task_update()
        wire = tu.to_bytes()
        tu2 = Message.from_bytes(wire)
        assert isinstance(tu2, TaskUpdate)
        assert tu2.is_packed
        assert tu2.added_tasks[0].is_packed
        assert tu2.added_tasks[0].task_id == "added-1"
        tu2.unpack()
        assert tu2.added_tasks[0].executable == "echo added"
        assert tu2.deleted_tasks[0].executable(4) == 8


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

    def test_result_ack_roundtrip(self):
        ra = ResultAck(sender="main", receiver="w0")
        wire = ra.to_bytes()
        ra2 = Message.from_bytes(wire)
        assert isinstance(ra2, ResultAck)
        ra2.unpack()
        assert ra2.sender == "main"

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
            (IResultBatch(data=[_make_result()]), IResultBatch),
            (_make_task_update(), TaskUpdate),
            (Ready(sender="a"), Ready),
            (Stop(type=StopType.KILL), Stop),
            (ResultAck(), ResultAck),
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
