import asyncio
import multiprocessing as mp

import cloudpickle
import pytest

from ensemble_launcher.ensemble.actor import Actor, actor


def add(a, b):
    return a + b


def square(x):
    return x * x


class AddActor(Actor):
    def action(self, *args):
        return args[0] + args[1]


class LifecycleActor(Actor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.started = False
        self.stopped = False

    def on_start(self):
        self.started = True

    def on_stop(self):
        self.stopped = True

    def action(self, *args):
        return sum(args)


def test_actor_decorator():
    add_actor = actor(add)
    assert isinstance(add_actor, Actor)
    assert add_actor._name == "add"


def test_actor_subclass():
    a = AddActor(name="adder")
    assert isinstance(a, Actor)
    assert a._name == "adder"
    assert a.action(3, 4) == 7


def test_actor_create_task():
    a = AddActor(name="adder")
    a._start_transport()
    task = a.create_task(task_id="t0", nnodes=1, ppn=1)
    assert task.task_id == "t0"
    assert task.nnodes == 1
    assert task.ppn == 1
    assert task.executable is a


def test_actor_create_task_with_kwargs():
    a = AddActor(name="adder")
    a._start_transport()
    task = a.create_task(
        task_id="t1", nnodes=2, ppn=4, ngpus_per_process=1, tag="gpu-actor"
    )
    assert task.ngpus_per_process == 1
    assert task.tag == "gpu-actor"
    assert task.nnodes == 2
    assert task.ppn == 4


def test_actor_create_handle_before_transport():
    a = AddActor(name="adder")
    assert a.create_handle() is None


def test_actor_create_handle():
    a = AddActor(name="adder")
    a._start_transport()
    handle = a.create_handle()
    assert handle is not None


@pytest.mark.asyncio
async def test_actor_single_call():
    a = AddActor(name="actor-single")
    a._start_transport()
    handle = a.create_handle()

    await a._server.open()
    await handle.open()

    a._stop = asyncio.Event()
    a._input_queue = asyncio.Queue()
    a._output_queue = asyncio.Queue()

    recv_task = asyncio.create_task(a._recv())
    send_task = asyncio.create_task(a._send())
    main_task = asyncio.create_task(a._main_loop())

    await handle.send(cloudpickle.dumps((3, 4)))
    frames = await asyncio.wait_for(handle.recv(), timeout=5.0)
    result = cloudpickle.loads(frames[1])
    assert result == 7

    await handle.send(cloudpickle.dumps("stop"))
    await asyncio.wait_for(main_task, timeout=5.0)
    recv_task.cancel()
    send_task.cancel()

    await a._server.close()
    await handle.close()


@pytest.mark.asyncio
async def test_actor_batch_call():
    a = actor(square)
    a._start_transport()
    handle = a.create_handle()

    await a._server.open()
    await handle.open()

    a._stop = asyncio.Event()
    a._input_queue = asyncio.Queue()
    a._output_queue = asyncio.Queue()

    recv_task = asyncio.create_task(a._recv())
    send_task = asyncio.create_task(a._send())
    main_task = asyncio.create_task(a._main_loop())

    batch_args = [(2,), (3,), (5,)]
    await handle.send(cloudpickle.dumps(batch_args))
    frames = await asyncio.wait_for(handle.recv(), timeout=5.0)
    results = cloudpickle.loads(frames[1])
    assert results == [4, 9, 25]

    await handle.send(cloudpickle.dumps("stop"))
    await asyncio.wait_for(main_task, timeout=5.0)
    recv_task.cancel()
    send_task.cancel()

    await a._server.close()
    await handle.close()


@pytest.mark.asyncio
async def test_actor_multiple_calls():
    a = AddActor(name="actor-multi")
    a._start_transport()
    handle = a.create_handle()

    await a._server.open()
    await handle.open()

    a._stop = asyncio.Event()
    a._input_queue = asyncio.Queue()
    a._output_queue = asyncio.Queue()

    recv_task = asyncio.create_task(a._recv())
    send_task = asyncio.create_task(a._send())
    main_task = asyncio.create_task(a._main_loop())

    for x, y, expected in [(1, 2, 3), (10, 20, 30), (-1, 1, 0)]:
        await handle.send(cloudpickle.dumps((x, y)))
        frames = await asyncio.wait_for(handle.recv(), timeout=5.0)
        result = cloudpickle.loads(frames[1])
        assert result == expected, f"add({x}, {y}) expected {expected}, got {result}"

    await handle.send(cloudpickle.dumps("stop"))
    await asyncio.wait_for(main_task, timeout=5.0)
    recv_task.cancel()
    send_task.cancel()

    await a._server.close()
    await handle.close()


@pytest.mark.asyncio
async def test_actor_lifecycle_hooks():
    a = LifecycleActor(name="lifecycle")
    a._start_transport()
    handle = a.create_handle()

    await a._server.open()
    await handle.open()

    a._stop = asyncio.Event()
    a._input_queue = asyncio.Queue()
    a._output_queue = asyncio.Queue()

    assert not a.started
    a.on_start()
    assert a.started

    recv_task = asyncio.create_task(a._recv())
    send_task = asyncio.create_task(a._send())
    main_task = asyncio.create_task(a._main_loop())

    await handle.send(cloudpickle.dumps((1, 2, 3)))
    frames = await asyncio.wait_for(handle.recv(), timeout=5.0)
    result = cloudpickle.loads(frames[1])
    assert result == 6

    await handle.send(cloudpickle.dumps("stop"))
    await asyncio.wait_for(main_task, timeout=5.0)
    recv_task.cancel()
    send_task.cancel()

    assert not a.stopped
    a.on_stop()
    assert a.stopped

    await a._server.close()
    await handle.close()
