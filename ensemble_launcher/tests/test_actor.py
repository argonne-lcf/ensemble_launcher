import asyncio
import os

import pytest

from ensemble_launcher.ensemble.actor import Actor, PrivateActor, action, actor

pytestmark = pytest.mark.core


def add(a, b):
    return a + b


def square(x):
    return x * x


class AddActor(Actor):
    @action
    def add(self, a, b):
        return a + b


class LifecycleActor(Actor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.started = False
        self.stopped = False

    def on_start(self):
        self.started = True

    def on_stop(self):
        self.stopped = True

    @action
    def sum(self, *args):
        return sum(args)


def test_actor_decorator():
    add_actor = actor(add)
    assert isinstance(add_actor, Actor)
    assert add_actor._name == "add"


def test_actor_subclass():
    a = AddActor(name="adder")
    assert isinstance(a, Actor)
    assert a._name == "adder"
    assert a.add(3, 4) == 7


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
    assert a.create_handle(timeout=1) is None


@pytest.mark.asyncio
async def test_actor_create_handle():
    a = AddActor(name="adder-handle")
    a._start_transport()
    await a._conn.open()

    os.makedirs(a._ckpt_dir, exist_ok=True)
    fname = f"{a._ckpt_dir}/{a._name}.ckpt"
    with open(fname, "w") as f:
        f.write(a._conn.get_state().serialize())

    handle = a.create_handle(timeout=1)
    assert handle is not None

    await a._conn.close()
    os.remove(fname)


async def _write_ckpt(a):
    os.makedirs(a._ckpt_dir, exist_ok=True)
    with open(f"{a._ckpt_dir}/{a._name}.ckpt", "w") as f:
        f.write(a._conn.get_state().serialize())


async def _cleanup_ckpt(a):
    fname = f"{a._ckpt_dir}/{a._name}.ckpt"
    if os.path.exists(fname):
        os.remove(fname)


@pytest.mark.asyncio
async def test_actor_single_call():
    a = AddActor(name="actor-single")
    a._start_transport()
    await a._conn.open()
    await _write_ckpt(a)
    handle = a.create_handle(timeout=1)

    await handle.open()

    a._init_runtime()

    recv_task = asyncio.create_task(a._recv())
    send_task = asyncio.create_task(a._send())
    main_task = asyncio.create_task(a._main_loop())

    result = await asyncio.wait_for(handle.add(3, 4), timeout=5.0)
    assert result == 7

    await handle.stop()
    await asyncio.wait_for(main_task, timeout=5.0)
    recv_task.cancel()
    send_task.cancel()

    await a._conn.close()
    await handle.close()
    await _cleanup_ckpt(a)


@pytest.mark.asyncio
async def test_actor_batch_call():
    a = actor(square)
    a._start_transport()
    await a._conn.open()
    await _write_ckpt(a)
    handle = a.create_handle(timeout=1)

    await handle.open()

    a._init_runtime()

    recv_task = asyncio.create_task(a._recv())
    send_task = asyncio.create_task(a._send())
    main_task = asyncio.create_task(a._main_loop())

    batch_args = [("call", (2,), None), ("call", (3,), None), ("call", (5,), None)]
    await handle.send(batch_args)
    results = await asyncio.wait_for(handle.recv(), timeout=10.0)
    assert results == [4, 9, 25]

    await handle.stop()
    await asyncio.wait_for(main_task, timeout=5.0)
    recv_task.cancel()
    send_task.cancel()

    await a._conn.close()
    await handle.close()
    await _cleanup_ckpt(a)


@pytest.mark.asyncio
async def test_actor_multiple_calls():
    a = AddActor(name="actor-multi")
    a._start_transport()
    await a._conn.open()
    await _write_ckpt(a)
    handle = a.create_handle(timeout=1)

    await handle.open()

    a._init_runtime()

    recv_task = asyncio.create_task(a._recv())
    send_task = asyncio.create_task(a._send())
    main_task = asyncio.create_task(a._main_loop())

    for x, y, expected in [(1, 2, 3), (10, 20, 30), (-1, 1, 0)]:
        result = await asyncio.wait_for(handle.add(x, y), timeout=5.0)
        assert result == expected, f"add({x}, {y}) expected {expected}, got {result}"

    await handle.stop()
    await asyncio.wait_for(main_task, timeout=5.0)
    recv_task.cancel()
    send_task.cancel()

    await a._conn.close()
    await handle.close()
    await _cleanup_ckpt(a)


@pytest.mark.asyncio
async def test_actor_lifecycle_hooks():
    a = LifecycleActor(name="lifecycle")
    a._start_transport()
    await a._conn.open()
    await _write_ckpt(a)
    handle = a.create_handle(timeout=1)

    await handle.open()

    a._init_runtime()

    assert not a.started
    a.on_start()
    assert a.started

    recv_task = asyncio.create_task(a._recv())
    send_task = asyncio.create_task(a._send())
    main_task = asyncio.create_task(a._main_loop())

    result = await asyncio.wait_for(handle.sum(1, 2, 3), timeout=5.0)
    assert result == 6

    await handle.stop()
    await asyncio.wait_for(main_task, timeout=5.0)
    recv_task.cancel()
    send_task.cancel()

    assert not a.stopped
    a.on_stop()
    assert a.stopped

    await a._conn.close()
    await handle.close()
    await _cleanup_ckpt(a)


class AddPrivateActor(PrivateActor):
    @action
    def add(self, a, b):
        return a + b


@pytest.mark.asyncio
async def test_private_actor_single_call():
    a = AddPrivateActor(name="priv-actor")
    task = a.create_task(task_id="t0", nnodes=1, ppn=1)
    handle = a.create_handle()

    a._init_runtime()
    await handle.open()
    await a._conn.open()
    await asyncio.sleep(0.1)

    recv_task = asyncio.create_task(a._recv())
    send_task = asyncio.create_task(a._send())
    main_task = asyncio.create_task(a._main_loop())

    result = await asyncio.wait_for(handle.add(10, 20), timeout=5.0)
    assert result == 30

    await handle.stop()
    await asyncio.wait_for(main_task, timeout=5.0)
    recv_task.cancel()
    send_task.cancel()

    await a._conn.close()
    await handle.close()


@pytest.mark.asyncio
async def test_private_actor_batch_call():
    a = AddPrivateActor(name="priv-batch")
    task = a.create_task(task_id="t0", nnodes=1, ppn=1)
    handle = a.create_handle()

    a._init_runtime()
    await handle.open()
    await a._conn.open()
    await asyncio.sleep(0.1)

    recv_task = asyncio.create_task(a._recv())
    send_task = asyncio.create_task(a._send())
    main_task = asyncio.create_task(a._main_loop())

    batch = [("add", (1, 2), None), ("add", (3, 4), None), ("add", (5, 6), None)]
    await handle.send(batch)
    results = await asyncio.wait_for(handle.recv(), timeout=5.0)
    assert results == [3, 7, 11]

    await handle.stop()
    await asyncio.wait_for(main_task, timeout=5.0)
    recv_task.cancel()
    send_task.cancel()

    await a._conn.close()
    await handle.close()


if __name__ == "__main__":
    from ensemble_launcher import EnsembleLauncher
    from ensemble_launcher.config import LauncherConfig, SystemConfig
    from ensemble_launcher.orchestrator import ClusterClient

    async def main():
        sys_config = SystemConfig(name="local")
        launcher_config = LauncherConfig(cluster=True, checkpoint_dir="./ckpt_dir")
        el = EnsembleLauncher(
            ensemble_file={}, system_config=sys_config, launcher_config=launcher_config
        )
        el.start()
        with ClusterClient(checkpoint_dir="./ckpt_dir") as client:
            actor = AddActor(name="add_actor")
            task = actor.create_task(task_id="actor_task")
            client.submit(task)
            handle = actor.create_handle()
            await handle.open()
            result = await handle.add(1, 2)
        return result

    asyncio.run(main())
