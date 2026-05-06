import secrets
from abc import ABC, abstractmethod
import asyncio
import cloudpickle
from typing import Any, Callable, Optional
from typing_extensions import Unpack
import uuid

from ensemble_launcher.comm.pipe import (
    AsyncTransport,
    ClientConnection,
    ServerConnection,
    transport_registry,
    decode_identity
)

from ensemble_launcher.ensemble import Task
from ensemble_launcher.ensemble.ensemble import TaskKwargs


class Actor(ABC):
    def __init__(self, name: str = "actor", transport: str = "zmq"):
        self._name = name
        self._transport_classes = transport_registry.get(transport)
        self._transport: AsyncTransport = None
        self._server: ServerConnection = None
        self._server_id: str = None
        self._server_secret: str = None
        self._transport_started = False
        self._stop: asyncio.Event = None
        self._input_queue: asyncio.Queue = None
        self._output_queue: asyncio.Queue = None

    def create_handle(self) -> Optional[ClientConnection]:
        if not self._transport_started:
            return None

        _, clientconn = self._transport.create_child_pipe(self._server_id, self._server_secret, f"{str(uuid.uuid4())}", secrets.token_hex(16))
        return clientconn

    def _start_transport(self):
        if self._transport_started:
            return
        self._transport = self._transport_classes["transport"]()
        self._server_id = self._name
        self._server_secret = secrets.token_hex(16)
        self._server = self._transport.get_server_connection(
            self._server_id, self._server_secret
        )
        self._transport_started = True

    async def _recv(self):
        if not self._transport_started:
            raise RuntimeError("Transport not started")

        while not self._stop.is_set():
            try:
                frames = asyncio.wait_for(await self._server.recv(),timeout=5.0)
                _, id, secret = decode_identity(frames[0])
                args = cloudpickle.loads(frames[1])
                await self._input_queue.put((f"{id}:{secret}",args))
            except TimeoutError:
                await asyncio.sleep(0.5)

    async def _send(self):
        if not self._transport_started:
            raise RuntimeError("Transport not started")

        while not self._stop.is_set():
            try:
                target_id, data = asyncio.wait_for(await self._output_queue.get(),timeout=5.0)
                await self._server.send(cloudpickle.dumps(data), target_id)
            except TimeoutError:
                await asyncio.sleep(0.5)


    async def _invoke(self, *args: Any) -> Any:
        result = self.action(*args)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    async def _main_loop(self):
        while not self._stop.is_set():
            target_id, args = await self._input_queue.get()
            if args == "stop":
                self._stop.set()
            elif isinstance(args, list):
                results = []
                for arg in args:
                    results.append(await self._invoke(*arg))
                await self._output_queue.put((target_id, results))
            elif isinstance(args, tuple):
                result = await self._invoke(*args)
                await self._output_queue.put((target_id, result))

    @abstractmethod
    def action(self, *args: Any) -> Any:
        ...

    def on_start(self):
        pass

    def on_stop(self):
        pass

    async def _run(self):
        self._stop = asyncio.Event()
        self._input_queue = asyncio.Queue()
        self._output_queue = asyncio.Queue()
        self._start_transport()
        await self._server.open()

        result = self.on_start()
        if asyncio.iscoroutine(result):
            await result

        await asyncio.gather(self._recv(), self._send(), self._main_loop())

        result = self.on_stop()
        if asyncio.iscoroutine(result):
            await result

    def __call__(self):
        asyncio.run(self._run())

    def create_task(
        self,
        task_id: str,
        nnodes: int,
        ppn: int,
        **kwargs: Unpack[TaskKwargs],
    ) -> Task:
        return Task(
            task_id=task_id,
            nnodes=nnodes,
            ppn=ppn,
            executable=self,
            **kwargs,
        )


def actor(fn: Callable) -> Actor:
    class _FnActor(Actor):
        def action(self, *args: Any) -> Any:
            return fn(*args)
    _FnActor.__name__ = fn.__name__
    _FnActor.__qualname__ = fn.__qualname__
    return _FnActor(name=fn.__name__)
