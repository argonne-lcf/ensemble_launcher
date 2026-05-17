import asyncio
import os
import secrets
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Union

import cloudpickle
from typing_extensions import Unpack

from ensemble_launcher.comm.pipe import (
    AsyncTransport,
    AsyncZMQDealerConnection,
    ClientConnection,
    ServerConnection,
    ServerConnectionState,
    decode_identity,
    transport_registry,
)
from ensemble_launcher.ensemble import Task
from ensemble_launcher.ensemble.ensemble import TaskKwargs
from ensemble_launcher.logging import setup_logger


def action(fn: Callable):
    fn.__action_name__ = fn.__name__
    return fn


class AgentHandle:
    def __init__(
        self,
        conn: Union[ClientConnection, ServerConnection],
        actions: List[str],
        default_target_id: Optional[str] = None,
    ):
        self._conn = conn
        self._actions = set(actions)
        self._default_target_id = default_target_id

    def __getattr__(self, name):
        if name in self._actions:

            async def proxy(*args, target_id: Optional[str] = None):
                await self.send((name, *args), target_id=target_id)
                return await self.recv()

            return proxy
        elif hasattr(self._conn, name):
            return getattr(self._conn, name)
        else:
            raise AttributeError(f"No attribute named {name}")

    async def recv(self):
        frames = await self._conn.recv()
        return cloudpickle.loads(frames[1])

    async def send(self, msg: Any, target_id: Optional[str] = None):
        data = cloudpickle.dumps(msg)
        if isinstance(self._conn, ServerConnection):
            tid = target_id or self._default_target_id
            await self._conn.send(data, tid)
        else:
            await self._conn.send(data)
        return


class _ActorBase(ABC):
    def __init__(self, name: str):
        self._name = name
        self._secret = secrets.token_hex(16)
        self._conn: Union[ServerConnection, ClientConnection] = None
        self._stop: asyncio.Event = None
        self._input_queue: asyncio.Queue = None
        self._output_queue: asyncio.Queue = None
        self.logger = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__actions__ = {}
        for base in cls.__mro__:
            for attr in base.__dict__.values():
                if callable(attr) and hasattr(attr, "__action_name__"):
                    cls.__actions__.setdefault(attr.__action_name__, attr)

    def _init_runtime(self):
        self.logger = setup_logger(name=self._name, log_dir=f"{os.getcwd()}/logs")
        self._stop = asyncio.Event()
        self._input_queue = asyncio.Queue()
        self._output_queue = asyncio.Queue()

    @property
    def secret(self) -> str:
        return self._secret

    @property
    def name(self) -> str:
        return self._name

    def _extract_sender(self, frames) -> str:
        _, id, secret = decode_identity(frames[0])
        return f"{id}:{secret}"

    @abstractmethod
    def create_handle(self, *args, **kwargs): ...

    @abstractmethod
    async def _run(self): ...

    async def _recv(self):
        self.logger.info("Receive loop started.")
        while not self._stop.is_set():
            try:
                frames = await asyncio.wait_for(self._conn.recv(), timeout=5.0)
                sender_id = self._extract_sender(frames)
                self.logger.info(f"Received args from {sender_id}")
                args = cloudpickle.loads(frames[1])
                await self._input_queue.put((sender_id, args))
            except Exception:
                pass

    async def _send(self):
        self.logger.info("Send loop started.")
        while not self._stop.is_set():
            try:
                target_id, data = await asyncio.wait_for(
                    self._output_queue.get(), timeout=5.0
                )
                payload = cloudpickle.dumps(data)
                if isinstance(self._conn, ServerConnection):
                    await self._conn.send(payload, target_id)
                else:
                    await self._conn.send(payload)
                self.logger.info(f"Sent results to {target_id}")
            except Exception as e:
                self.logger.debug(f"Send failed with error: {str(e)}")

    @action
    def stop(self):
        self._stop.set()
        self.logger.info("Actor stop set")

    async def _invoke(self, action_name: str, *args: Any) -> Any:
        act = self.__actions__.get(action_name)
        result = act(self, *args)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    async def _main_loop(self):
        self.logger.info("Main loop started.")
        while not self._stop.is_set():
            target_id, args = await self._input_queue.get()
            if isinstance(args, list):
                results = []
                for arg in args:
                    try:
                        results.append(await self._invoke(*arg))
                    except Exception as e:
                        self.logger.error(f"Invoke failed with error: {e}")
                        raise e
                await self._output_queue.put((target_id, results))
            elif isinstance(args, tuple):
                try:
                    result = await self._invoke(*args)
                except Exception as e:
                    self.logger.error(f"Invoke failed with error: {e}")
                    raise e
                await self._output_queue.put((target_id, result))
            else:
                raise ValueError("Argments has to be either List[Tuple] or Tuple")

    def on_start(self):
        pass

    def on_stop(self):
        pass

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


class PublicActor(_ActorBase):
    def __init__(
        self,
        name: str,
        transport: str = "zmq",
        ckpt_dir: str = f"{os.getcwd()}/.actor_ckpt",
    ):
        super().__init__(name)
        self._ckpt_dir = ckpt_dir
        self._transport_classes = transport_registry.get(transport)
        self._transport: AsyncTransport = None
        self._transport_started = False
        self._handle_counter = 0

    def _make_validator(self) -> Callable:
        def validator(sender_id: str, sender_secret: str):
            if sender_secret == self.secret:
                return True
            return False

        return validator

    @property
    def ckpt_dir(self) -> str:
        return self._ckpt_dir

    def create_handle(
        self,
        timeout=300,
    ) -> Optional[AgentHandle]:
        fname = f"{self.ckpt_dir}/{self.name}.ckpt"
        start = time.time()
        while time.time() - start < timeout:
            if os.path.exists(fname):
                break
            time.sleep(1.0)

        if time.time() - start > timeout:
            return

        state_cls: ServerConnectionState = self._transport_classes[
            "server_connection_state"
        ]
        with open(fname, "r") as f:
            json_str = f.read()
        server_state = state_cls.deserialize(json_str)
        if server_state.transport_type == "zmq":
            handle_id = self._handle_counter
            self._handle_counter += 1
            clientconn = AsyncZMQDealerConnection(
                f"{self.name}-handle-{handle_id}",
                self.secret,
                remote_address=server_state.address,
            )
        else:
            raise NotImplementedError("Only zmq is implemented")

        return AgentHandle(
            clientconn,
            list(self.__actions__.keys()),
            default_target_id=f"{self.name}:{self.secret}",
        )

    def _start_transport(self):
        if self._transport_started:
            return
        self._transport = self._transport_classes["transport"]()
        self._server_id = self._name
        self._server_secret = secrets.token_hex(16)
        self._conn = self._transport.get_server_connection(
            self._server_id, self._server_secret, address=None
        )

        self._conn.set_unknown_sender_validator(self._make_validator())

        self._transport_started = True

    async def _run(self):
        self._init_runtime()

        result = self.on_start()

        self._start_transport()
        await self._conn.open()

        os.makedirs(self.ckpt_dir, exist_ok=True)
        fname = f"{self.ckpt_dir}/{self.name}.ckpt"
        with open(fname, "w") as f:
            f.write(self._conn.get_state().serialize())

        self.logger.info("Done opening the server!")

        if asyncio.iscoroutine(result):
            await result

        await asyncio.gather(self._recv(), self._send(), self._main_loop())

        result = self.on_stop()
        if asyncio.iscoroutine(result):
            await result


class PrivateActor(_ActorBase):
    def __init__(
        self,
        name: str,
        transport: str = "zmq",
    ):
        super().__init__(name)
        self._transport_classes = transport_registry.get(transport)
        self._transport: AsyncTransport = None
        self._server_conn: ServerConnection = None

    def _start_transport(self):
        if self._transport is not None:
            return
        transport = self._transport_classes["transport"]()
        self._transport = transport
        server, client = transport.create_child_pipe(
            parent_id=f"{self._name}-handle",
            parent_secret=self._secret,
            child_id=self._name,
            child_secret=self._secret,
        )
        self._conn = client
        self._server_conn = server

    def create_handle(self) -> AgentHandle:
        assert self._server_conn is not None, "call create_task first"
        target_id = f"{self._name}:{self._secret}"
        return AgentHandle(
            self._server_conn,
            list(self.__actions__.keys()),
            default_target_id=target_id,
        )

    async def _run(self):
        self._init_runtime()

        result = self.on_start()

        await self._conn.open()

        self.logger.info("Connected to server, ready!")

        if asyncio.iscoroutine(result):
            await result

        await asyncio.gather(self._recv(), self._send(), self._main_loop())

        result = self.on_stop()
        if asyncio.iscoroutine(result):
            await result

    def create_task(self, task_id, nnodes, ppn, **kwargs):
        task = super().create_task(task_id, nnodes, ppn, **kwargs)
        self._start_transport()
        return task


Actor = PublicActor


def actor(fn: Callable) -> PublicActor:
    class _FnActor(PublicActor):
        @action
        def call(self, *args: Any) -> Any:
            return fn(*args)

    _FnActor.__name__ = fn.__name__
    _FnActor.__qualname__ = fn.__qualname__
    return _FnActor(name=fn.__name__)
