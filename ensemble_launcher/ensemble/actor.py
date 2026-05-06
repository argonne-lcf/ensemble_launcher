import asyncio
import os
import secrets
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

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

class Actor(ABC):
    def __init__(
        self,
        name: str,
        transport: str = "zmq",
        ckpt_dir: str = f"{os.getcwd()}/.actor_ckpt",
    ):
        self._name = name
        self._ckpt_dir = ckpt_dir
        self._transport_classes = transport_registry.get(transport)
        self._transport: AsyncTransport = None
        self._server: ServerConnection = None
        self._server_id: str = None
        self._server_secret: str = None
        self._transport_started = False
        self._stop: asyncio.Event = None
        self._input_queue: asyncio.Queue = None
        self._output_queue: asyncio.Queue = None
        self.logger = None
        self._secret = secrets.token_hex(16)
    
    def _make_validator(self, actor_secret: str) -> Callable:
        def validator(sender_id:str, sender_secret: str):
            if sender_secret == actor_secret:
                return True
            return False
        return validator
    
    @property
    def secret(self) -> str:
        return self._secret
    
    @property
    def ckpt_dir(self) -> str:
        return self._ckpt_dir

    @property
    def name(self) -> str:
        return self._name

    @property
    def transport_classes(self) -> Dict[str, type]:
        return self._transport_classes

    @classmethod
    def create_handle(
        cls, ckpt_dir: str, name: str, transport_classes: Dict[str, type], secret: str, timeout=300
    ) -> Optional[ClientConnection]:
        fname = f"{ckpt_dir}/{name}.ckpt"
        start = time.time()
        while time.time() - start < timeout:
            if os.path.exists(fname):
                break
            time.sleep(1.0)

        if time.time() - start > timeout:
            return

        state_cls: ServerConnectionState = transport_classes["server_connection_state"]
        with open(fname, "r") as f:
            json_str = f.read()
        server_state = state_cls.deserialize(json_str)
        if server_state.transport_type == "zmq":
            clientconn = AsyncZMQDealerConnection(
                f"{str(uuid.uuid4())}",
                secret,
                remote_address=server_state.address,
            )
        else:
            raise NotImplementedError("Only zmq is implemented")

        return clientconn

    def _start_transport(self):
        if self._transport_started:
            return
        self._transport = self._transport_classes["transport"]()
        self._server_id = self._name
        self._server_secret = secrets.token_hex(16)
        self._server = self._transport.get_server_connection(
            self._server_id, self._server_secret, address=None
        )

        ## Set a validator
        self._server.set_unknown_sender_validator(self._make_validator(actor_secret=self._secret))

        self._transport_started = True

    async def _recv(self):
        if not self._transport_started:
            raise RuntimeError("Transport not started")

        self.logger.info("Receive loop started.")
        while not self._stop.is_set():
            try:
                frames = await asyncio.wait_for(self._server.recv(), timeout=5.0)
                _, id, secret = decode_identity(frames[0])
                args = cloudpickle.loads(frames[1])
                await self._input_queue.put((f"{id}:{secret}", args))
            except Exception as e:
                self.logger.warning(f"Recv failed with error: {str(e)}")

    async def _send(self):
        if not self._transport_started:
            raise RuntimeError("Transport not started")

        self.logger.info("Send loop started.")
        while not self._stop.is_set():
            try:
                target_id, data = await asyncio.wait_for(
                    self._output_queue.get(), timeout=5.0
                )
                await self._server.send(cloudpickle.dumps(data), target_id)
            except Exception as e:
                self.logger.warning(f"Send failed with error: {str(e)}")


    async def _invoke(self, *args: Any) -> Any:
        result = self.action(*args)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    async def _main_loop(self):
        self.logger.info("Main loop started.")
        while not self._stop.is_set():
            target_id, args = await self._input_queue.get()
            if args == "stop":
                self._stop.set()
                self.logger.info(f"Actor stop set")
            elif isinstance(args, list):
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

    @abstractmethod
    def action(self, *args: Any) -> Any: ...

    def on_start(self):
        pass

    def on_stop(self):
        pass

    async def _run(self):
        self.logger = setup_logger(name=self._name, log_dir=f"{os.getcwd()}/logs")
        self._stop = asyncio.Event()
        self._input_queue = asyncio.Queue()
        self._output_queue = asyncio.Queue()

        result = self.on_start()

        self._start_transport()
        await self._server.open()

        ##write the checkpoint
        os.makedirs(self._ckpt_dir, exist_ok=True)
        fname = f"{self._ckpt_dir}/{self._name}.ckpt"
        with open(fname, "w") as f:
            f.write(self._server.get_state().serialize())

        self.logger.info("Done opening the server!")

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
