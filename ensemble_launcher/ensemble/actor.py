import asyncio
from functools import partial
import inspect
import os
import secrets
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

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

_READY_SENTINEL = b"__ACTOR_READY__"


def action(fn: Callable):
    fn.__action_name__ = fn.__name__
    return fn


class ActorHandle:
    def __init__(
        self,
        conn: Union[ClientConnection, ServerConnection],
        actions: Dict[str, inspect.Signature],
        default_target_id: Optional[str] = None,
    ):
        self._conn = conn
        self._actions = actions
        self._default_target_id = default_target_id
        self._cache: Dict[str, asyncio.Queue] = {}
        self._recv_task: asyncio.Task = None
        self._stop_event: asyncio.Event = None

    def __getattr__(self, name):
        _actions = self.__dict__.get("_actions")
        if _actions is not None:
            if name in _actions:
                sig = _actions[name]

                async def proxy(
                    *args,
                    target_id: Optional[str] = None,
                    **kwargs,
                ):
                    await self.send((name, args, kwargs), target_id=target_id)
                    return await self.recv(name)

                proxy.__name__ = name
                proxy.__qualname__ = name
                proxy.__signature__ = sig
                proxy.__annotations__ = {
                    k: v.annotation
                    for k, v in sig.parameters.items()
                    if v.annotation is not inspect.Parameter.empty
                }
                return proxy
        _conn = self.__dict__.get("_conn")
        if _conn is not None and hasattr(_conn, name):
            return getattr(_conn, name)
        raise AttributeError(f"No attribute named {name}")

    async def recv(self, action_name: str):
        return await self._cache[action_name].get()

    async def open(self):
        self.logger = setup_logger(name=self._conn.identity, subdir="handles")
        try:
            self._cache: Dict[str, asyncio.Queue] = {name: asyncio.Queue() for name in self._actions.keys()}    
            self._stop_event = asyncio.Event()
            self.logger.info("Wainting to open connection")
            await self._conn.open()
            self.logger.info("Opened connection")
            self._recv_task = asyncio.create_task(self._recv_loop())
            self.logger.info("Successfully opened the handle")
        except Exception as e:
            self.logger.error(f"Opening the handle failed with error {e}")
            raise e
    
    async def close(self):
        self._stop_event.set()
        await asyncio.sleep(1.0)

        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        await self._conn.close()

    async def stop(self, timeout: float = 10.0):
        data = cloudpickle.dumps(("stop", (), None))
        await self._conn.send(data)
        await asyncio.sleep(timeout)
        
    async def _recv_loop(self):
        try:
            while not self._stop_event.is_set():
                try:
                    frames = await self._conn.recv(timeout=5.0)
                    action_name, result = cloudpickle.loads(frames[1])
                    self._cache[action_name].put_nowait(result)
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass

    async def send(self, msg: Any, target_id: Optional[str] = None):
        data = cloudpickle.dumps(msg)
        if isinstance(self._conn, ServerConnection):
            tid = target_id or self._default_target_id
            await self._conn.send(data, tid)
        else:
            await self._conn.send(data)
        return


class PrivateActorHandle:
    def __init__(
        self,
        conn: ServerConnection,
        actions: Dict[str, inspect.Signature],
        flush_interval: float = 0.01,
        send_timeout: float = 5.0,
        send_retries: int = 3,
        default_target_id: Optional[str] = None,
    ):
        self._conn = conn
        self._actions = actions
        self._default_target_id = default_target_id
        self._ready_actors: set = set()
        self._ready_events: Dict[str, asyncio.Event] = {}
        self._ready_condition: asyncio.Condition = asyncio.Condition()
        self._cache: Dict[str, asyncio.Queue] = {}
        self._input_queue: asyncio.Queue = asyncio.Queue()
        self._recv_task = None
        self._send_task = None
        self._flush_interval = flush_interval
        self._send_timeout = send_timeout
        self._send_retries = send_retries

    def __getattr__(self, name):
        _actions = self.__dict__.get("_actions")
        if _actions is not None and name in _actions:
            sig = _actions[name]

            async def proxy(*args, actor_id: Optional[str] = None, **kwargs):
                tid = actor_id or self._default_target_id
                if tid is None:
                    raise ValueError(
                        f"actor_id required for {name}() (no default_target_id set)"
                    )
                await self.send((name, args, kwargs), target_id=tid)
                return await self.recv(name)

            params = list(sig.parameters.values())
            if len(params) > 0 and params[-1].kind == inspect.Parameter.VAR_KEYWORD:
                params.insert(
                    -1,
                    inspect.Parameter(
                        "actor_id",
                        inspect.Parameter.KEYWORD_ONLY,
                        default=None,
                        annotation=Optional[str],
                    ),
                )
            else:
                params.append(
                    inspect.Parameter(
                        "actor_id",
                        inspect.Parameter.KEYWORD_ONLY,
                        default=None,
                        annotation=Optional[str],
                    )
                )
            proxy.__name__ = name
            proxy.__qualname__ = name
            proxy.__signature__ = sig.replace(parameters=params)
            proxy.__annotations__ = {
                k: v.annotation
                for k, v in sig.parameters.items()
                if v.annotation is not inspect.Parameter.empty
            }
            return proxy

        raise AttributeError(f"No attribute named {name}")

    @property
    def ready_actors(self) -> Set:
        return self._ready_actors

    async def open(self):
        self.logger = setup_logger(name=self._conn.identity, subdir="handles")
        self._cache = {name: asyncio.Queue() for name in self._actions.keys()}
        await self._conn.open()
        self._recv_task = asyncio.create_task(self._recv_loop())
        self._send_task = asyncio.create_task(self._send_loop())

    async def close(self):
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self._send_task:
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass
        await self._conn.close()

    async def _recv_loop(self):
        try:
            while True:
                try:
                    frames = await self._conn.recv()
                    full_id, _, _ = decode_identity(frames[0])
                    if frames[1] == _READY_SENTINEL:
                        self._ready_actors.add(full_id)
                        event = self._ready_events.setdefault(full_id, asyncio.Event())
                        event.set()
                        async with self._ready_condition:
                            self._ready_condition.notify_all()
                    else:
                        action_name, result = cloudpickle.loads(frames[1])
                        self._cache[action_name].put_nowait(result)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.error(f"PrivateActorHandle recv error: {e}")
        except asyncio.CancelledError:
            pass

    async def _send_loop(self):
        try:
            while True:
                data, target_id = await self._input_queue.get()
                try:
                    success = False
                    for i in range(self._send_retries):
                        success = await self._conn.send(
                            data, target_id, timeout=self._send_timeout
                        )
                        if success:
                            break
                    if not success:
                        self.logger.warning(
                            f"Send failed to {target_id} after {self._send_retries} retries"
                        )
                except Exception as e:
                    self.logger.error(
                        f"PrivateActorHandle: failed to send to {target_id}: {e}"
                    )
                await asyncio.sleep(self._flush_interval)
        except asyncio.CancelledError:
            pass

    async def recv(self, action_name: str):
        return await self._cache[action_name].get()

    async def send(self, msg: Any, target_id: str):
        if target_id not in self._ready_actors:
            event = self._ready_events.setdefault(target_id, asyncio.Event())
            await event.wait()
        self.logger.info(f"Got message: {msg}")
        data = cloudpickle.dumps(msg)
        self._input_queue.put_nowait((data, target_id))

    async def wait_for_ready(self, expected: int, timeout: Optional[float] = None):
        # Define the core waiting logic
        async def _wait_for_condition():
            async with self._ready_condition:
                await self._ready_condition.wait_for(
                    lambda: len(self._ready_actors) >= expected
                )

        if timeout is not None:
            try:
                await asyncio.wait_for(_wait_for_condition(), timeout=timeout)
            except asyncio.TimeoutError:
                self.logger.error(
                    f"Only {len(self._ready_actors)}/{expected} ready after {timeout}s"
                )
                raise asyncio.TimeoutError
        else:
            await _wait_for_condition()

    async def broadcast(self, msg: Any, expected: int):
        await self.wait_for_ready(expected=expected)

        data = cloudpickle.dumps(msg)
        for actor_id in list(self._ready_actors):
            await self._conn.send(data, actor_id)

    async def stop(self, timeout: float = 10.0):
        data = cloudpickle.dumps(("stop", (), None))
        for actor_id in list(self._ready_actors):
            await self._conn.send(data, actor_id)
        await asyncio.sleep(timeout)


class _ActorBase(ABC):
    def __init__(
        self,
        name: str,
        max_workers: int = 1,
        run_in_executor: bool = True,
        send_timeout: float = 5.0,
        send_retries: int = 3,
    ):
        self._name = name
        self._secret = secrets.token_hex(16)
        self._conn: Union[ServerConnection, ClientConnection] = None
        self._stop: asyncio.Event = None
        self._input_queue: asyncio.Queue = None
        self._output_queue: asyncio.Queue = None
        self.logger = None
        self._sync_executor = None
        self._loop = None
        self._max_workers = max_workers
        self._send_timeout = send_timeout
        self._send_retries = send_retries
        self._free_slots = None
        self._running_tasks = None
        self._run_in_executor = run_in_executor

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__actions__ = {}
        for base in cls.__mro__:
            for attr in base.__dict__.values():
                if callable(attr) and hasattr(attr, "__action_name__"):
                    cls.__actions__.setdefault(attr.__action_name__, attr)

    def _init_runtime(self):
        self.logger = setup_logger(name=self._name, subdir="actors")
        self._stop = asyncio.Event()
        self._input_queue = asyncio.Queue()
        self._output_queue = asyncio.Queue()
        self._sync_executor = ThreadPoolExecutor(max_workers=self._max_workers)
        self._loop = asyncio.get_running_loop()
        self._free_slots = asyncio.Semaphore(self._max_workers)
        self._running_tasks = set()

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

    @classmethod
    def _build_action_signatures(cls) -> Dict[str, inspect.Signature]:
        sigs = {}
        for name, fn in cls.__actions__.items():
            sig = inspect.signature(fn)
            params = [p for p in sig.parameters.values() if p.name != "self"]
            sigs[name] = sig.replace(parameters=params)
        return sigs

    @abstractmethod
    async def _run(self): ...

    async def _recv(self):
        self.logger.info("Receive loop started.")
        while not self._stop.is_set():
            try:
                frames = await self._conn.recv(timeout=5.0)
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
                payload = cloudpickle.dumps(data) if data != _READY_SENTINEL else data
                success = False
                if isinstance(self._conn, ServerConnection):
                    for i in range(self._send_retries):
                        success = await self._conn.send(
                            payload, target_id, timeout=self._send_timeout
                        )
                        if success:
                            break
                else:
                    for i in range(self._send_retries):
                        success = await self._conn.send(
                            payload, timeout=self._send_timeout
                        )
                        if success:
                            break
                if not success:
                    self.logger.warning(f"Sending message to {target_id} failed")
                else:
                    self.logger.info(f"Sent results to {target_id}")
            except Exception as e:
                self.logger.debug(f"Send failed with error: {str(e)}")

    @action
    def stop(self):
        try:
            if len(self._running_tasks) > 0:
                self.logger.warning(f"Not all invocations are done. running tasks = {len(self._running_tasks)}")
            self._stop.set()
            self.logger.info("Actor stop set")
            
        except Exception as e:
            self.logger.error(f"Setting stop failed with exception {e}")
            raise

    async def _invoke(self, target_id: str, msg: Union[List, Tuple]):
        async def _execute(action_name, args, kwargs):
            act = self.__actions__.get(action_name)
            self.logger.info(f"Invoking {action_name}")
            if inspect.iscoroutinefunction(act):
                result = await act(self, *args, **(kwargs or {}))
            elif inspect.isasyncgenfunction(act):
                # async for r in act(self, *args, **(kwargs or {})):
                #     self._output_queue.put_nowait((target_id, r))
                self.logger.warning("Actors don't support async generators yet.")
                result = None
            else:
                if self._run_in_executor:
                    result = await self._loop.run_in_executor(
                        self._sync_executor, partial(act, self, *args, **(kwargs or {}))
                    )
                else:
                    result = act(self, *args, **(kwargs or {}))
            return result

        async with self._free_slots:
            if isinstance(msg, list):
                result = []
                for action_name, args, kwargs in msg:
                    r = await _execute(action_name, args, kwargs)
                    result.append(r)

                if not inspect.isasyncgenfunction(self.__actions__.get(action_name)):
                    self._output_queue.put_nowait((target_id, (action_name, result)))
            elif isinstance(msg, tuple):
                action_name, args, kwargs = msg
                result = await _execute(action_name, args, kwargs)

                if not inspect.isasyncgenfunction(self.__actions__.get(action_name)):
                    self._output_queue.put_nowait((target_id, (action_name, result)))
            else:
                self.logger.warning("Actions only support either a tuple or list(tuple)")
                result = None

    def _task_done_callback(self, task: asyncio.Task):
        self._running_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self.logger.error(f"Invoke task failed with error: {exc}")

    async def _main_loop(self):
        self.logger.info("Main loop started.")
        while not self._stop.is_set():
            try:
                target_id, msg = await asyncio.wait_for(
                    self._input_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            task = asyncio.create_task(self._invoke(target_id, msg))
            self._running_tasks.add(task)
            task.add_done_callback(self._task_done_callback)
        self.logger.info("Main loop stopped.")

    async def on_start(self):
        pass

    async def on_stop(self):
        pass

    def __call__(self):
        from ensemble_launcher.loop import run
        run(self._run())

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
        ckpt_dir: Optional[str] = None,
        max_workers: int = 2,
        run_in_executor: bool = True,
        send_timeout: float = 5.0,
        send_retries: int = 3,
    ):
        super().__init__(
            name,
            max_workers=max_workers,
            send_timeout=send_timeout,
            send_retries=send_retries,
            run_in_executor=run_in_executor,
        )
        if ckpt_dir is None:
            ckpt_dir = f"{os.getcwd()}/.actor_ckpt_{uuid.uuid4().hex[:6]}"
        os.makedirs(ckpt_dir, exist_ok=True)
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
    ) -> Optional[ActorHandle]:
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

        return ActorHandle(
            clientconn,
            self._build_action_signatures(),
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

        await self.on_start()

        self._start_transport()
        await self._conn.open()

        os.makedirs(self.ckpt_dir, exist_ok=True)
        fname = f"{self.ckpt_dir}/{self.name}.ckpt"
        with open(fname, "w") as f:
            f.write(self._conn.get_state().serialize())

        self.logger.info("Done opening the server!")

        await asyncio.gather(self._recv(), self._send(), self._main_loop())

        await self.on_stop()

        await self._conn.close()


class PrivateActor(_ActorBase):
    def __init__(
        self,
        name: str,
        client_conn: ClientConnection,
        max_workers: int = 2,
        run_in_executor: bool = True,
        send_timeout: float = 5.0,
        send_retries: int = 3,
    ):
        super().__init__(
            name,
            max_workers=max_workers,
            send_timeout=send_timeout,
            send_retries=send_retries,
            run_in_executor=run_in_executor,
        )
        self._conn = client_conn

    @classmethod
    def create_handle(
        cls,
        server_conn: ServerConnection,
        default_target_id: Optional[str] = None,
        **kwargs,
    ) -> PrivateActorHandle:
        return PrivateActorHandle(
            server_conn,
            cls._build_action_signatures(),
            default_target_id=default_target_id,
            **kwargs,
        )

    async def _signal_ready(self):
        await self._output_queue.put((None, _READY_SENTINEL))
        self.logger.info("Pushed ready signal to the output queue.")

    async def _run(self):
        self._init_runtime()

        await self.on_start()

        await self._conn.open()

        self.logger.info("Connected to server, ready!")

        await asyncio.gather(
            self._recv(), self._send(), self._main_loop(), self._signal_ready()
        )

        await self.on_stop()

        await self._conn.close()


Actor = PublicActor


def actor(fn: Callable) -> PublicActor:
    class _FnActor(PublicActor):
        @action
        def call(self, *args: Any) -> Any:
            return fn(*args)

    _FnActor.__name__ = fn.__name__
    _FnActor.__qualname__ = fn.__qualname__
    return _FnActor(name=fn.__name__)
