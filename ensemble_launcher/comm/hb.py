import asyncio
import os
import queue
import random
import time
from multiprocessing import Queue as MPQueue
from typing import Dict, List, Optional, Tuple

from .pipe import AsyncConnection, ClientConnection, ServerConnection

_HB_PING = b"\x01"


def _decode_identity(raw: bytes) -> Tuple[str, str, Optional[str]]:
    full_id = raw.decode()
    parts = full_id.split(":", 1)
    return full_id, parts[0], parts[1] if len(parts) > 1 else None


class HeartBeatProcess:
    def __init__(
        self,
        node_id: str,
        secret_id: str,
        parent_id: Optional[str],
        hb_parent_conn: Optional[ClientConnection],
        hb_server_conns: List[ServerConnection],
        heartbeat_interval: float,
        heartbeat_dead_threshold: float,
        control_queue: MPQueue,
        notify_queue: MPQueue,
        initial_children: Optional[Dict[str, str]] = None,
    ):
        self.logger = None
        self._node_id = node_id
        self._secret_id = secret_id
        self._parent_id = parent_id
        self._hb_parent_conn = hb_parent_conn
        self._hb_server_conns = list(hb_server_conns)
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_dead_threshold = heartbeat_dead_threshold
        self._control_queue = control_queue
        self._notify_queue = notify_queue

        self._children: Dict[str, Optional[float]] = {}
        self._children_secrets: Dict[str, str] = {}
        if initial_children:
            for child_id, child_secret in initial_children.items():
                self._children[child_id] = None
                self._children_secrets[child_id] = child_secret

        self._child_queues: Dict[str, asyncio.Queue] = {}
        self._child_tasks: Dict[str, asyncio.Task] = {}
        self._stop: Optional[asyncio.Event] = None
        self._tasks: List[asyncio.Task] = []
        self._last_parent_hb_time: Optional[float] = None
        self._parent_ready_sent = False

    def _setup_logger(self):
        from ensemble_launcher.logging import setup_logger

        self.logger = setup_logger(
            name=f"hb.{self._node_id}",
            node_id=f"hb-{self._node_id}",
            log_dir="logs",
        )

    def __call__(self):
        self._setup_logger()
        self._parent_pid = os.getppid()
        self.logger.info(
            f"{self._node_id}: HB process started (pid={os.getpid()}, parent_pid={self._parent_pid})"
        )
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run())
        finally:
            loop.close()
        self.logger.info(f"{self._node_id}: HB process exiting")

    def _start_child_dispatch(self, child_id: str):
        q = asyncio.Queue()
        self._child_queues[child_id] = q
        task = asyncio.create_task(self._child_dispatch_loop(child_id))
        self._child_tasks[child_id] = task
        self._tasks.append(task)

    async def _run(self):
        self._stop = asyncio.Event()

        self.logger.info(
            f"{self._node_id}: HB opening {len(self._hb_server_conns)} server conn(s)"
        )
        for conn in self._hb_server_conns:
            if not conn.is_open:
                await conn.open()
            self._notify_queue.put(("hb_address", conn.address))
            self.logger.info(f"{self._node_id}: HB server bound to {conn.address}")

        for child_id, child_secret in self._children_secrets.items():
            for conn in self._hb_server_conns:
                conn.add_expected_remote(child_id, child_secret)
            self.logger.info(f"{self._node_id}: HB added expected remote {child_id}")

        if self._hb_parent_conn is not None:
            self.logger.info(f"{self._node_id}: HB opening parent conn")
            await self._hb_parent_conn.open()
            self.logger.info(f"{self._node_id}: HB parent conn opened")

        if self._hb_parent_conn is not None:
            self._tasks.append(asyncio.create_task(self._parent_hb_loop()))

        for conn in self._hb_server_conns:
            self._tasks.append(asyncio.create_task(self._hb_recv_loop(conn)))

        for child_id in self._children:
            self._start_child_dispatch(child_id)

        self._tasks.append(asyncio.create_task(self._dead_check_loop()))
        self._tasks.append(asyncio.create_task(self._control_loop()))
        self._tasks.append(asyncio.create_task(self._parent_alive_check_loop()))

        self.logger.info(f"{self._node_id}: HB all tasks started, waiting for stop")
        await self._stop.wait()
        self.logger.info(f"{self._node_id}: HB stop received, cleaning up")

        for t in self._tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

        for conn in self._hb_server_conns:
            if conn.is_open:
                await conn.close()
        if self._hb_parent_conn is not None and self._hb_parent_conn.is_open:
            await self._hb_parent_conn.close()

    async def _parent_hb_loop(self):
        conn = self._hb_parent_conn
        self._last_parent_hb_time = time.time()
        self.logger.info(f"{self._node_id}: HB parent loop started")
        try:
            while not self._stop.is_set():
                try:
                    await conn.send(_HB_PING)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.warning(
                        f"{self._node_id}: HB send to parent error: {e}"
                    )

                try:
                    raw = await asyncio.wait_for(
                        conn.recv(), timeout=self._heartbeat_interval
                    )
                    if raw is not None and len(raw) >= 2 and raw[1] == _HB_PING:
                        if not self._parent_ready_sent:
                            self._notify_queue.put(("ready_parent",))
                            self._parent_ready_sent = True
                        self._last_parent_hb_time = time.time()
                except (asyncio.TimeoutError, TimeoutError):
                    pass
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.warning(
                        f"{self._node_id}: HB recv from parent error: {e}"
                    )

                elapsed = time.time() - self._last_parent_hb_time
                if elapsed > self._heartbeat_dead_threshold:
                    self.logger.warning(
                        f"{self._node_id}: Parent HB dead (elapsed={elapsed:.1f}s, threshold={self._heartbeat_dead_threshold}s)"
                    )
                    self._notify_queue.put(("dead_parent",))
                    break

                jitter = self._heartbeat_interval * (1 + random.uniform(-0.1, 0.1))
                await asyncio.sleep(jitter)
        except asyncio.CancelledError:
            pass
        self.logger.info(f"{self._node_id}: HB parent loop exiting")

    async def _hb_recv_loop(self, conn: AsyncConnection):
        try:
            while not self._stop.is_set():
                try:
                    parts = await conn.recv()
                    full_id, sender_id, _ = _decode_identity(parts[0])
                    self._child_queues[sender_id].put_nowait((full_id, conn))
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.warning(f"{self._node_id}: HB recv error: {e}")
        except asyncio.CancelledError:
            pass

    async def _child_dispatch_loop(self, child_id: str):
        self.logger.info(f"{self._node_id}: HB child dispatch loop started for {child_id}")
        q = self._child_queues[child_id]
        try:
            while not self._stop.is_set():
                try:
                    full_id, conn = await q.get()

                    first_ping = self._children[child_id] is None
                    self._children[child_id] = time.perf_counter()

                    await conn.send(_HB_PING, full_id)

                    if first_ping:
                        self.logger.info(
                            f"{self._node_id}: HB first ping from child {child_id}, sending ready_child"
                        )
                        self._notify_queue.put(("ready_child", child_id))

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.warning(
                        f"{self._node_id}: HB child dispatch error for {child_id}: {e}"
                    )
        except asyncio.CancelledError:
            pass

    async def _dead_check_loop(self):
        try:
            while not self._stop.is_set():
                jitter = self._heartbeat_interval * (1 + random.uniform(-0.1, 0.1))
                await asyncio.sleep(jitter)
                now = time.perf_counter()
                for child_id, last in list(self._children.items()):
                    if last is not None:
                        elapsed = now - last
                        if elapsed > self._heartbeat_dead_threshold:
                            self.logger.warning(
                                f"{self._node_id}: Child {child_id} HB dead "
                                f"(last={elapsed:.1f}s ago, threshold={self._heartbeat_dead_threshold}s)"
                            )
                            self._notify_queue.put(("dead_child", child_id))
        except asyncio.CancelledError:
            pass

    async def _parent_alive_check_loop(self):
        try:
            while not self._stop.is_set():
                if os.getppid() != self._parent_pid:
                    self.logger.warning(
                        f"{self._node_id}: Parent process died (ppid changed from "
                        f"{self._parent_pid} to {os.getppid()}), shutting down HB process"
                    )
                    self._stop.set()
                    break
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass

    async def _control_loop(self):
        self.logger.info(f"{self._node_id}: HB control loop started")
        loop = asyncio.get_running_loop()
        try:
            while not self._stop.is_set():
                try:
                    msg = await loop.run_in_executor(
                        None, self._control_queue.get, True, 0.1
                    )
                except queue.Empty:
                    continue
                except asyncio.CancelledError:
                    break

                kind = msg[0]
                self.logger.info(f"{self._node_id}: HB control received: {kind}")
                if kind == "stop":
                    self._stop.set()
                    break
                elif kind == "add_child":
                    child_id, child_secret = msg[1], msg[2]
                    self._children[child_id] = None
                    self._children_secrets[child_id] = child_secret
                    for conn in self._hb_server_conns:
                        conn.add_expected_remote(child_id, child_secret)
                    self._start_child_dispatch(child_id)
                    self.logger.info(f"{self._node_id}: HB added child {child_id}")
                elif kind == "remove_child":
                    child_id = msg[1]
                    self._children.pop(child_id, None)
                    self._children_secrets.pop(child_id, None)
                    for conn in self._hb_server_conns:
                        conn.remove_expected_remote(child_id)
                    task = self._child_tasks.pop(child_id, None)
                    if task and not task.done():
                        task.cancel()
                    self._child_queues.pop(child_id, None)
                    self.logger.info(f"{self._node_id}: HB removed child {child_id}")
                elif kind == "add_server_connection":
                    conn = msg[1]
                    if not conn.is_open:
                        await conn.open()
                    self._notify_queue.put(("hb_address", conn.address))
                    for child_id, child_secret in self._children_secrets.items():
                        conn.add_expected_remote(child_id, child_secret)
                    self._hb_server_conns.append(conn)
                    task = asyncio.create_task(self._hb_recv_loop(conn))
                    self._tasks.append(task)
                    self.logger.info(
                        f"{self._node_id}: HB added server conn {conn.address}"
                    )
        except asyncio.CancelledError:
            pass
