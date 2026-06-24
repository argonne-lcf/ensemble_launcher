import secrets
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from ensemble_launcher.comm.pipe import ClientConnection, transport_registry
from ensemble_launcher.ensemble.actor import PrivateActor, PrivateActorHandle, action


class ActorPool(PrivateActor):
    """A PrivateActor that manages a pool of child PrivateActors.

    Submittable as a task to run remotely on a cluster worker. Proxies
    interactions to child actors via actions (send_to, broadcast,
    broadcast_stream). Children are created lazily in on_start().
    """

    def __init__(
        self,
        name: str,
        client_conn: ClientConnection,
        actor_class: Union[Type[PrivateActor], List[Type[PrivateActor]]],
        n_actors: int,
        actor_kwargs: Union[Dict[str, Any], List[Dict[str, Any]]],
        task_kwargs: Union[Dict[str, Any], List[Dict[str, Any]]],
        checkpoint_dir: str,
        checkpoint_timeout: float = 300,
        child_transport: str = "zmq",
        req_res: bool = False,
        child_send_timeout: float = 5.0,
        child_send_retries: int = 3,
        submit_to: str = "global",
        child_ready_timeout: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(name, client_conn, **kwargs)
        if isinstance(actor_class, list):
            assert len(actor_class) == n_actors
            self._actor_class_list = list(actor_class)
        else:
            self._actor_class_list = [actor_class] * n_actors
        self._n_children = n_actors
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_timeout = checkpoint_timeout
        self._submit_to = submit_to
        self._child_transport_name = child_transport
        self._req_res = req_res
        self._child_send_timeout = child_send_timeout
        self._child_send_retries = child_send_retries
        self._child_ready_timeout = child_ready_timeout

        if isinstance(actor_kwargs, dict):
            self._actor_kwargs_list = [dict(actor_kwargs) for _ in range(n_actors)]
        else:
            assert len(actor_kwargs) == n_actors
            self._actor_kwargs_list = [dict(kw) for kw in actor_kwargs]

        if isinstance(task_kwargs, dict):
            self._task_kwargs_list = [dict(task_kwargs) for _ in range(n_actors)]
        else:
            assert len(task_kwargs) == n_actors
            self._task_kwargs_list = [dict(kw) for kw in task_kwargs]

        self._child_handle: Optional[PrivateActorHandle] = None
        self._child_names: List[str] = []
        self._child_futures: list = []
        self._server_secret: Optional[str] = None
        self._cluster_client = None

    async def on_start(self):
        try:
            transport_entry = transport_registry.get(self._child_transport_name)
            transport = transport_entry["transport"]()
            server_id = f"{self._name}_pool"
            self._server_secret = secrets.token_hex(16)

            actors = []
            tasks = []
            server_conn = None

            for i in range(self._n_children):
                akw = dict(self._actor_kwargs_list[i])
                actor_name = akw.pop("name", f"{self._name}-child-{i}")
                self._child_names.append(actor_name)

                server, client = transport.create_child_pipe(
                    server_id,
                    self._server_secret,
                    actor_name,
                    self._server_secret,
                    req_res=self._req_res,
                )
                if server_conn is None:
                    server_conn = server

                akw["client_conn"] = client
                akw["name"] = actor_name
                actor = self._actor_class_list[i](**akw)
                actors.append(actor)

                tkw = dict(self._task_kwargs_list[i])
                task_id = tkw.pop("task_id", actor_name)
                nnodes = tkw.pop("nnodes")
                ppn = tkw.pop("ppn")
                task = actor.create_task(task_id=task_id, nnodes=nnodes, ppn=ppn, **tkw)
                tasks.append(task)

            from ensemble_launcher.orchestrator import ClusterClient  # noqa: E402

            self._cluster_client = ClusterClient(
                node_id=self._submit_to,
                checkpoint_dir=self._checkpoint_dir,
                checkpoint_timeout=self._checkpoint_timeout,
            )
            self._cluster_client.__enter__()
            self._child_futures = [self._cluster_client.submit(t) for t in tasks]

            self._child_handle = PrivateActor.create_handle(
                server_conn,
                send_timeout=self._child_send_timeout,
                send_retries=self._child_send_retries,
            )
            await self._child_handle.open()

            await self._child_handle.wait_for_ready(
                expected=self._n_children, timeout=self._child_ready_timeout
            )
            self.logger.info(
                f"ActorPool '{self._name}': {self._n_children} children ready"
            )
        except Exception as e:
            if self._child_handle is not None:
                await self._child_handle.stop()
            self.logger.error(f"on_start failed with Exception {e}")
            raise e

    @action
    async def invoke(self, actor_index: int, msg: Tuple):
        target_id = f"{self._child_names[actor_index]}:{self._server_secret}"
        await self._child_handle.send(msg, target_id=target_id)
        _, result = await self._child_handle.recv()
        return result

    @action
    async def invoke_all(self, msgs: Union[Tuple, List[Tuple]]):
        if isinstance(msgs, tuple):
            for i in range(self._n_children):
                target_id = f"{self._child_names[i]}:{self._server_secret}"
                await self._child_handle.send(msgs, target_id=target_id)
        else:
            for i in range(self._n_children):
                target_id = f"{self._child_names[i]}:{self._server_secret}"
                await self._child_handle.send(msgs[i], target_id=target_id)
        results = []
        for _ in range(self._n_children):
            _, result = await self._child_handle.recv()
            results.append(result)
        return results

    @action
    async def invoke_all_stream(self, msgs: Union[Tuple, List[Tuple]]):
        if isinstance(msgs, tuple):
            for i in range(self._n_children):
                target_id = f"{self._child_names[i]}:{self._server_secret}"
                await self._child_handle.send(msgs, target_id=target_id)
        else:
            for i in range(self._n_children):
                target_id = f"{self._child_names[i]}:{self._server_secret}"
                await self._child_handle.send(msgs[i], target_id=target_id)
        for _ in range(self._n_children):
            _, result = await self._child_handle.recv()
            yield result

    @action
    def get_n_actors(self) -> int:
        return self._n_children

    @action
    def get_actor_ids(self) -> List[str]:
        return list(self._child_names)

    async def on_stop(self):
        self.logger.info("In on stop")
        if self._child_handle:
            try:
                self.logger.info(f"Broadcasting stop to {self._n_children} children")
                await self._child_handle.broadcast(
                    ("stop", (), None), expected=self._n_children
                )
            except Exception as e:
                self.logger.warning(f"Broadcasting stop to children failed: {e}")

            for f in self._child_futures:
                try:
                    f.result(timeout=60)
                except Exception:
                    pass

            await self._child_handle.close()

        if self._cluster_client:
            try:
                self._cluster_client.__exit__(None, None, None)
            except Exception:
                pass
