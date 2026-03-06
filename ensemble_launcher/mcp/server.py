import inspect
import os
import time
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional

from mcp.server.fastmcp import FastMCP

from ensemble_launcher import EnsembleLauncher

from ..config import LauncherConfig, SystemConfig
from ..ensemble import Task
from ..orchestrator import ClusterClient


class Server:
    def __init__(
        self,
        name: str = "MCP server to execute ensemble tasks",
        system_config: Optional[SystemConfig] = None,
        launcher_config: Optional[LauncherConfig] = None,
        Nodes: Optional[List[str]] = None,
        **kwargs,
    ):
        self._mcp = FastMCP(name=name, **kwargs)
        self._system_config = system_config
        self._launcher_config = launcher_config
        self._Nodes = Nodes
        self._el: Optional[EnsembleLauncher] = None
        self._client: Optional[ClusterClient] = None

    @property
    def mcp(self):
        return self._mcp

    def ensemble_tool(
        self,
        fn: Optional[Callable] = None,
        *,
        nnodes: int = 1,
        ppn: int = 1,
        gpus_per_process: int = 0,
        env: Optional[Dict[str, str]] = None,
        system_config: SystemConfig = SystemConfig(name="local"),
        launcher_config: Optional[LauncherConfig] = None,
    ):
        """
        Decorator that transforms a function into an MCP tool
        where all arguments are converted to Lists for ensembles.

        Can be used as:
          @server.ensemble_tool
        or
          @server.ensemble_tool(nnodes=2, ppn=4, system_config=SystemConfig(...), ...)

        Args:
            nnodes:           Number of nodes per task.
            ppn:              Processes per node per task.
            gpus_per_process: GPUs per process per task.
            env:              Extra environment variables for each task.
            system_config:    System configuration for EnsembleLauncher.
            launcher_config:  Launcher configuration for EnsembleLauncher.
        """
        _env = env or {}

        def _register(target_fn: Callable):
            doc_string = inspect.getdoc(target_fn) or ""
            sig = inspect.signature(target_fn)

            new_parameters = []
            for param in sig.parameters.values():
                original_annotation = param.annotation
                if original_annotation is inspect.Parameter.empty:
                    original_annotation = Any
                new_annotation = List[original_annotation]
                new_param = param.replace(annotation=new_annotation)
                new_parameters.append(new_param)

            new_sig = inspect.Signature(
                parameters=new_parameters,
                return_annotation=List[sig.return_annotation]
                if sig.return_annotation != inspect.Signature.empty
                else List,
            )

            def ensemble_wrapper(*args, **kwargs) -> List:
                try:
                    bound_args = new_sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                except TypeError as e:
                    raise TypeError(f"Error binding arguments for ensemble: {e}")

                list_arguments = bound_args.arguments
                if not list_arguments:
                    return []

                try:
                    ntasks = len(next(iter(list_arguments.values())))
                except (StopIteration, TypeError):
                    return []

                tasks = {}
                for i in range(ntasks):
                    task_id = f"task-{i}"
                    task_args = tuple([arg[i] for arg in args])
                    task_kwargs = {k: v[i] for k, v in kwargs.items()}
                    tasks[task_id] = Task(
                        task_id=task_id,
                        nnodes=nnodes,
                        ppn=ppn,
                        ngpus_per_process=gpus_per_process,
                        env=_env,
                        executable=target_fn,
                        args=task_args,
                        kwargs=task_kwargs,
                    )

                el = EnsembleLauncher(
                    ensemble_file=tasks,
                    system_config=system_config,
                    launcher_config=launcher_config,
                    return_stdout=True,
                )
                tic = time.perf_counter()
                results = el.run()
                run_time = time.perf_counter() - tic  # noqa: F841
                ret_list = []
                for i in range(ntasks):
                    task_id = f"task-{i}"
                    for r in results.data:
                        if r.task_id == task_id:
                            ret_list.append(r.data)
                return ret_list

            ensemble_wrapper.__name__ = target_fn.__name__
            ensemble_wrapper.__signature__ = new_sig
            ensemble_wrapper.__doc__ = "\n".join([
                f"[Ensemble Tool] Runs '{target_fn.__name__}' on a range of input parameters.",
                "This tool expects a list for each argument. It creates ensemble runs by pairing arguments in a one-to-one (zip) fashion.",
                "**All provided argument lists must have the same length.**",
                f"Resources per task: nnodes={nnodes}, ppn={ppn}, gpus_per_process={gpus_per_process}",
                "--- Original Function Documentation ---",
                f"{doc_string}",
            ])
            return self._mcp.add_tool(ensemble_wrapper)

        if fn is not None and callable(fn):
            return _register(fn)
        return _register

    def tool(
        self,
        fn: Optional[Callable] = None,
        *,
        nnodes: int = 1,
        ppn: int = 1,
        gpus_per_process: int = 0,
        env: Optional[Dict[str, str]] = None,
    ):
        """
        Decorator that registers a function as a cluster-mode MCP tool.

        Each invocation submits the function as a Task to the running
        EnsembleLauncher via ClusterClient, with the specified resource spec.

        Can be used as:
          @server.tool
        or
          @server.tool(nnodes=2, ppn=4, gpus_per_process=1, env={"VAR": "val"})

        Args:
            nnodes:           Number of nodes to request per call.
            ppn:              Processes per node.
            gpus_per_process: GPUs per process.
            env:              Extra environment variables for the task.
        """
        _env = env or {}

        def _register(target_fn: Callable):
            doc_string = inspect.getdoc(target_fn) or ""
            sig = inspect.signature(target_fn)

            def cluster_wrapper(*args, **kwargs):
                if self._client is None:
                    raise RuntimeError(
                        "Cluster is not running. Ensure server.run() was called "
                        "with system_config, launcher_config, and Nodes set."
                    )
                task = Task(
                    task_id=str(uuid.uuid4()),
                    nnodes=nnodes,
                    ppn=ppn,
                    ngpus_per_process=gpus_per_process,
                    env=_env,
                    executable=target_fn,
                    args=args,
                    kwargs=kwargs,
                )
                fut = self._client.submit(task)
                return fut.result()

            cluster_wrapper.__name__ = target_fn.__name__
            cluster_wrapper.__signature__ = sig
            cluster_wrapper.__doc__ = "\n".join([
                f"[Cluster Tool] Runs '{target_fn.__name__}' as a cluster task.",
                f"Resources: nnodes={nnodes}, ppn={ppn}, gpus_per_process={gpus_per_process}",
                "--- Original Function Documentation ---",
                f"{doc_string}",
            ])
            return self._mcp.add_tool(cluster_wrapper)

        if fn is not None and callable(fn):
            return _register(fn)
        return _register

    def _start_cluster(self):
        """Start EnsembleLauncher in cluster mode and connect ClusterClient."""
        if self._system_config is None or self._launcher_config is None or self._Nodes is None:
            raise RuntimeError(
                "system_config, launcher_config, and Nodes must be provided to use cluster tools."
            )
        if not self._launcher_config.cluster:
            raise RuntimeError("launcher_config.cluster must be True for cluster mode.")

        checkpoint_dir = self._launcher_config.checkpoint_dir
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(os.getcwd(), f"ckpt_{uuid.uuid4()}")
            self._launcher_config = self._launcher_config.model_copy(
                update={"checkpoint_dir": checkpoint_dir}
            )

        self._el = EnsembleLauncher(
            ensemble_file={},
            system_config=self._system_config,
            launcher_config=self._launcher_config,
            Nodes=self._Nodes,
        )
        self._el.start()
        time.sleep(2.0)
        self._client = ClusterClient(checkpoint_dir=checkpoint_dir, node_id="global")
        self._client.start()

    def _stop_cluster(self):
        if self._client is not None:
            self._client.teardown()
            self._client = None
        if self._el is not None:
            self._el.stop()
            self._el = None

    def run(self, transport: Literal["sse", "stdio", "streamable-http"] = "stdio"):
        if self._system_config is not None and self._launcher_config is not None:
            self._start_cluster()
        try:
            self._mcp.run(transport=transport)
        finally:
            self._stop_cluster()