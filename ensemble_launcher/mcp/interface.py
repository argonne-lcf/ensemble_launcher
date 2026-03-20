import concurrent.futures
import inspect
import os
import string as _string_stdlib
import uuid
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from mcp.server.fastmcp import FastMCP

from ensemble_launcher.logging import setup_logger

from ..config import LauncherConfig, SystemConfig
from ..ensemble import AsyncTask, Task
from ..orchestrator import ClusterClient


def _parse_template_params(cmd_template: str) -> List[str]:
    """Return ordered, deduplicated list of ``{param}`` names from a command template."""
    seen: set = set()
    params: List[str] = []
    for _, fname, _, _ in _string_stdlib.Formatter().parse(cmd_template):
        if fname is not None and fname not in seen:
            seen.add(fname)
            params.append(fname)
    return params


class ELFastMCP(FastMCP):
    def __init__(
        self,
        name: str = "MCP interface for ensemble tasks",
        checkpoint_dir: Optional[str] = None,
        node_id: str = "global",
        **kwargs,
    ):
        """
        Args:
            name:           Name passed to FastMCP.
            checkpoint_dir: Checkpoint directory of a running EnsembleLauncher
                            cluster. The Interface will create and manage its own
                            ClusterClient pointing at this cluster.
                            Start the EnsembleLauncher externally before calling
                            interface.run().
            node_id:        Node to connect to (default ``"global"``).
            **kwargs:       Forwarded to FastMCP.
        """
        super().__init__(name=name, **kwargs)
        self._checkpoint_dir = checkpoint_dir
        self._node_id = node_id
        self._client: Optional[ClusterClient] = None
        self.logger = setup_logger("mcp_interface", log_dir=f"{os.getcwd()}/logs")

    def ensemble_tool(
        self,
        fn: Optional[Union[Callable, str]] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        nnodes: int = 1,
        ppn: int = 1,
        gpus_per_process: int = 0,
        env: Optional[Dict[str, str]] = None,
        system_config: SystemConfig = SystemConfig(name="local"),
        launcher_config: Optional[LauncherConfig] = None,
    ):
        """
        Decorator that transforms a function or shell command template into a
        batch MCP tool where each argument accepts a list of values and one
        task is created per element (zip / one-to-one).

        **Callable** — auto-detects ``async def`` and uses ``AsyncTask``:
          @server.ensemble_tool
          @server.ensemble_tool(nnodes=2, ppn=4)

        **String command template** — placeholders ``{param}`` become the
        tool's parameters. ``name`` and ``description`` are required:
          server.ensemble_tool("./sim --P {P} --T {T}",
                               name="run_sim", description="Run simulation ensemble")

        Args:
            fn:               Callable or shell command template string.
            name:             Tool name (required for string templates).
            description:      Tool description (required for string templates).
            nnodes:           Number of nodes per task.
            ppn:              Processes per node per task.
            gpus_per_process: GPUs per process per task.
            env:              Extra environment variables for each task.
            system_config:    System configuration for EnsembleLauncher.
            launcher_config:  Launcher configuration for EnsembleLauncher.
        """
        _env = env or {}

        def _register_callable(target_fn: Callable):
            doc_string = inspect.getdoc(target_fn) or ""
            sig = inspect.signature(target_fn)
            task_cls = AsyncTask if inspect.iscoroutinefunction(target_fn) else Task
            tool_name = name or ("ensemble_" + target_fn.__name__)

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
                    tasks[task_id] = task_cls(
                        task_id=task_id,
                        nnodes=nnodes,
                        ppn=ppn,
                        ngpus_per_process=gpus_per_process,
                        env=_env,
                        executable=target_fn,
                        args=task_args,
                        kwargs=task_kwargs,
                    )
                futures = self._client.submit_batch(list(tasks.values()))
                concurrent.futures.wait(futures)
                return [future.result() for future in futures]

            ensemble_wrapper.__name__ = tool_name
            ensemble_wrapper.__signature__ = new_sig
            ensemble_wrapper.__globals__.update(target_fn.__globals__)
            ensemble_wrapper.__doc__ = "\n".join(
                [
                    f"[Ensemble Tool] Runs '{target_fn.__name__}' on a range of input parameters.",
                    "This tool expects a list for each argument. It creates ensemble runs by pairing arguments in a one-to-one (zip) fashion.",
                    "**All provided argument lists must have the same length.**",
                    f"Resources per task: nnodes={nnodes}, ppn={ppn}, gpus_per_process={gpus_per_process}",
                    "--- Original Function Documentation ---",
                    f"{doc_string}",
                ]
            )
            return self.add_tool(ensemble_wrapper)

        def _register_str(cmd_template: str):
            if not name:
                raise ValueError(
                    "name is required when registering a string command template"
                )
            if not description:
                raise ValueError(
                    "description is required when registering a string command template"
                )

            warnings.warn(
                f"String command tool '{name}': return value is always a str captured from stdout. "
                "Ensure your command prints its result to stdout and that the EnsembleLauncher "
                "cluster was started with LauncherConfig(return_stdout=True).",
                UserWarning,
                stacklevel=2,
            )

            params = _parse_template_params(cmd_template)
            new_sig = inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        p, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=List[Any]
                    )
                    for p in params
                ],
                return_annotation=List[str],
            )

            def ensemble_wrapper(**kwargs) -> List[str]:
                try:
                    ntasks = len(next(iter(kwargs.values())))
                except (StopIteration, TypeError):
                    return []

                tasks = {}
                for i in range(ntasks):
                    task_id = f"task-{i}"
                    cmd = cmd_template
                    for param, values in kwargs.items():
                        cmd = cmd.replace(f"{{{param}}}", str(values[i]))
                    tasks[task_id] = Task(
                        task_id=task_id,
                        nnodes=nnodes,
                        ppn=ppn,
                        ngpus_per_process=gpus_per_process,
                        env=_env,
                        executable=cmd,
                    )
                futures = self._client.submit_batch(list(tasks.values()))
                concurrent.futures.wait(futures)
                return [future.result() for future in futures]

            ensemble_wrapper.__name__ = "ensemble_" + name
            ensemble_wrapper.__signature__ = new_sig
            ensemble_wrapper.__doc__ = "\n".join(
                [
                    f"[Ensemble Tool] Runs '{name}' on a range of input parameters.",
                    "This tool expects a list for each argument. It creates ensemble runs by pairing arguments in a one-to-one (zip) fashion.",
                    "**All provided argument lists must have the same length.**",
                    f"Resources per task: nnodes={nnodes}, ppn={ppn}, gpus_per_process={gpus_per_process}",
                    "**Return value:** List[str] — stdout captured from each task. The command must print its result to stdout.",
                    "**Requirement:** EnsembleLauncher cluster must be started with LauncherConfig(return_stdout=True).",
                    "--- Description ---",
                    description,
                ]
            )
            return self.add_tool(ensemble_wrapper)

        if fn is not None and isinstance(fn, str):
            return _register_str(fn)
        if fn is not None and callable(fn):
            return _register_callable(fn)

        def _register(target: Union[Callable, str]):
            if isinstance(target, str):
                return _register_str(target)
            return _register_callable(target)

        return _register

    def tool(
        self,
        fn: Optional[Union[Callable, str]] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        nnodes: int = 1,
        ppn: int = 1,
        gpus_per_process: int = 0,
        env: Optional[Dict[str, str]] = None,
    ):
        """
        Decorator that registers a function or shell command template as a
        cluster-mode MCP tool. Each invocation submits one Task to the running
        EnsembleLauncher via ClusterClient.

        **Callable** — auto-detects ``async def`` and uses ``AsyncTask``:
          @server.tool
          @server.tool(nnodes=2, ppn=4, gpus_per_process=1)

        **String command template** — placeholders ``{param}`` become the
        tool's parameters. ``name`` and ``description`` are required:
          server.tool("./sim --pressure {P} --temp {T}",
                      name="run_sim", description="Run one simulation")

        Args:
            fn:               Callable or shell command template string.
            name:             Tool name (required for string templates).
            description:      Tool description (required for string templates).
            nnodes:           Number of nodes to request per call.
            ppn:              Processes per node.
            gpus_per_process: GPUs per process.
            env:              Extra environment variables for the task.
        """
        _env = env or {}

        def _register_callable(target_fn: Callable):
            doc_string = inspect.getdoc(target_fn) or ""
            sig = inspect.signature(target_fn)
            task_cls = AsyncTask if inspect.iscoroutinefunction(target_fn) else Task
            tool_name = name or target_fn.__name__

            def cluster_wrapper(*args, **kwargs):
                if self._client is None:
                    raise RuntimeError(
                        "No active ClusterClient. Ensure the EnsembleLauncher cluster "
                        "is running and interface.run() has been called."
                    )
                task = task_cls(
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
                result = fut.result()
                self.logger.info(f"Got the result from {task.task_id} = {result}")
                return result

            cluster_wrapper.__name__ = tool_name
            cluster_wrapper.__signature__ = sig
            cluster_wrapper.__globals__.update(target_fn.__globals__)
            cluster_wrapper.__doc__ = "\n".join(
                [
                    f"[Cluster Tool] Runs '{target_fn.__name__}' as a cluster task.",
                    f"Resources: nnodes={nnodes}, ppn={ppn}, gpus_per_process={gpus_per_process}",
                    "--- Original Function Documentation ---",
                    f"{doc_string}",
                ]
            )
            return self.add_tool(cluster_wrapper)

        def _register_str(cmd_template: str):
            if not name:
                raise ValueError(
                    "name is required when registering a string command template"
                )
            if not description:
                raise ValueError(
                    "description is required when registering a string command template"
                )

            warnings.warn(
                f"String command tool '{name}': return value is always a str captured from stdout. "
                "Ensure your command prints its result to stdout and that the EnsembleLauncher "
                "cluster was started with LauncherConfig(return_stdout=True).",
                UserWarning,
                stacklevel=2,
            )

            params = _parse_template_params(cmd_template)
            sig = inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        p, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Any
                    )
                    for p in params
                ],
                return_annotation=str,
            )

            def cluster_wrapper(**kwargs) -> str:
                if self._client is None:
                    raise RuntimeError(
                        "No active ClusterClient. Ensure the EnsembleLauncher cluster "
                        "is running and interface.run() has been called."
                    )
                cmd = cmd_template
                for param, value in kwargs.items():
                    cmd = cmd.replace(f"{{{param}}}", str(value))
                task = Task(
                    task_id=str(uuid.uuid4()),
                    nnodes=nnodes,
                    ppn=ppn,
                    ngpus_per_process=gpus_per_process,
                    env=_env,
                    executable=cmd,
                )
                fut = self._client.submit(task)
                result = fut.result()
                self.logger.info(f"Got the result from {task.task_id} = {result}")
                return result

            cluster_wrapper.__name__ = name
            cluster_wrapper.__signature__ = sig
            cluster_wrapper.__doc__ = "\n".join(
                [
                    f"[Cluster Tool] Runs '{name}' as a cluster task.",
                    f"Resources: nnodes={nnodes}, ppn={ppn}, gpus_per_process={gpus_per_process}",
                    "**Return value:** str — stdout captured from the task. The command must print its result to stdout.",
                    "**Requirement:** EnsembleLauncher cluster must be started with LauncherConfig(return_stdout=True).",
                    "--- Description ---",
                    description,
                ]
            )
            return self.add_tool(cluster_wrapper)

        if fn is not None and isinstance(fn, str):
            return _register_str(fn)
        if fn is not None and callable(fn):
            return _register_callable(fn)

        def _register(target: Union[Callable, str]):
            if isinstance(target, str):
                return _register_str(target)
            return _register_callable(target)

        return _register

    def init_client(self):
        if self._checkpoint_dir is not None:
            self._client = ClusterClient(
                checkpoint_dir=self._checkpoint_dir, node_id=self._node_id
            )
            self._client.start()

    def run(self, transport: Literal["sse", "stdio", "streamable-http"] = "stdio"):
        self.init_client()
        try:
            super().run(transport=transport)
        finally:
            if self._client is not None:
                self._client.teardown()
                self._client = None
