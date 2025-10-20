from mcp.server.fastmcp import FastMCP
import inspect
from typing import Callable, List, Any, Literal, Optional
from ..ensemble import Task
from ensemble_launcher import EnsembleLauncher
from ..config import SystemConfig, LauncherConfig
import time


class Server:
    def __init__(self,name:str = "MCP server to execute ensemble tasks", **kwargs):
        self._mcp = FastMCP(name=name,**kwargs)
    
    @property
    def mcp(self):
        return self._mcp
    
    def ensemble_tool(self,
                      fn: Optional[Callable] = None,
                      *,
                      system_config: SystemConfig = SystemConfig(name="local"),
                      launcher_config: Optional[LauncherConfig] = None):
        """
        Decorator that transforms a function into an MCP tool
        where all arguments are converted to Lists for ensembles.
        
        Can be used as:
          @server.ensemble_tool
        or
          @server.ensemble_tool(system_config=SystemConfig(...), launcher_config=...)
        """
        
        # decorator factory: if called with params returns decorator, otherwise registers the function directly
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
                return_annotation=List[sig.return_annotation] if sig.return_annotation != inspect.Signature.empty else List
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
                    # positional args (each should be a list)
                    task_args = tuple([arg[i] for arg in args])
                    task_kwargs = {}
                    for k, v in kwargs.items():
                        task_kwargs[k] = v[i]
                    tasks[task_id] = Task(
                        task_id=task_id,
                        nnodes=1,
                        ppn=1,
                        executable=target_fn,
                        args=task_args,
                        kwargs=task_kwargs
                    )

                el = EnsembleLauncher(
                    ensemble_file=tasks,
                    system_config=system_config,
                    launcher_config=launcher_config,
                    return_stdout=True
                )
                tic = time.perf_counter()
                results = el.run()
                run_time = time.perf_counter() - tic  # kept if you want to log/inspect
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
                "--- Original Function Documentation ---",
                f"{doc_string}"
            ])
            return self._mcp.add_tool(ensemble_wrapper)

        # if used as @server.ensemble_tool without args, fn is the function to register
        if fn is not None and callable(fn):
            return _register(fn)
        # otherwise return decorator configured with the provided system/launcher config
        return _register
     
    def run(self, transport: Literal['sse','stdio',"streamable-http"]='stdio'):
        self._mcp.run(transport=transport)
