import asyncio
import json
import os
import random
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from glob import glob
from typing import Any, Dict, List, Optional, Union
import queue

import cloudpickle

from ensemble_launcher.comm.pipe import ClientConnection, get_hsn_ip_cli, find_free_port
from ensemble_launcher.ensemble.actor import PrivateActor, PublicActor, action
from ensemble_launcher.logging import get_log_dir, setup_logger

from .utils import _build_model_cache, _setup_vllm_file_logging


# ---------------------------------------------------------------------------
# Mixin 1: Offline (in-process) vLLM inference
# ---------------------------------------------------------------------------
# Two engine backends via async_engine flag:
#   False (default) — sync LLM with InprocClient. All prompts are added to
#       the engine core before stepping, so batch prefill runs in one GPU
#       kernel. Best throughput for batch workloads.
#   True — AsyncLLMEngine with AsyncMPClient (separate process + ZMQ).
#       Supports concurrent generate() calls (max_workers > 1), but the
#       engine core's busy loop steps on the first ZMQ arrival, causing
#       batch fragmentation (prompts prefill in small groups instead of
#       one large batch).
# ---------------------------------------------------------------------------

def _setup_env(name,
                logger,
                model: str,
                cache_dir: str,
                use_cached_modelinfo: bool = False,
                model_info_cache: Optional[str] = None):
    os.environ["MASTER_ADDR"] = "localhost"
    _actor_port = 10000 + (os.getpid() % 100) * 200
    _actor_port = find_free_port((_actor_port, _actor_port + 200), "localhost")
    os.environ["MASTER_PORT"] = (
        str(_actor_port) if _actor_port is not None else "0"
    )
    os.environ["VLLM_PORT"] = (
        str(_actor_port) if _actor_port is not None else "0"
    )
    os.environ["VLLM_HOST_IP"] = "localhost"
    if use_cached_modelinfo:
        os.environ["VLLM_CACHE_ROOT"] = model_info_cache
        logger.info(f"Reusing cache at {model_info_cache}")
        logger.debug("File list in cache")
        for root, dirs, files in os.walk(model_info_cache):
            for file in files:
                logger.debug(f"{file}")
    else:
        if model_info_cache is None:
            model_info_cache = f"/tmp/vllm_cache_{uuid.uuid4().hex[:6]}"
        os.environ["VLLM_CACHE_ROOT"] = model_info_cache
        try:
            os.makedirs(os.environ["VLLM_CACHE_ROOT"])
            _build_model_cache(logger=logger.getChild("BuildModelInfo"))
        except Exception as e:
            logger.error(f"Building model infos failed with error {e}")
            raise
    vllm_log_dir = get_log_dir("vllm")
    os.makedirs(vllm_log_dir, exist_ok=True)
    _setup_vllm_file_logging(f"{vllm_log_dir}/vllm_{name}.log")
    snapshots = glob(
        f"{cache_dir}/hub/models--{model.replace('/', '--')}/snapshots/*"
    )
    if len(snapshots) > 0:
        logger.info(f"model: {snapshots[0]}")
    else:
        logger.error(f"No snapshots found in {cache_dir}/hub.")
        raise RuntimeError("No snapshots found.")
    
    return snapshots

def _create_engine(logger, snapshots, llm_kwargs: Dict[str, Any] = {}, async_engine: bool = False):
    
    try:
        if async_engine:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            engine_args = AsyncEngineArgs(
                model=snapshots[0],
                trust_remote_code=True,
                **llm_kwargs,
            )
            engine = AsyncLLMEngine.from_engine_args(engine_args)
        else:
            from vllm import LLM
            engine = LLM(
                model=snapshots[0],
                trust_remote_code=True,
                **llm_kwargs,
            )
        return engine
    except Exception as e:
        logger.error(f"Starting LLM failed with Exception: {e}")
        raise RuntimeError(str(e))

def _create_online_engine(logger, snapshots, server_args: Dict = {}, engine_cls=None):

    from vllm.v1.engine.core import EngineCoreProc
    EngineCoreProc.run_engine_core = staticmethod(_spmd_run_engine_core)
    logger.info("Monkey patch successful")
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser


    if engine_cls is None:
        engine_cls = AsyncLLMEngine

    cli_overrides = []
    for key, value in server_args.items():
        arg_key = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cli_overrides.append(arg_key)
        elif isinstance(value, list):
            cli_overrides.append(arg_key)
            cli_overrides.extend(str(v) for v in value)
        else:
            cli_overrides.extend([arg_key, str(value)])

    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)
    args = parser.parse_args(
        ["--model", snapshots[0], "--trust-remote-code"] + cli_overrides)

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = engine_cls.from_engine_args(engine_args)

    return args, engine

    

class _VLLMOfflineMixin:
    def _init_vllm(
        self,
        model: str,
        cache_dir: str,
        use_cached_modelinfo: bool = False,
        model_info_cache: Optional[str] = None,
        llm_kwargs: Dict[str, Any] = {},
        async_engine: bool = False,
    ):
        self.model = model
        self.cache_dir = cache_dir
        self._engine = None
        self._async_engine = async_engine
        self._use_cached_modelinfo = use_cached_modelinfo
        self._model_info_cache = model_info_cache
        self.llm_kwargs = llm_kwargs
        if self._use_cached_modelinfo:
            assert model_info_cache is not None, "model_info_cache can't be None"

    async def on_start(self):
        if self.logger is None:
            self.logger = setup_logger(name=self._name)
        self.logger.info(
            f"{socket.gethostname()}:{os.environ.get('ZE_AFFINITY_MASK', None)}"
        )
        snapshots = _setup_env(self._name, self.logger, self.model, self.cache_dir, 
                                self._use_cached_modelinfo, self._model_info_cache)
        if len(snapshots) == 0:
            self.logger.error("No model snapshots found.")
            raise FileNotFoundError
        
        if self._engine is None:
            self._engine = _create_engine(self.logger, snapshots, self.llm_kwargs, self._async_engine)
        self.logger.info("init done!")

    @action
    async def generate(
        self,
        prompts: Union[str, List],
        sampling_params: Dict = {"temperature": 0.0, "max_tokens": 1024},
        **kwargs,
    ):
        from vllm import SamplingParams

        if isinstance(prompts, str):
            prompts = [prompts]
            single = True
        else:
            single = False

        if self._async_engine:
            from vllm.sampling_params import RequestOutputKind
            from vllm.utils import random_uuid

            sp = SamplingParams(
                **sampling_params, output_kind=RequestOutputKind.FINAL_ONLY
            )

            async def _single_generate(prompt):
                final = None
                async for out in self._engine.generate(
                    prompt, sp, request_id=random_uuid()
                ):
                    final = out
                return final.outputs[0].text

            results = await asyncio.gather(*[_single_generate(p) for p in prompts])
        else:
            sp = SamplingParams(**sampling_params)
            outputs = self._engine.generate(prompts, sampling_params=sp, **kwargs)
            results = [output.outputs[0].text for output in outputs]

        return results[0] if single else list(results)

    async def on_stop(self):
        if self._engine is not None:
            if self._async_engine:
                self._engine.shutdown()
            else:
                self._engine.llm_engine.engine_core.shutdown()
                del self._engine
                self._engine = None


class VLLMInference(_VLLMOfflineMixin, PublicActor):
    def __init__(
        self,
        name: str,
        model: str,
        cache_dir: str,
        transport: str = "zmq",
        use_cached_modelinfo: bool = False,
        model_info_cache: Optional[str] = None,
        ckpt_dir: str = f"{os.getcwd()}/.actor_ckpt",
        llm_kwargs: Dict = {"max_model_len": 2048, "tensor_parallel_size": 1,},
        async_engine: bool = False,
        max_workers: int = 1,
        **kwargs,
    ):
        if async_engine:
            _max_workers = max_workers
        else:
            _max_workers = 1
        PublicActor.__init__(
            self, name, transport, ckpt_dir=ckpt_dir, max_workers=_max_workers, run_in_executor=False, **kwargs
        )
        self._init_vllm(
            model,
            cache_dir,
            use_cached_modelinfo=use_cached_modelinfo,
            model_info_cache=model_info_cache,
            llm_kwargs=llm_kwargs,
            async_engine=async_engine,
        )


class PrivateVLLMInference(_VLLMOfflineMixin, PrivateActor):
    def __init__(
        self,
        name: str,
        model: str,
        cache_dir: str,
        client_conn: ClientConnection,
        use_cached_modelinfo: bool = False,
        model_info_cache: Optional[str] = None,
        llm_kwargs: Dict = {"max_model_len": 2048, "tensor_parallel_size" : 1,},
        async_engine: bool = False,
        max_workers: int = 1,
        **kwargs,
    ):
        if async_engine:
            _max_workers = max_workers
        else:
            _max_workers = 1
        PrivateActor.__init__(
            self, name, client_conn, max_workers=_max_workers, run_in_executor=False, **kwargs
        )
        self._init_vllm(
            model,
            cache_dir,
            use_cached_modelinfo=use_cached_modelinfo,
            model_info_cache=model_info_cache,
            llm_kwargs=llm_kwargs,
            async_engine=async_engine,
        )


# ---------------------------------------------------------------------------
# Mixin 2: Online vLLM server (in-process uvicorn + FastAPI)
# ---------------------------------------------------------------------------


class _VLLMOnlineMixin:
    def _init_vllm_online(
        self,
        model: str,
        cache_dir: str,
        use_cached_modelinfo: bool = False,
        model_info_cache: Optional[str] = None,
        server_args: Dict = {},
    ):
        self.model = model
        self.cache_dir = cache_dir
        self._engine = None
        self._args = None
        self._serve_task = None
        self._use_cached_modelinfo = use_cached_modelinfo
        self._model_info_cache = model_info_cache
        self.server_args = server_args
        self.server_args["served_model_name"] = self.model
        if self._use_cached_modelinfo:
            assert model_info_cache is not None, "model_info_cache can't be None"

    async def on_start(self):
        if self.logger is None:
            self.logger = setup_logger(name=self._name)
        self.logger.info(
            f"{socket.gethostname()}:{os.environ.get('ZE_AFFINITY_MASK', None)}"
        )
        snapshots = _setup_env(self._name, self.logger, self.model, self.cache_dir,
                           self._use_cached_modelinfo, self._model_info_cache)
        
        if len(snapshots) == 0:
            self.logger.error("No model snapshots found.")
            raise FileNotFoundError
        
        if self._engine is None:
            self._args, self._engine = _create_online_engine(self.logger, snapshots, self.server_args)

        self._hostname = (
            socket.gethostname()
            if ".local" not in socket.gethostname()
            else "localhost"
        )
        self.port = self._args.port

        from vllm.entrypoints.openai.api_server import build_and_serve
        from vllm.tool_parsers import ToolParserManager
        from vllm.reasoning import ReasoningParserManager

        if getattr(self._args, "tool_parser_plugin", None) and len(self._args.tool_parser_plugin) > 3:
            ToolParserManager.import_tool_parser(self._args.tool_parser_plugin)

        if getattr(self._args, "reasoning_parser_plugin", None) and len(self._args.reasoning_parser_plugin) > 3:
            ReasoningParserManager.import_reasoning_parser(self._args.reasoning_parser_plugin)

        self._serve_task = asyncio.create_task(
            build_and_serve(self._engine, None, None, self._args))

        self.logger.info(f"vLLM server ready at {self._hostname}:{self.port}")

    async def on_stop(self):
        if self._serve_task is not None:
            self._serve_task.cancel()
            self._serve_task = None
        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None

    @action
    def get_address(self):
        return f"{self._hostname}:{self.port}"

    @action
    def model(self):
        return self.model


class OnlineVLLMInference(_VLLMOnlineMixin, PublicActor):
    def __init__(
        self,
        name: str,
        model: str,
        cache_dir: str,
        server_args: Dict = {},
        transport: str = "zmq",
        use_cached_modelinfo: bool = False,
        model_info_cache: Optional[str] = None,
        ckpt_dir: str = f"{os.getcwd()}/.actor_ckpt",
        **kwargs,
    ):
        PublicActor.__init__(self, name, transport, ckpt_dir=ckpt_dir, max_workers=1, run_in_executor=False, **kwargs)
        self._init_vllm_online(
            model, cache_dir,
            use_cached_modelinfo=use_cached_modelinfo,
            model_info_cache=model_info_cache,
            server_args=server_args,
        )


class PrivateOnlineVLLMInference(_VLLMOnlineMixin, PrivateActor):
    def __init__(
        self,
        name: str,
        model: str,
        cache_dir: str,
        client_conn: ClientConnection,
        server_args: Dict = {},
        use_cached_modelinfo: bool = False,
        model_info_cache: Optional[str] = None,
        **kwargs,
    ):
        PrivateActor.__init__(self, name, client_conn, max_workers=1, run_in_executor=False, **kwargs)
        self._init_vllm_online(
            model, cache_dir,
            use_cached_modelinfo=use_cached_modelinfo,
            model_info_cache=model_info_cache,
            server_args=server_args,
        )


# ---------------------------------------------------------------------------
# Mixin 3: Multi-node vLLM with PUB/SUB rank coordination
# ---------------------------------------------------------------------------


def _bind_zmq_socket(zmq_context, zmq_type, hostname, port_range, logger, label, AsyncSocket):
    import zmq as _zmq
    sock = zmq_context.socket(zmq_type, socket_class=AsyncSocket)
    port = find_free_port(port_range, host=hostname)
    if port is None:
        port = "0"
        logger.warning(f"Couldn't find any free port for {label}, using 0")
    address = f"{hostname}:{port}"
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            sock.bind(f"tcp://{address}")
            break
        except _zmq.error.ZMQError as e:
            if "Address already in use" in str(e) and attempt < max_attempts - 1:
                port = random.randint(30000, 40000)
                address = f"{hostname}:{port}"
            else:
                raise
    logger.info(f"{label} socket bound to {address}")
    return sock, address


def _create_sockets(rank, logger, sync_location, sync_timeout, ckpt_dir):
    import zmq
    from zmq.asyncio import Context as AsyncContext
    from zmq.asyncio import Socket as AsyncSocket

    hostname = get_hsn_ip_cli() or socket.gethostname()

    if rank == 0:
        os.environ["MASTER_ADDR"] = hostname
        os.environ["MASTER_PORT"] = str(random.randint(20000, 30000))
        zmq_context = AsyncContext()

        pub_socket, pub_address = _bind_zmq_socket(
            zmq_context, zmq.PUB, hostname, (10000, 30000), logger, "PUB", AsyncSocket)
        pull_socket, pull_address = _bind_zmq_socket(
            zmq_context, zmq.PULL, hostname, (10000, 30000), logger, "PULL", AsyncSocket)

        logger.info(f"Sync location: {sync_location}")
        try:
            os.makedirs(ckpt_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", dir=ckpt_dir, delete=False
            ) as temp:
                temp.write(
                    f"{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}\n"
                    f"{pub_address}\n"
                    f"{pull_address}\n"
                )
                temp.flush()
                os.fsync(temp.fileno())
                temp_path = temp.name
        except Exception as e:
            logger.error(f"Writing temp file failed with exception {e}")
            raise e

        try:
            os.replace(temp_path, sync_location)
            logger.info(f"Wrote the sync file to {sync_location}")
        except Exception as e:
            logger.error(f"os.replace failed with error: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

        return zmq_context, pub_socket, pull_socket
    else:
        start = time.perf_counter()
        while time.perf_counter() - start < sync_timeout:
            if os.path.exists(sync_location):
                break
            time.sleep(1.0)
        try:
            with open(sync_location, "r") as f:
                lines = f.read().strip().splitlines()
        except Exception as e:
            logger.error(f"Reading sync location failed with Exception {e}")
            raise
        master_addr, master_port = lines[0].split(":")
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        pub_address = lines[1]
        pull_address = lines[2]

        zmq_context = AsyncContext()
        sub_socket = zmq_context.socket(zmq.SUB, socket_class=AsyncSocket)
        sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
        sub_socket.connect(f"tcp://{pub_address}")
        logger.info(f"SUB socket connected to {pub_address}")

        push_socket = zmq_context.socket(zmq.PUSH, socket_class=AsyncSocket)
        push_socket.connect(f"tcp://{pull_address}")
        logger.info(f"PUSH socket connected to {pull_address}")

        return zmq_context, sub_socket, push_socket

async def _set_multinode_env(model,
                            cache_dir,
                            ckpt_dir,
                            use_cached_modelinfo,
                            model_info_cache,
                            name,
                            local_rank, 
                            rank, 
                            logger, 
                            tensor_parallel_size, 
                            pipeline_parallel_size, 
                            gpu_selector,
                            zmq_socket):
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(
        tensor_parallel_size * pipeline_parallel_size
    )
    logger.info("after setting local rank")
    os.environ.pop(gpu_selector, None)
    os.environ[gpu_selector] = os.environ.get(
        f"AVAILABLE_GPUS_{socket.gethostname()}",
        os.environ.get(
            "AVAILABLE_GPUS",
            ",".join(map(str, list(range(tensor_parallel_size)))),
        ),
    )
    logger.info(f"{gpu_selector}: {os.environ[gpu_selector]}")
    vllm_log_dir = get_log_dir("vllm")
    os.makedirs(vllm_log_dir, exist_ok=True)
    _setup_vllm_file_logging(
        f"{vllm_log_dir}/vllm_{name}_rank{rank}.log"
    )
    import vllm.envs as envs  # isort: skip
    import torch  # isort: skip
    torch.xpu.set_device(local_rank)
    _build_success = b"0"
    _build_failed = b"1"
    if use_cached_modelinfo:
        os.environ["VLLM_CACHE_ROOT"] = model_info_cache
    else:
        if rank == 0:
            if model_info_cache is None:
                model_info_cache = os.path.join(
                    ckpt_dir, f"vllm_cache_{uuid.uuid4().hex[:6]}"
                )
            os.environ["VLLM_CACHE_ROOT"] = model_info_cache
            try:
                os.makedirs(model_info_cache)
                _build_model_cache(logger=logger.getChild("BuildModelInfo"))
                await zmq_socket.send(_build_success)
                await zmq_socket.send(
                    cloudpickle.dumps(model_info_cache)
                )
            except Exception as e:
                await zmq_socket.send(_build_failed)
                logger.error(f"Building model infos failed with error {e}")
                raise
        else:
            success = await zmq_socket.recv()
            if success == _build_failed:
                raise
            else:
                frame = await zmq_socket.recv()
                model_info_cache = cloudpickle.loads(frame)
        os.environ["VLLM_CACHE_ROOT"] = model_info_cache

    snapshots = glob(
            f"{cache_dir}/hub/models--{model.replace('/', '--')}/snapshots/*"
        )
    return snapshots

class _MultiNodeVLLMMixin:
    def _init_multinode_vllm(
        self,
        model: str,
        cache_dir: str,
        ckpt_dir: str,
        use_cached_modelinfo: bool = False,
        model_info_cache: Optional[str] = None,
        sync_location: Optional[str] = None,
        rank_env: str = "PALS_RANKID",
        local_rank_env: str = "PALS_LOCAL_RANKID",
        sync_timeout: float = 60,
        llm_kwargs: Optional[Dict] = None,
        async_engine: bool = False,
        gpu_selector: str = "ZE_AFFINITY_MASK",
    ):
        self._model_name = model
        self.cache_dir = cache_dir
        self._ckpt_dir = ckpt_dir
        self.tensor_parallel_size = llm_kwargs.get("tensor_parallel_size",1) if llm_kwargs is not None else 1
        self.pipeline_parallel_size = llm_kwargs.get("pipeline_parallel_size",1) if llm_kwargs is not None else 1
        self._async_engine = async_engine
        self._use_cached_modelinfo = use_cached_modelinfo
        self._model_info_cache = model_info_cache
        if self._use_cached_modelinfo:
            assert self._model_info_cache is not None, "model_info_cache can't be None"
        self.rank_env = rank_env
        self.local_rank_env = local_rank_env
        self.sync_timeout = sync_timeout

        if sync_location is None:
            sync_location = f"file://file_{uuid.uuid4().hex}"

        if sync_location.startswith("file://"):
            self.sync_location = os.path.join(
                self._ckpt_dir, sync_location.replace("file://", "")
            )
        else:
            raise ValueError("Unknown sync location prefix")
        self._rank = None
        self._local_rank = None
        self._pub_socket = None
        self._sub_socket = None
        self._zmq_context = None
        self._engine = None
        self.llm_kwargs = llm_kwargs
        self.gpu_selector = gpu_selector

    async def on_start(self):
        if self.logger is None:
            self.logger = setup_logger(name=self._name)

        os.environ["TMPDIR"] = "/tmp"

        try:
            self._local_rank = int(os.getenv(self.local_rank_env))
            self._rank = int(os.getenv(self.rank_env))
        except (TypeError, ValueError) as e:
            self.logger.error(
                f"Failed to read rank env vars "
                f"({self.rank_env}, {self.local_rank_env}): {e}"
            )
            raise RuntimeError(
                f"Rank environment variables not set: "
                f"{self.rank_env}, {self.local_rank_env}"
            ) from e

        self._zmq_context, sock_a, sock_b = _create_sockets(
            self._rank, self.logger, self.sync_location,
            self.sync_timeout, self._ckpt_dir)
        sock_b.close()
        if self._rank == 0:
            self._pub_socket = sock_a
        else:
            self._sub_socket = sock_a

        zmq_socket = self._pub_socket if self._rank == 0 else self._sub_socket

        snapshots = await _set_multinode_env(self._model_name,
                                       self.cache_dir,
                                       self._ckpt_dir,
                                       self._use_cached_modelinfo,
                                       self._model_info_cache,
                                       self._name, self._local_rank,
                                       self._rank,
                                       self.logger,
                                       self.tensor_parallel_size,
                                       self.pipeline_parallel_size,
                                       self.gpu_selector,
                                       zmq_socket)
        if len(snapshots) == 0:
            self.logger.error(f"No snapshots found in {self.cache_dir}")
            raise FileNotFoundError

        self.logger.info(f"model: {snapshots[0]}")
        self.logger.info(
            f"{self._rank},{self._local_rank},{os.environ[self.gpu_selector]}"
        )
        try:
            if self._async_engine:
                from vllm.engine.arg_utils import AsyncEngineArgs
                from vllm.engine.async_llm_engine import AsyncLLMEngine

                engine_args = AsyncEngineArgs(
                    model=snapshots[0],
                    trust_remote_code=True,
                    distributed_executor_backend="external_launcher",
                    seed=1,
                    **self.llm_kwargs,
                )
                self._engine = AsyncLLMEngine.from_engine_args(engine_args)
            else:
                from vllm import LLM

                self._engine = LLM(
                    model=snapshots[0],
                    trust_remote_code=True,
                    distributed_executor_backend="external_launcher",
                    seed=1,
                    **self.llm_kwargs,
                )
        except Exception as e:
            self.logger.error(f"Starting LLM failed with Exception: {e}")
            raise RuntimeError(str(e))
        self.logger.info(f"Rank {self._rank}: vLLM init done!")

    async def _setup_rank0_connection(self):
        raise NotImplementedError

    async def _run(self):
        self._init_runtime()
        await self.on_start()

        if self._rank == 0:
            await self._setup_rank0_connection()

        signal_ready = getattr(self, "_signal_ready", None)
        if callable(signal_ready):
            asyncio.create_task(self._signal_ready())

        await asyncio.gather(self._recv(), self._send(), self._main_loop())

        await self.on_stop()
        if self._conn is not None:
            await self._conn.close()

    async def _recv(self):
        if self._rank == 0:
            self.logger.info("Rank 0: Receive loop started (ROUTER + PUB forward).")
            while not self._stop.is_set():
                try:
                    frames = await asyncio.wait_for(self._conn.recv(), timeout=5.0)
                    sender_id = self._extract_sender(frames)
                    self.logger.info(f"Received args from {sender_id}")
                    args = cloudpickle.loads(frames[1])
                    await self._pub_socket.send(frames[1])
                    await self._input_queue.put((sender_id, args))
                except Exception:
                    pass
        else:
            self.logger.info(f"Rank {self._rank}: Receive loop started (SUB).")
            while not self._stop.is_set():
                try:
                    raw = await asyncio.wait_for(self._sub_socket.recv(), timeout=5.0)
                    args = cloudpickle.loads(raw)
                    await self._input_queue.put((None, args))
                except Exception:
                    pass

    async def _send(self):
        if self._rank == 0:
            await super()._send()
        else:
            self.logger.info(f"Rank {self._rank}: Send loop (drain-only).")
            while not self._stop.is_set():
                try:
                    await asyncio.wait_for(self._output_queue.get(), timeout=5.0)
                except Exception:
                    pass

    async def on_stop(self):
        if self._engine is not None:
            if self._async_engine:
                self._engine.shutdown()
            else:
                self._engine.llm_engine.engine_core.shutdown()
                del self._engine
                self._engine = None
        if self._pub_socket is not None:
            self._pub_socket.close()
        if self._sub_socket is not None:
            self._sub_socket.close()
        if self._zmq_context is not None:
            self._zmq_context.term()

    @action
    async def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Dict = {"temperature": 0.0, "max_tokens": 1024},
        **kwargs,
    ):
        from vllm import SamplingParams

        if isinstance(prompts, str):
            prompts = [prompts]
            single = True
        else:
            single = False

        self.logger.info(f"Invoking engine with {prompts} {sampling_params} {kwargs}")
        try:
            if self._async_engine:
                from vllm.sampling_params import RequestOutputKind
                from vllm.utils import random_uuid

                sp = SamplingParams(
                    **sampling_params, output_kind=RequestOutputKind.FINAL_ONLY
                )

                async def _single_generate(prompt):
                    final = None
                    async for out in self._engine.generate(
                        prompt, sp, request_id=random_uuid()
                    ):
                        final = out
                    return final.outputs[0].text

                results = await asyncio.gather(*[_single_generate(p) for p in prompts])
                results = list(results)
            else:
                sp = SamplingParams(**sampling_params)
                outputs = self._engine.generate(prompts, sampling_params=sp, **kwargs)
                results = [output.outputs[0].text for output in outputs]

            self.logger.info(f"Obtained result: {results}")

            return results[0] if single else results
        except Exception as e:
            self.logger.error(f"Engine generate failed with exception: {e}")


class MultiNodeVLLMInference(_MultiNodeVLLMMixin, PublicActor):
    def __init__(
        self,
        name: str,
        model: str,
        cache_dir: str,
        transport: str = "zmq",
        use_cached_modelinfo: bool = False,
        model_info_cache: Optional[str] = None,
        sync_location: Optional[str] = None,
        rank_env: str = "PALS_RANKID",
        local_rank_env: str = "PALS_LOCAL_RANKID",
        sync_timeout: float = 60,
        gpu_selector: str = "ZE_AFFINITY_MASK",
        llm_kwargs: Dict = {"max_model_len": 2048, "tensor_parallel_size": 1, "pipeline_parallel_size":1},
        async_engine: bool = False,
        max_workers: int = 1,
        **kwargs,
    ):
        if async_engine:
            _max_workers = max_workers
        else:
            _max_workers = 1
        PublicActor.__init__(self, name, transport, max_workers=_max_workers, run_in_executor=False, **kwargs)
        self._init_multinode_vllm(
            model,
            cache_dir,
            self.ckpt_dir,
            use_cached_modelinfo=use_cached_modelinfo,
            model_info_cache=model_info_cache,
            sync_location=sync_location,
            rank_env=rank_env,
            local_rank_env=local_rank_env,
            sync_timeout=sync_timeout,
            llm_kwargs=llm_kwargs,
            async_engine=async_engine,
        )

    async def _setup_rank0_connection(self):
        self._start_transport()
        await self._conn.open()
        os.makedirs(self.ckpt_dir, exist_ok=True)
        fname = f"{self.ckpt_dir}/{self.name}.ckpt"
        with open(fname, "w") as f:
            f.write(self._conn.get_state().serialize())
        self.logger.info("Rank 0: ROUTER transport ready, checkpoint written.")


class PrivateMultiNodeVLLMInference(_MultiNodeVLLMMixin, PrivateActor):
    def __init__(
        self,
        name: str,
        model: str,
        cache_dir: str,
        client_conn: ClientConnection,
        ckpt_dir: str = f"{os.getcwd()}/.actor_ckpt",
        use_cached_modelinfo: bool = False,
        model_info_cache: Optional[str] = None,
        sync_location: Optional[str] = None,
        rank_env: str = "PALS_RANKID",
        local_rank_env: str = "PALS_LOCAL_RANKID",
        sync_timeout: float = 60,
        llm_kwargs: Dict = {"max_model_len": 2048, "tensor_parallel_size": 1, "pipeline_parallel_size":1},
        async_engine: bool = False,
        max_workers: int = 1,
        **kwargs,
    ):
        if async_engine:
            _max_workers = max_workers
        else:
            _max_workers = 1
        PrivateActor.__init__(self, name, client_conn, max_workers=_max_workers, run_in_executor=False, **kwargs)
        self._init_multinode_vllm(
            model,
            cache_dir,
            ckpt_dir,
            use_cached_modelinfo=use_cached_modelinfo,
            model_info_cache=model_info_cache,
            sync_location=sync_location,
            rank_env=rank_env,
            local_rank_env=local_rank_env,
            sync_timeout=sync_timeout,
            llm_kwargs=llm_kwargs,
            async_engine=async_engine,
        )

    async def _setup_rank0_connection(self):
        await self._conn.open()
        self.logger.info("Rank 0: transport ready.")

# ---------------------------------------------------------------------------
# Mixin 4: Online Multi-Node vLLM server (in-process uvicorn + FastAPI of Rank 0. zmq sub on others)
# ---------------------------------------------------------------------------

class SPMDEngineCoreProc:
    def __init__(self, engine_core_proc, vllm_logger):
        self._engine_core = engine_core_proc
        self._spmd_rank = int(os.environ.get("SPMD_RANK", "0"))
        self._spmd_world_size = int(os.environ.get("SPMD_WORLD_SIZE", "1"))
        self._bcast_queue = []
        self._logger = vllm_logger

    def _spmd_handle_client_request(self, *args):
        if self._spmd_rank == 0:
            self._bcast_queue.append(args)
        self._engine_core._original_handle_client_request(*args)

    def run_busy_loop(self):
        import torch
        from torch import Tensor
        from vllm.distributed.parallel_state import get_world_group
        import cloudpickle

        cpu_group = get_world_group().cpu_group

        self._engine_core._original_handle_client_request = (
            self._engine_core._handle_client_request)
        self._engine_core._handle_client_request = (
            self._spmd_handle_client_request)

        nitems = torch.zeros(size=(1,),dtype=torch.int32)
        while True:
            if self._spmd_rank == 0:
                self._bcast_queue = []
                self._engine_core._process_input_queue()
                nitems[0] = len(self._bcast_queue)
                if len(self._bcast_queue)>0:
                    items = [cloudpickle.dumps(self._bcast_queue)]
            else:
                items = [None]

            try:
                torch.distributed.broadcast(nitems, src=0, group=cpu_group)
                if nitems[0] > 0:
                    self._logger.info(f"Ranks {self._spmd_rank}: Starting torch bcast for {len(self._bcast_queue)}")
                    torch.distributed.broadcast_object_list(
                        items, src=0, group=cpu_group)
            except Exception as e:
                self._logger.error(f"Bcast failed with exception: {e}")

            if self._spmd_rank != 0:
                while not self._engine_core.input_queue.empty():
                    try:
                        self._engine_core.input_queue.get_nowait()
                    except Exception:
                        break
                if nitems[0] > 0:
                    for obj_bytes in items:
                        obj = cloudpickle.loads(obj_bytes)
                        for o in obj:
                            self._engine_core.input_queue.put_nowait(o)
                self._engine_core._process_input_queue()

            self._engine_core._process_engine_step()

    def __setattr__(self, name, value):
        if name in ("_engine_core", "_spmd_rank", "_spmd_world_size",
                     "_bcast_queue"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._engine_core, name, value)

    def __getattr__(self, name):
        return getattr(self._engine_core, name)

def _spmd_run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):

    import signal
    from vllm.config import VllmConfig, ParallelConfig
    from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
    from vllm.utils.system_utils import decorate_logs, set_process_title
    from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc
    from vllm.v1.engine.core import logger as vllm_logger

    # Signal handler used for graceful termination.
    # SystemExit exception is only raised once to allow this and worker
    # processes to terminate without error
    shutdown_requested = False
    # Ensure we can serialize transformer config after spawning
    maybe_register_config_serialize_by_value()
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            raise SystemExit()
    # Either SIGTERM or SIGINT will terminate the engine_core
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    engine_core: SPMDEngineCoreProc = None
    try:
        vllm_config: VllmConfig = kwargs["vllm_config"]
        parallel_config: ParallelConfig = vllm_config.parallel_config
        data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
        if data_parallel:
            parallel_config.data_parallel_rank_local = local_dp_rank
            set_process_title("EngineCore", f"DP{dp_rank}")
        else:
            set_process_title("EngineCore")
        decorate_logs()
        if data_parallel and vllm_config.kv_transfer_config is not None:
            # modify the engine_id and append the local_dp_rank to it to ensure
            # that the kv_transfer_config is unique for each DP rank.
            vllm_config.kv_transfer_config.engine_id = (
                f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
            )
            # logger.debug(
            #     "Setting kv_transfer_config.engine_id to %s",
            #     vllm_config.kv_transfer_config.engine_id,
            # )
        parallel_config.data_parallel_index = dp_rank
        if data_parallel and vllm_config.model_config.is_moe:
            # Set data parallel rank for this engine process.
            parallel_config.data_parallel_rank = dp_rank
            engine_core = DPEngineCoreProc(*args, **kwargs)
        else:
            # Non-MoE DP ranks are completely independent, so treat like DP=1.
            # Note that parallel_config.data_parallel_index will still reflect
            # the original DP rank.
            parallel_config.data_parallel_size = 1
            parallel_config.data_parallel_size_local = 1
            parallel_config.data_parallel_rank = 0
            engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
            engine_core = SPMDEngineCoreProc(engine_core, vllm_logger)
        engine_core.run_busy_loop()
    except SystemExit:
        # logger.debug("EngineCore exiting.")
        raise
    except Exception as e:
        # if engine_core is None:
        #     # logger.exception("EngineCore failed to start.")
        # else:
        #     logger.exception("EngineCore encountered a fatal error.")
        #     engine_core._send_engine_dead()
        raise e
    finally:
        if engine_core is not None:
            engine_core.shutdown()

# def _spmd_run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
#     import signal
#     from vllm.config import VllmConfig, ParallelConfig
#     from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
#     from vllm.utils.system_utils import decorate_logs, set_process_title
#     from vllm.utils import numa_utils
#     from vllm.tracing import maybe_init_worker_tracer
#     from vllm.v1.engine.core import (
#         EngineCoreProc, EngineCoreRequestType, EngineShutdownState,
#         SignalCallback,
#     )

#     maybe_register_config_serialize_by_value()

#     engine_core = None
#     signal_callback = None
#     try:
#         vllm_config: VllmConfig = kwargs["vllm_config"]
#         parallel_config: ParallelConfig = vllm_config.parallel_config
#         data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
#         if data_parallel:
#             parallel_config.data_parallel_rank_local = local_dp_rank
#             process_title = f"EngineCore_DP{dp_rank}"
#         else:
#             process_title = "EngineCore"
#         set_process_title(process_title)
#         maybe_init_worker_tracer("vllm.engine_core", "engine_core", process_title)
#         decorate_logs()
#         if parallel_config.numa_bind:
#             numa_utils.log_current_affinity_state(process_title)

#         if data_parallel and vllm_config.kv_transfer_config is not None:
#             vllm_config.kv_transfer_config.engine_id = (
#                 f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
#             )

#         parallel_config.data_parallel_index = dp_rank
#         parallel_config.data_parallel_size = 1
#         parallel_config.data_parallel_size_local = 1
#         parallel_config.data_parallel_rank = 0
#         raw_engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)

#         engine_core = SPMDEngineCoreProc(raw_engine_core)

#         def wakeup_engine():
#             engine_core.input_queue.put_nowait(
#                 (EngineCoreRequestType.WAKEUP, None))

#         signal_callback = SignalCallback(wakeup_engine)

#         def signal_handler(signum, frame):
#             engine_core.shutdown_state = EngineShutdownState.REQUESTED
#             signal_callback.trigger()

#         signal.signal(signal.SIGTERM, signal_handler)
#         signal.signal(signal.SIGINT, signal_handler)

#         engine_core.spmd_run_busy_loop()

#     except SystemExit:
#         raise
#     except Exception as e:
#         if engine_core is None:
#             import logging
#             logging.getLogger("vllm").exception("EngineCore failed to start.")
#         else:
#             import logging
#             logging.getLogger("vllm").exception(
#                 "EngineCore encountered a fatal error.")
#             engine_core._send_engine_dead()
#         raise e
#     finally:
#         signal.signal(signal.SIGTERM, signal.SIG_DFL)
#         signal.signal(signal.SIGINT, signal.SIG_DFL)
#         if signal_callback is not None:
#             signal_callback.stop()
#         if engine_core is not None:
#             engine_core.shutdown()
    
class _MultiNodeOnlineVLLMMixin:
    def _init_multinode_vllm(
        self,
        model: str,
        cache_dir: str,
        ckpt_dir: str,
        use_cached_modelinfo: bool = False,
        model_info_cache: Optional[str] = None,
        sync_location: Optional[str] = None,
        rank_env: str = "PALS_RANKID",
        local_rank_env: str = "PALS_LOCAL_RANKID",
        sync_timeout: float = 60,
        server_kwargs: Optional[Dict] = None,
        gpu_selector: str = "ZE_AFFINITY_MASK",
    ):
        self._model_name = model
        self.cache_dir = cache_dir
        self._ckpt_dir = ckpt_dir
        self.tensor_parallel_size = server_kwargs.get("tensor_parallel_size",1) if server_kwargs is not None else 1
        self.pipeline_parallel_size = server_kwargs.get("pipeline_parallel_size",1) if server_kwargs is not None else 1
        self._use_cached_modelinfo = use_cached_modelinfo
        self._model_info_cache = model_info_cache
        if self._use_cached_modelinfo:
            assert self._model_info_cache is not None, "model_info_cache can't be None"
        self.rank_env = rank_env
        self.local_rank_env = local_rank_env
        self.sync_timeout = sync_timeout

        if sync_location is None:
            sync_location = f"file://file_{uuid.uuid4().hex}"

        if sync_location.startswith("file://"):
            self.sync_location = os.path.join(
                self._ckpt_dir, sync_location.replace("file://", "")
            )
        else:
            raise ValueError("Unknown sync location prefix")
        self._world_size = self.tensor_parallel_size * self.pipeline_parallel_size
        self._rank = None
        self._local_rank = None
        self._pub_socket = None
        self._sub_socket = None
        self._zmq_context = None
        self._engine = None
        self._serve_task = None
        self.server_kwargs = server_kwargs
        self.server_kwargs["served_model_name"] = self._model_name
        self.gpu_selector = gpu_selector
        self._args = None

    async def on_start(self):
        if self.logger is None:
            self.logger = setup_logger(name=self._name)

        os.environ["TMPDIR"] = "/tmp"

        try:
            self._local_rank = int(os.getenv(self.local_rank_env))
            self._rank = int(os.getenv(self.rank_env))
        except (TypeError, ValueError) as e:
            self.logger.error(
                f"Failed to read rank env vars "
                f"({self.rank_env}, {self.local_rank_env}): {e}"
            )
            raise RuntimeError(
                f"Rank environment variables not set: "
                f"{self.rank_env}, {self.local_rank_env}"
            ) from e

        self._zmq_context, sock_a, sock_b = _create_sockets(
            self._rank, self.logger, self.sync_location,
            self.sync_timeout, self._ckpt_dir)
        if self._rank == 0:
            self._pub_socket = sock_a
        else:
            self._sub_socket = sock_a
        sock_b.close()

        # For _set_multinode_env, rank 0 uses PUB, non-rank-0 uses SUB
        zmq_socket = self._pub_socket if self._rank == 0 else self._sub_socket

        snapshots = await _set_multinode_env(self._model_name,
                                       self.cache_dir,
                                       self._ckpt_dir,
                                       self._use_cached_modelinfo,
                                       self._model_info_cache,
                                       self._name, self._local_rank,
                                       self._rank,
                                       self.logger,
                                       self.tensor_parallel_size,
                                       self.pipeline_parallel_size,
                                       self.gpu_selector,
                                       zmq_socket)
        if len(snapshots) == 0:
            self.logger.error(f"No snapshots found in {self.cache_dir}")
            raise RuntimeError

        self.logger.info(f"model: {snapshots[0]}")
        self.logger.info(
            f"{self._rank},{self._local_rank},{os.environ[self.gpu_selector]}"
        )
        try:
            os.environ["SPMD_RANK"] = str(self._rank)
            os.environ["SPMD_WORLD_SIZE"] = str(self._world_size)

            multinode_defaults = {
                "distributed_executor_backend": "external_launcher",
                "seed": 1,
            }
            merged_kwargs = {**multinode_defaults, **(self.server_kwargs or {})}
            self._args, raw_engine = _create_online_engine(
                self.logger, snapshots, merged_kwargs)

            self._engine = raw_engine

            if self._rank == 0:
                self._hostname = get_hsn_ip_cli() or socket.gethostname()
                self.port = self._args.port
                self._args.host = self._hostname

                try:
                    from vllm.entrypoints.openai.api_server import build_and_serve
                except Exception:
                    import vllm.envs as envs  # isort: skip
                    from vllm.entrypoints.openai.api_server import build_app, init_app_state, serve_http

                    async def build_and_serve(engine_client, listen_address, sock, args, **uvicorn_kwargs):
                        app = build_app(args)
                        await init_app_state(engine_client, app.state, args)
                        return await serve_http(
                                        app,
                                        sock=sock,
                                        enable_ssl_refresh=args.enable_ssl_refresh,
                                        host=args.host,
                                        port=args.port,
                                        log_level=args.uvicorn_log_level,
                                        access_log=not args.disable_uvicorn_access_log,
                                        timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
                                        ssl_keyfile=args.ssl_keyfile,
                                        ssl_certfile=args.ssl_certfile,
                                        ssl_ca_certs=args.ssl_ca_certs,
                                        ssl_cert_reqs=args.ssl_cert_reqs,
                                        ssl_ciphers=args.ssl_ciphers,
                                        h11_max_incomplete_event_size=args.h11_max_incomplete_event_size,
                                        h11_max_header_count=args.h11_max_header_count,
                                        **uvicorn_kwargs,
                                        )

                from vllm.tool_parsers import ToolParserManager
                from vllm.reasoning import ReasoningParserManager

                if getattr(self._args, "tool_parser_plugin", None) and len(self._args.tool_parser_plugin) > 3:
                    ToolParserManager.import_tool_parser(self._args.tool_parser_plugin)

                if getattr(self._args, "reasoning_parser_plugin", None) and len(self._args.reasoning_parser_plugin) > 3:
                    ReasoningParserManager.import_reasoning_parser(self._args.reasoning_parser_plugin)

                self._serve_task = asyncio.create_task(
                        build_and_serve(self._engine, None, None, self._args))
                def _serve_done(task):
                    if task.cancelled():
                        self.logger.info("serve task cancelled")
                    elif task.exception():
                        self.logger.error(f"serve task failed: {task.exception()}")
                    else:
                        self.logger.info("serve task finished")
                self._serve_task.add_done_callback(_serve_done)

        except Exception as e:
            self.logger.error(f"Starting LLM failed with Exception: {e}")
            raise RuntimeError(str(e))
        self.logger.info(f"Rank {self._rank}: vLLM init done!")

    async def _setup_rank0_connection(self):
        raise NotImplementedError

    async def _run(self):
        self._init_runtime()
        await self.on_start()

        if self._rank == 0:
            await self._setup_rank0_connection()
            signal_ready = getattr(self, "_signal_ready", None)
            if callable(signal_ready):
                asyncio.create_task(self._signal_ready())
            await asyncio.gather(self._recv(), self._send(), self._main_loop())
            await self.on_stop()
            if self._conn is not None:
                await self._conn.close()
        else:
            await self._sub_recv()
            await self.on_stop()

    async def _sub_recv(self):
        self.logger.info(f"Rank {self._rank}: SUB recv loop started.")
        while not self._stop.is_set():
            try:
                raw = await asyncio.wait_for(
                    self._sub_socket.recv(), timeout=5.0)
                msg = cloudpickle.loads(raw)
                if not isinstance(msg, tuple) or len(msg) < 1:
                    continue
                if msg[0] == "stop":
                    self._stop.set()
                    break
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                self.logger.error(f"Rank {self._rank}: SUB recv error: {e}")

    @action
    def get_address(self):
        return f"{self._hostname}:{self.port}"

    @action
    def get_model(self):
        return self._model_name

    async def on_stop(self):
        if self._serve_task is not None:
            self._serve_task.cancel()
            self._serve_task = None
        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None
        if self._pub_socket is not None:
            self._pub_socket.close()
        if self._sub_socket is not None:
            self._sub_socket.close()
        if self._zmq_context is not None:
            self._zmq_context.term()


class MultiNodeOnlineVLLMInference(_MultiNodeOnlineVLLMMixin, PublicActor):
    def __init__(
        self,
        name: str,
        model: str,
        cache_dir: str,
        transport: str = "zmq",
        use_cached_modelinfo: bool = False,
        model_info_cache: Optional[str] = None,
        sync_location: Optional[str] = None,
        rank_env: str = "PALS_RANKID",
        local_rank_env: str = "PALS_LOCAL_RANKID",
        sync_timeout: float = 60,
        gpu_selector: str = "ZE_AFFINITY_MASK",
        server_kwargs: Dict = {},
        max_workers: int = 1,
        **kwargs,
    ):
        PublicActor.__init__(self, name, transport, max_workers=max_workers, run_in_executor=False, **kwargs)
        self._init_multinode_vllm(
            model,
            cache_dir,
            self.ckpt_dir,
            use_cached_modelinfo=use_cached_modelinfo,
            model_info_cache=model_info_cache,
            sync_location=sync_location,
            rank_env=rank_env,
            local_rank_env=local_rank_env,
            sync_timeout=sync_timeout,
            server_kwargs=server_kwargs,
            gpu_selector=gpu_selector,
        )

    async def _setup_rank0_connection(self):
        self._start_transport()
        await self._conn.open()
        os.makedirs(self.ckpt_dir, exist_ok=True)
        fname = f"{self.ckpt_dir}/{self.name}.ckpt"
        with open(fname, "w") as f:
            f.write(self._conn.get_state().serialize())
        self.logger.info("Rank 0: ROUTER transport ready, checkpoint written.")


class PrivateMultiNodeOnlineVLLMInference(_MultiNodeOnlineVLLMMixin, PrivateActor):
    def __init__(
        self,
        name: str,
        model: str,
        cache_dir: str,
        client_conn: ClientConnection,
        ckpt_dir: str = f"{os.getcwd()}/.actor_ckpt",
        use_cached_modelinfo: bool = False,
        model_info_cache: Optional[str] = None,
        sync_location: Optional[str] = None,
        rank_env: str = "PALS_RANKID",
        local_rank_env: str = "PALS_LOCAL_RANKID",
        sync_timeout: float = 60,
        gpu_selector: str = "ZE_AFFINITY_MASK",
        server_kwargs: Dict = {},
        max_workers: int = 1,
        **kwargs,
    ):
        PrivateActor.__init__(self, name, client_conn, max_workers=max_workers, run_in_executor=False, **kwargs)
        self._init_multinode_vllm(
            model,
            cache_dir,
            ckpt_dir,
            use_cached_modelinfo=use_cached_modelinfo,
            model_info_cache=model_info_cache,
            sync_location=sync_location,
            rank_env=rank_env,
            local_rank_env=local_rank_env,
            sync_timeout=sync_timeout,
            server_kwargs=server_kwargs,
            gpu_selector=gpu_selector,
        )

    async def _setup_rank0_connection(self):
        await self._conn.open()
        self.logger.info("Rank 0: transport ready.")
