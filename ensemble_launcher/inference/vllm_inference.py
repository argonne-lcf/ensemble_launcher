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

import cloudpickle

from ensemble_launcher.comm.pipe import ClientConnection, get_hsn_ip_cli, find_free_port
from ensemble_launcher.ensemble.actor import PrivateActor, PublicActor, action
from ensemble_launcher.logging import get_log_dir, setup_logger

from .utils import _build_model_cache, _setup_vllm_file_logging


# ---------------------------------------------------------------------------
# Mixin 1: Offline (in-process) vLLM inference
# ---------------------------------------------------------------------------


class _VLLMOfflineMixin:
    def _init_vllm(
        self,
        model: str,
        cache_dir: str,
        use_cached_modelinfo: bool = False,
        model_info_cache: Optional[str] = None,
        llm_kwargs: Dict[str, Any] = {},
    ):
        self.model = model
        self.cache_dir = cache_dir
        self._engine = None
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
        if self._engine is None:
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
            if self._use_cached_modelinfo:
                os.environ["VLLM_CACHE_ROOT"] = self._model_info_cache
                self.logger.info(f"Reusing cache at {self._model_info_cache}")
                self.logger.debug("File list in cache")
                for root, dirs, files in os.walk(self._model_info_cache):
                    for file in files:
                        self.logger.debug(f"{file}")
            else:
                if self._model_info_cache is None:
                    self._model_info_cache = f"/tmp/vllm_cache_{uuid.uuid4().hex[:6]}"
                os.environ["VLLM_CACHE_ROOT"] = self._model_info_cache
                try:
                    os.makedirs(os.environ["VLLM_CACHE_ROOT"])
                    _build_model_cache(logger=self.logger.getChild("BuildModelInfo"))
                except Exception as e:
                    self.logger.error(f"Building model infos failed with error {e}")
                    raise
            vllm_log_dir = get_log_dir("vllm")
            os.makedirs(vllm_log_dir, exist_ok=True)
            _setup_vllm_file_logging(f"{vllm_log_dir}/vllm_{self._name}.log")
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine

            snapshots = glob(
                f"{self.cache_dir}/hub/models--{self.model.replace('/', '--')}/snapshots/*"
            )
            if len(snapshots) > 0:
                self.logger.info(f"model: {snapshots[0]}")
            else:
                self.logger.error(f"No snapshots found in {self.cache_dir}/hub.")
                raise RuntimeError("No snapshots found.")
            try:
                engine_args = AsyncEngineArgs(
                    model=snapshots[0],
                    trust_remote_code=True,
                    **self.llm_kwargs,
                )
                self._engine = AsyncLLMEngine.from_engine_args(engine_args)
            except Exception as e:
                self.logger.error(f"Starting LLM failed with Exception: {e}")
                raise RuntimeError(str(e))
            self.logger.info("init done!")

    @action
    async def generate(
        self,
        prompts: Union[str, List],
        sampling_params: Dict = {"temperature": 0.0, "max_tokens": 1024},
        **kwargs,
    ):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        from vllm.utils import random_uuid

        sp = SamplingParams(
            **sampling_params, output_kind=RequestOutputKind.FINAL_ONLY
        )

        if isinstance(prompts, str):
            prompts = [prompts]
            single = True
        else:
            single = False

        async def _single_generate(prompt):
            final = None
            async for out in self._engine.generate(
                prompt, sp, request_id=random_uuid()
            ):
                final = out
            return final.outputs[0].text

        results = await asyncio.gather(*[_single_generate(p) for p in prompts])
        return results[0] if single else list(results)

    async def on_stop(self):
        if self._engine is not None:
            self._engine.shutdown()


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
        **kwargs,
    ):
        PublicActor.__init__(
            self, name, transport, ckpt_dir=ckpt_dir, max_workers=1, run_in_executor=False, **kwargs
        )
        self._init_vllm(
            model,
            cache_dir,
            use_cached_modelinfo=use_cached_modelinfo,
            model_info_cache=model_info_cache,
            llm_kwargs = llm_kwargs,
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
        **kwargs,
    ):
        PrivateActor.__init__(
            self, name, client_conn, max_workers=1, run_in_executor=False, **kwargs
        )
        self._init_vllm(
            model,
            cache_dir,
            use_cached_modelinfo=use_cached_modelinfo,
            model_info_cache=model_info_cache,
            llm_kwargs=llm_kwargs,
        )


# ---------------------------------------------------------------------------
# Mixin 2: Online vLLM server (subprocess)
# ---------------------------------------------------------------------------


class _VLLMOnlineMixin:
    def _init_vllm_online(
        self,
        model: str,
        cache_dir: str,
        port: int = 8000,
        tensor_parallel_size: int = 1,
        use_cached_modelinfo: bool = False,
        model_info_cache: Optional[str] = None,
        **kwargs,
    ):
        self._model_name = model
        self.cache_dir = cache_dir
        self.port = port
        self.tensor_parallel_size = tensor_parallel_size
        self._server_process = None
        self._use_cached_modelinfo = use_cached_modelinfo
        self._model_info_cache = model_info_cache
        if self._use_cached_modelinfo:
            assert model_info_cache is not None, "model_info_cache can't be None"
        self.kwargs = kwargs

    async def on_start(self):
        if self.logger is None:
            self.logger = setup_logger(name=self._name)
        script_path = os.path.join(os.path.dirname(__file__), "start_vllm_server.sh")
        self._hostname = (
            socket.gethostname()
            if ".local" not in socket.gethostname()
            else "localhost"
        )

        if self._use_cached_modelinfo:
            os.environ["VLLM_CACHE_ROOT"] = self._model_info_cache
        else:
            if self._model_info_cache is None:
                self._model_info_cache = f"/tmp/vllm_cache_{uuid.uuid4().hex[:6]}"
            os.environ["VLLM_CACHE_ROOT"] = self._model_info_cache
            try:
                os.makedirs(os.environ["VLLM_CACHE_ROOT"])
                _build_model_cache(logger=self.logger.getChild("BuildModelInfo"))
            except Exception as e:
                self.logger.error(f"Building model infos failed with error {e}")
                raise
        self._server_process = subprocess.Popen(
            [
                script_path,
                self._hostname,
                str(self.port),
                str(self.tensor_parallel_size),
                self._model_name,
                self.cache_dir,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.logger.info(
            f"Started vLLM server process (pid={self._server_process.pid})"
        )
        url = f"http://{self._hostname}:{self.port}/v1/models"
        start = time.time()
        timeout = 600
        while time.time() - start < timeout:
            try:
                urllib.request.urlopen(url, timeout=5)
                self.logger.info(f"vLLM server ready at {self._hostname}:{self.port}")
                return
            except Exception:
                self.logger.info(f"Waiting for vLLM ({time.time() - start:.0f}s)...")
                time.sleep(10)
        raise RuntimeError(f"vLLM server not ready after {timeout}s")

    async def on_stop(self):
        if self._server_process is not None:
            subprocess.run(["pkill", "-f", "vllm serve *"])
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
                self._server_process.wait()

    @action
    def get_address(self):
        return f"{self._hostname}:{self.port}"

    @action
    def model(self):
        return self._model_name


class OnlineVLLMInference(_VLLMOnlineMixin, PublicActor):
    def __init__(
        self,
        name: str,
        model: str,
        cache_dir: str,
        port: int = 8000,
        tensor_parallel_size: int = 1,
        transport: str = "zmq",
        use_cached_modelinfo: bool = False,
        model_info_cache: Optional[str] = None,
        ckpt_dir: str = f"{os.getcwd()}/.actor_ckpt",
        llm_kwargs: Dict = {},
        **kwargs,
    ):
        PublicActor.__init__(self, name, transport, ckpt_dir=ckpt_dir, max_workers=1, run_in_executor=False, **kwargs)
        self._init_vllm_online(
            model,
            cache_dir,
            port,
            tensor_parallel_size,
            use_cached_modelinfo,
            model_info_cache,
            **llm_kwargs,
        )


class PrivateOnlineVLLMInference(_VLLMOnlineMixin, PrivateActor):
    def __init__(
        self,
        name: str,
        model: str,
        cache_dir: str,
        client_conn: ClientConnection,
        port: int = 8000,
        tensor_parallel_size: int = 1,
        use_cached_modelinfo: bool = False,
        model_info_cache: Optional[str] = None,
        llm_kwargs: Dict = {},
        **kwargs,
    ):
        PrivateActor.__init__(self, name, client_conn, max_workers=1, run_in_executor=False, **kwargs)
        self._init_vllm_online(
            model,
            cache_dir,
            port,
            tensor_parallel_size,
            use_cached_modelinfo,
            model_info_cache,
            **llm_kwargs,
        )


# ---------------------------------------------------------------------------
# Mixin 3: Multi-node vLLM with PUB/SUB rank coordination
# ---------------------------------------------------------------------------


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
        gpu_selector: str = "ZE_AFFINITY_MASK",
    ):
        self._model_name = model
        self.cache_dir = cache_dir
        self._ckpt_dir = ckpt_dir
        self.tensor_parallel_size = llm_kwargs.get("tensor_parallel_size",1) if llm_kwargs is not None else {}
        self.pipeline_parallel_size = llm_kwargs.get("pipeline_parallel_size",1) if llm_kwargs is not None else {}
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

        import zmq
        from zmq.asyncio import Context as AsyncContext
        from zmq.asyncio import Socket as AsyncSocket

        hostname = get_hsn_ip_cli() or socket.gethostname()

        if self._rank == 0:
            os.environ["MASTER_ADDR"] = hostname
            os.environ["MASTER_PORT"] = str(random.randint(20000, 30000))

            self._zmq_context = AsyncContext()
            self._pub_socket = self._zmq_context.socket(
                zmq.PUB, socket_class=AsyncSocket
            )
            pub_port = find_free_port((10000, 30000), host=hostname)
            if pub_port is None:
                pub_port = "0"
                self.logger.warning("Couldn't find any free port, using 0")
            pub_address = f"{hostname}:{pub_port}"
            max_attempts = 10
            for attempt in range(max_attempts):
                try:
                    self._pub_socket.bind(f"tcp://{pub_address}")
                    break
                except zmq.error.ZMQError as e:
                    if (
                        "Address already in use" in str(e)
                        and attempt < max_attempts - 1
                    ):
                        pub_port = random.randint(30000, 40000)
                        pub_address = f"{hostname}:{pub_port}"
                    else:
                        raise
            self.logger.info(f"PUB socket bound to {pub_address}")

            self.logger.info(f"Sync location: {self.sync_location}")
            try:
                os.makedirs(self._ckpt_dir, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=self._ckpt_dir, delete=False
                ) as temp:
                    temp.write(
                        f"{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}\n"
                        f"{pub_address}\n"
                    )
                    temp.flush()
                    os.fsync(temp.fileno())
                    temp_path = temp.name
            except Exception as e:
                self.logger.error(f"Writing temp file failed with exception {e}")
                raise e
            
            try:
                os.replace(temp_path, self.sync_location)
                self.logger.info(f"Wrote the sync file to {self.sync_location}")
            except Exception as e:
                self.logger.error(f"os.replace failed with error: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise
        else:
            start = time.perf_counter()
            while time.perf_counter() - start < self.sync_timeout:
                if os.path.exists(self.sync_location):
                    break
                time.sleep(1.0)

            try:
                with open(self.sync_location, "r") as f:
                    lines = f.read().strip().splitlines()
            except Exception as e:
                self.logger.error(f"Reading sync location failed with Exception {e}")
                raise

            master_addr, master_port = lines[0].split(":")
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port

            pub_address = lines[1]
            self._zmq_context = AsyncContext()
            self._sub_socket = self._zmq_context.socket(
                zmq.SUB, socket_class=AsyncSocket
            )
            self._sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
            self._sub_socket.connect(f"tcp://{pub_address}")
            self.logger.info(f"SUB socket connected to {pub_address}")

        os.environ["LOCAL_RANK"] = str(self._local_rank)
        os.environ["RANK"] = str(self._rank)
        os.environ["WORLD_SIZE"] = str(
            self.tensor_parallel_size * self.pipeline_parallel_size
        )
        self.logger.info("after setting local rank")
        os.environ.pop(self.gpu_selector, None)
        os.environ[self.gpu_selector] = os.environ.get(
            f"AVAILABLE_GPUS_{socket.gethostname()}",
            os.environ.get(
                "AVAILABLE_GPUS",
                ",".join(map(str, list(range(self.tensor_parallel_size)))),
            ),
        )

        self.logger.info(f"{self.gpu_selector}: {os.environ[self.gpu_selector]}")

        vllm_log_dir = get_log_dir("vllm")
        os.makedirs(vllm_log_dir, exist_ok=True)
        _setup_vllm_file_logging(
            f"{vllm_log_dir}/vllm_{self._name}_rank{self._rank}.log"
        )
        from vllm.engine.arg_utils import AsyncEngineArgs  # isort: skip
        from vllm.engine.async_llm_engine import AsyncLLMEngine  # isort: skip
        import vllm.envs as envs  # isort: skip
        import torch  # isort: skip

        torch.xpu.set_device(self._local_rank)

        _build_success = b"0"
        _build_failed = b"1"
        if self._use_cached_modelinfo:
            os.environ["VLLM_CACHE_ROOT"] = self._model_info_cache
        else:
            if self._rank == 0:
                if self._model_info_cache is None:
                    self._model_info_cache = os.path.join(
                        self._ckpt_dir, f"vllm_cache_{uuid.uuid4().hex[:6]}"
                    )
                os.environ["VLLM_CACHE_ROOT"] = self._model_info_cache
                try:
                    os.makedirs(self._model_info_cache)
                    _build_model_cache(logger=self.logger.getChild("BuildModelInfo"))
                    await self._pub_socket.send(_build_success)
                    await self._pub_socket.send(
                        cloudpickle.dumps(self._model_info_cache)
                    )
                except Exception as e:
                    await self._pub_socket.send(_build_failed)
                    self.logger.error(f"Building model infos failed with error {e}")
                    raise
            else:
                success = await self._sub_socket.recv()
                if success == _build_failed:
                    raise
                else:
                    frame = await self._sub_socket.recv()
                    self._model_info_cache = cloudpickle.loads(frame)

            os.environ["VLLM_CACHE_ROOT"] = self._model_info_cache

        snapshots = glob(
            f"{self.cache_dir}/hub/models--{self._model_name.replace('/', '--')}/snapshots/*"
        )
        if len(snapshots) == 0:
            self.logger.error(f"No snapshots found in {self.cache_dir}")
            raise RuntimeError

        self.logger.info(f"model: {snapshots[0]}")
        self.logger.info(f"VLLM_CACHE_ROOT:{envs.VLLM_CACHE_ROOT}")
        self.logger.info(
            f"{self._rank},{self._local_rank},{os.environ[self.gpu_selector]}"
        )
        try:
            engine_args = AsyncEngineArgs(
                model=snapshots[0],
                trust_remote_code=True,
                distributed_executor_backend="external_launcher",
                seed=1,
                **self.llm_kwargs,
            )
            self._engine = AsyncLLMEngine.from_engine_args(engine_args)
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
            self._engine.shutdown()
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
        from vllm.sampling_params import RequestOutputKind
        from vllm.utils import random_uuid

        sp = SamplingParams(
            **sampling_params, output_kind=RequestOutputKind.FINAL_ONLY
        )

        if isinstance(prompts, str):
            prompts = [prompts]
            single = True
        else:
            single = False

        self.logger.info(f"Invoking engine with {prompts} {sp} {kwargs}")
        try:
            async def _single_generate(prompt):
                final = None
                async for out in self._engine.generate(
                    prompt, sp, request_id=random_uuid()
                ):
                    final = out
                return final.outputs[0].text

            results = await asyncio.gather(*[_single_generate(p) for p in prompts])
            results = list(results)

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
        **kwargs,
    ):
        PublicActor.__init__(self, name, transport, max_workers=1, run_in_executor=False, **kwargs)
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
        gpu_selector: str = "ZE_AFFINITY_MASK",
        **kwargs,
    ):
        PrivateActor.__init__(self, name, client_conn, max_workers=1, run_in_executor=False, **kwargs)
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
            gpu_selector=gpu_selector,
        )

    async def _setup_rank0_connection(self):
        await self._conn.open()
        self.logger.info("Rank 0: transport ready.")
