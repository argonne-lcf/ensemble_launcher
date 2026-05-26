import asyncio
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

import cloudpickle

from ensemble_launcher.ensemble.actor import PublicActor, action
from ensemble_launcher.logging import setup_logger


class VLLMInference(PublicActor):
    def __init__(
        self,
        name: str,
        model: str,
        cache_dir: str,
        tensor_parallel_size: int = 1,
        transport: str = "zmq",
        cache_modelinfo: bool = False,
        ckpt_dir: str = f"{os.getcwd()}/.actor_ckpt",
    ):
        super().__init__(name, transport, ckpt_dir=ckpt_dir)
        self.model = model
        self.cache_dir = cache_dir
        self.tensor_parallel_size = tensor_parallel_size
        self._llm = None
        self._cache_modelinfo = cache_modelinfo

    async def on_start(self):
        if self.logger is None:
            self.logger = setup_logger(name=self._name, log_dir=f"{os.getcwd()}/logs")
        if self._llm is None:
            if self._cache_modelinfo:
                os.environ["VLLM_CACHE_ROOT"] = self.cache_dir
            from vllm import LLM

            snapshots = glob(
                f"{self.cache_dir}/hub/models--{self.model.replace('/', '--')}/snapshots/*"
            )
            self.logger.info(f"model: {snapshots[0]}")
            try:
                self._llm = LLM(
                    model=snapshots[0],
                    tensor_parallel_size=self.tensor_parallel_size,
                    trust_remote_code=True,
                )
            except Exception as e:
                self.logger.error(f"Starting LLM failed with Exception: {e}")
                raise RuntimeError(str(e))
            self.logger.info("init done!")

    @action
    def generate(self, prompts="hello", temperature=0.0, max_tokens=1024):
        from vllm import SamplingParams

        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

        if isinstance(prompts, str):
            prompts = [prompts]
            single = True
        else:
            single = False

        outputs = self._llm.generate(prompts, sampling_params)
        results = [output.outputs[0].text for output in outputs]

        return results[0] if single else results


class OnlineVLLMInference(PublicActor):
    def __init__(
        self,
        name: str,
        model: str,
        cache_dir: str,
        port: int = 8000,
        tensor_parallel_size: int = 1,
        transport: str = "zmq",
        ckpt_dir: str = f"{os.getcwd()}/.actor_ckpt",
    ):
        super().__init__(name, transport, ckpt_dir=ckpt_dir)
        self._model_name = model
        self.cache_dir = cache_dir
        self.port = port
        self.tensor_parallel_size = tensor_parallel_size
        self._server_process = None

    async def on_start(self):
        if self.logger is None:
            self.logger = setup_logger(name=self._name, log_dir=f"{os.getcwd()}/logs")
        script_path = os.path.join(os.path.dirname(__file__), "start_vllm_server.sh")
        self._hostname = (
            socket.gethostname()
            if ".local" not in socket.gethostname()
            else "localhost"
        )
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


class MultiNodeVLLMInference(PublicActor):
    def __init__(
        self,
        name: str,
        model: str,
        cache_dir: str,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        transport: str = "zmq",
        cache_modelinfo: bool = False,
        sync_location: str = f"file://file_{uuid.uuid4().hex}",
        rank_env: str = "PALS_RANKID",
        local_rank_env: str = "PALS_LOCAL_RANKID",
        sync_timeout: float = 60,
        ckpt_dir: str = f"{os.getcwd()}/.actor_ckpt",
    ):
        super().__init__(name, transport, ckpt_dir=ckpt_dir)
        self._model_name = model
        self.cache_dir = cache_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self._cache_modelinfo = cache_modelinfo
        self.rank_env = rank_env
        self.local_rank_env = local_rank_env
        self.sync_timeout = sync_timeout
        if sync_location.startswith("file://"):
            self.sync_location = os.path.join(
                self.ckpt_dir, sync_location.replace("file://", "")
            )
        else:
            raise ValueError("Unknown sync location prefix")
        self._rank = None
        self._local_rank = None
        self._pub_socket = None
        self._sub_socket = None
        self._zmq_context = None
        self._llm = None

    async def on_start(self):
        if self.logger is None:
            self.logger = setup_logger(name=self._name, log_dir=f"{os.getcwd()}/logs")

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

        hostname = socket.gethostname()

        if self._rank == 0:
            os.environ["MASTER_ADDR"] = hostname
            os.environ["MASTER_PORT"] = str(random.randint(20000, 30000))

            self._zmq_context = AsyncContext()
            self._pub_socket = self._zmq_context.socket(
                zmq.PUB, socket_class=AsyncSocket
            )
            pub_port = random.randint(30000, 40000)
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

            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.ckpt_dir, delete=False
            ) as temp:
                temp.write(
                    f"{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}\n"
                    f"{pub_address}\n"
                )
                temp.flush()
                os.fsync(temp.fileno())
                temp_path = temp.name
            try:
                os.replace(temp_path, self.sync_location)
            except Exception:
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

        if self._cache_modelinfo:
            os.environ["VLLM_CACHE_ROOT"] = self.cache_dir
        from vllm import LLM

        snapshots = glob(
            f"{self.cache_dir}/hub/models--{self._model_name.replace('/', '--')}/snapshots/*"
        )
        self.logger.info(f"model: {snapshots[0]}")
        try:
            self._llm = LLM(
                model=snapshots[0],
                tensor_parallel_size=self.tensor_parallel_size,
                pipeline_parallel_size=self.pipeline_parallel_size,
                trust_remote_code=True,
                distributed_executor_backend="external_launcher",
                seed=1,
            )
        except Exception as e:
            self.logger.error(f"Starting LLM failed with Exception: {e}")
            raise RuntimeError(str(e))
        self.logger.info(f"Rank {self._rank}: vLLM init done!")

    async def _run(self):
        self._init_runtime()
        await self.on_start()

        if self._rank == 0:
            self._start_transport()
            await self._conn.open()
            os.makedirs(self.ckpt_dir, exist_ok=True)
            fname = f"{self.ckpt_dir}/{self.name}.ckpt"
            with open(fname, "w") as f:
                f.write(self._conn.get_state().serialize())
            self.logger.info("Rank 0: ROUTER transport ready, checkpoint written.")

        await asyncio.gather(self._recv(), self._send(), self._main_loop())
        await self.on_stop()

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
        if self._pub_socket is not None:
            self._pub_socket.close()
        if self._sub_socket is not None:
            self._sub_socket.close()
        if self._zmq_context is not None:
            self._zmq_context.term()

    @action
    def generate(self, prompts="hello", temperature=0.0, max_tokens=1024):
        from vllm import SamplingParams

        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

        if isinstance(prompts, str):
            prompts = [prompts]
            single = True
        else:
            single = False

        outputs = self._llm.generate(prompts, sampling_params)
        results = [output.outputs[0].text for output in outputs]

        return results[0] if single else results
