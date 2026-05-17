import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from glob import glob

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

    def on_start(self):
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
        self.model = model
        self.cache_dir = cache_dir
        self.port = port
        self.tensor_parallel_size = tensor_parallel_size
        self._server_process = None

    def on_start(self):
        if self.logger is None:
            self.logger = setup_logger(name=self._name, log_dir=f"{os.getcwd()}/logs")
        script_path = os.path.join(os.path.dirname(__file__), "start_vllm_server.sh")
        hostname = socket.gethostname()
        self._server_process = subprocess.Popen(
            [script_path, str(self.port), str(self.tensor_parallel_size),
             self.model, self.cache_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.logger.info(f"Started vLLM server process (pid={self._server_process.pid})")
        url = f"http://{hostname}:{self.port}/v1/models"
        start = time.time()
        timeout = 600
        while time.time() - start < timeout:
            try:
                urllib.request.urlopen(url, timeout=5)
                self.logger.info(f"vLLM server ready at {hostname}:{self.port}")
                return
            except Exception:
                self.logger.info(f"Waiting for vLLM ({time.time() - start:.0f}s)...")
                time.sleep(10)
        raise RuntimeError(f"vLLM server not ready after {timeout}s")

    def on_stop(self):
        if self._server_process is not None:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
                self._server_process.wait()

    @action
    def get_address(self):
        return f"{socket.gethostname()}:{self.port}"
